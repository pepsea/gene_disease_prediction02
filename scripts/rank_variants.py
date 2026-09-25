"""相対評価（ノートブック 04 の強制選択）で、質問パターンを比べる検証スクリプト（GGUF 直接駆動）。

ノートブック 04 と同じ前置き・番号付きリスト（5遺伝子 + 「該当なし」）で、全遺伝子をシャッフルして重複なく
グループ分けするラウンドを N 回くり返す。ノートブック 04 と違い、各質問パターンは「前置き + 遺伝子リスト」の直後に
1問ずつ独立に聞く（前の答えで条件付けしない）。全パターンで同じグループ分けを使うので、差は質問文だけから来る。

使い方:
  python scripts/rank_variants.py --disease achondroplasia ra prostate_cancer scz cystinuria
  python scripts/rank_variants.py --disease ra --rounds 2 --max-genes 20        # 試運転
  python scripts/rank_variants.py --report-only
  python scripts/rank_variants.py --variants pick1 effect link expert   # 一部のパターンだけ聞いて既存の CSV に足す
出力: outputs/<疾患>_set100_rankvar.csv（ラウンド × グループ × パターン × 位置 の選択確率）と、標準出力の AUC 表
"""
import argparse, glob, json, math, os, random, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from yesno_question_variants import ROOT, auc, load_genes

VARIANTS = {
    # --- ノートブック 04 の既存3パターン（文面そのまま） ---
    "target":    "Which numbered gene above is the single most plausible drug target for {disease}? Weigh all forms of evidence together "
                 "(existing drugs, human genetics, and mechanism).",
    "mechanism": "Which numbered gene above plays the most central role in causing {disease}, based on its OWN specific substrate, "
                 "signalling pathway, cell type or circuit — as opposed to a related but functionally or anatomically distinct one?",
    "evidence":  "Which numbered gene above has the strongest existing evidence for {disease} — an approved drug target, or human genetic "
                 "variants that cause {disease} or alter its risk or severity?",
    # --- Yes/No の検証で良かった質問を「どれか1つ選ぶ」形に移したもの ---
    "H3r": "For which numbered gene above can the most plausible hypothesis be formulated that a therapeutic drug targeting it would "
           "treat {disease} or the symptoms of {disease}?",
    "R1r": "Which numbered gene above, even if it is not part of the pathway that causes {disease}, could best counteract or compensate "
           "for the abnormal process described above when activated or inhibited, for example through a parallel or opposing pathway "
           "in the same cells?",
    "M1r": "Consider these three criteria:\n"
           "(a) A hypothesis can be formulated that a therapeutic drug targeting the gene would treat {disease} or the symptoms of {disease}.\n"
           "(b) Even if the gene is not part of the pathway that causes {disease}, activating or inhibiting it could counteract or compensate "
           "for the abnormal process described above, for example through a parallel or opposing pathway in the same cells.\n"
           "(c) The gene's own specific role — its substrate, ligand, signalling pathway, cell type or circuit — matches the mechanism described "
           "above precisely, rather than a related but distinct one (e.g. a different molecule, cell type, tissue or subcellular compartment); "
           "a similar-sounding but distinct role, even in the same gene family, does not count.\n"
           "Which numbered gene above best meets at least one of these criteria?",
    # --- 比較（5遺伝子から選ぶ）ならではの聞き方。M1 の3条件は Yes/No 用なので、ここでは使わない ---
    "pick1":   "If only ONE of the numbered genes above could be pursued as a drug target for {disease}, which would you choose?",
    "effect":  "Inhibiting or activating which numbered gene above would be expected to improve the symptoms listed above the most?",
    "link":    "Which numbered gene above has the most direct biological connection to the abnormal process described above, whether as "
               "its cause, a key mediator, or a pathway that could compensate for it?",
    "expert":  "Which numbered gene above would an expert in {disease} most likely name as a promising therapeutic target?",
    # --- 「治療・症状改善の仮説を構築できる遺伝子は？」（ユーザー案）。H3r（治療の仮説のみ）との違いは、症状の改善も含めること ---
    "hyp":     "For which numbered gene above can a hypothesis be constructed that inhibiting or activating it would treat {disease} "
               "or improve its symptoms?",
    "hyp_best": "For which numbered gene above can the most convincing hypothesis be constructed that inhibiting or activating it would "
                "treat {disease} or improve at least one of the symptoms listed above?",
    # --- M1r の条件 (a) を「症状の改善」に広げた版（Yes/No の M2 に対応） ---
    "M2r": "Consider these three criteria:\n"
           "(a) A hypothesis can be constructed that inhibiting or activating the gene would treat {disease} or improve at least one of the "
           "symptoms listed above.\n"
           "(b) Even if the gene is not part of the pathway that causes {disease}, activating or inhibiting it could counteract or compensate "
           "for the abnormal process described above, for example through a parallel or opposing pathway in the same cells.\n"
           "(c) The gene's own specific role — its substrate, ligand, signalling pathway, cell type or circuit — matches the mechanism described "
           "above precisely, rather than a related but distinct one (e.g. a different molecule, cell type, tissue or subcellular compartment); "
           "a similar-sounding but distinct role, even in the same gene family, does not count.\n"
           "Which numbered gene above best meets at least one of these criteria?",
}
VIDS = list(VARIANTS)
WORST = set()                                                          # 最悪側の質問は「該当なし」と相性が悪いので ask_modes.py の worst で聞く
GROUP_SIZE = 5

STRICT_NOTE = ("Note: the vast majority of human genes are NOT drug targets for any given disease. Pick a gene only when there is a clear, "
               "specific reason; membership in the same gene family as a known disease gene is NOT sufficient by itself.\n")


def prefix_text(disease, info):
    bullets = "\n".join(f"- {b}" for b in info)
    n_opt = GROUP_SIZE + 1
    return ("You are an expert in drug discovery and human disease biology. You will be shown a numbered list of "
            f"{GROUP_SIZE} candidate genes for one disease, and asked which ONE number is the best answer to a question.\n" + STRICT_NOTE +
            f"Disease: {disease}\nTarget symptoms and the organ, cell and functional abnormalities behind them:\n{bullets}\n"
            f"Answer with a single number from 1 to {n_opt} only. No words, no explanation.\n\n")


def group_block(labels):
    lines = [f"{i + 1}. {g}" for i, g in enumerate(labels)] + [f"{len(labels) + 1}. None of the above genes seem clearly relevant"]
    return "Candidate genes:\n" + "\n".join(lines) + "\n"


class Engine:
    def __init__(self, model_path, n_ctx=2048):
        from llama_cpp import Llama
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=-1, logits_all=False, verbose=False)
        self.opt = {n: self.ids([str(n), f" {n}"]) for n in range(1, GROUP_SIZE + 2)}

    def tok(self, text, bos=False):
        return self.llm.tokenize(text.encode("utf-8"), add_bos=bos, special=bos)

    def ids(self, spellings):
        out = []
        for s in spellings:
            t = self.tok(s)
            if len(t) == 1 and t[0] not in out: out.append(t[0])
        return out

    def set_prefix(self, text):
        self.llm.reset(); self.llm.eval(self.tok(text, bos=True))
        self.state = self.llm.save_state()

    def score(self, labels, question_lines):
        """前置き → 遺伝子リスト の位置に毎回巻き戻して、各パターンを独立に聞く。return {vid: [p_1 .. p_none]}"""
        llm = self.llm
        llm.reset(); llm.load_state(self.state)
        llm.eval(self.tok(group_block(labels)))
        base = llm.n_tokens
        out = {}
        for vid, line in question_lines.items():
            llm.n_tokens = base                                   # eval() の先頭で base 以降の KV が消される
            llm.eval(self.tok(line))
            lg = np.ctypeslib.as_array(llm._ctx.get_logits(), shape=(llm.n_vocab(),)).astype(np.float64)
            lg -= lg.max(); lp = lg - math.log(np.exp(lg).sum())
            lse = lambda v: max(v) + math.log(sum(math.exp(x - max(v)) for x in v))
            l = np.array([lse([lp[i] for i in self.opt[n]]) for n in range(1, GROUP_SIZE + 2)])
            p = np.exp(l - l.max()); out[vid] = p / p.sum()
        return out


def run_disease(eng, key, reg, rounds, max_genes, model_name, vids):
    D = reg[key]; disease, info = D["name"], D["info"][:5]
    genes = load_genes(D["gene_prefix"], max_genes)
    eng.set_prefix(prefix_text(disease, info))
    # 末尾の空白が必要：Gemma は「 1」のような空白付き数字トークンを持たないため、「Answer:」の直後では空白（約85%）が予測され、
    # 数字の確率は内容と無関係なノイズになる（番号の癖として現れる）。「Answer: 」まで入れると数字が 90% 以上を占める。
    lines = {v: f"Question: {VARIANTS[v].format(disease=disease)} Answer: " for v in vids}
    rng, idx, rows, t0 = random.Random(0), list(genes.index), [], time.time()
    for r in range(rounds):
        order = idx[:]; rng.shuffle(order)
        for gi, start in enumerate(range(0, len(order), GROUP_SIZE)):
            chunk = order[start:start + GROUP_SIZE]
            if len(chunk) < GROUP_SIZE: chunk = chunk + rng.sample([i for i in idx if i not in chunk], GROUP_SIZE - len(chunk))
            probs = eng.score([genes.at[i, "gene_label"] for i in chunk], lines)
            for vid, p in probs.items():
                for slot, i in enumerate(chunk):
                    rows.append({"round": r, "group": gi, "variant": vid, "position": slot + 1, "symbol": genes.at[i, "symbol"],
                                 "category": genes.at[i, "category"], "p": round(float(p[slot]), 6), "p_none": round(float(p[-1]), 6),
                                 "win": int(np.argmax(p) == slot)})
        print(f"  {key} round {r + 1}/{rounds} ({time.time() - t0:.0f}s)", flush=True)
    long = pd.DataFrame(rows); long["model"] = model_name; long["disease"] = disease
    out = os.path.join(ROOT, "outputs", f"{D['gene_prefix']}_set100_rankvar{'_test' if max_genes else ''}.csv")   # 試運転は別ファイル
    if os.path.exists(out) and max_genes is None:                  # 一部のパターンだけ聞いたときは既存の他パターンを残す（グループ分けは同じ乱数なので揃う）
        old = pd.read_csv(out)
        long = pd.concat([old[~old["variant"].isin(vids)], long], ignore_index=True)
    long.to_csv(out, index=False)
    print(f"{key}: {len(genes)} genes x {rounds} rounds x {len(vids)} variants in {time.time() - t0:.0f}s -> {out}", flush=True)
    return long


def gene_scores(long):
    """遺伝子ごとのスコア：mean_p（選ばれた確率の平均）、win_rate（1位だった割合）、lr_none（log p − log p_none の平均。『該当なし』より上か）。"""
    d = long.assign(lr=np.log(np.clip(long["p"], 1e-9, 1)) - np.log(np.clip(long["p_none"], 1e-9, 1)))
    w = d["variant"].isin(WORST)                                   # 最悪側：1〜5番だけで正規化し、符号を反転（大きいほど標的らしい、にそろえる）
    d.loc[w, "p"] = -d.loc[w, "p"] / (1 - d.loc[w, "p_none"]).clip(lower=1e-9)
    d.loc[w, "win"] = -d.loc[w, "win"]; d.loc[w, "lr"] = d.loc[w, "p"]
    return d.groupby(["variant", "symbol", "category"]).agg(mean_p=("p", "mean"), win_rate=("win", "mean"), lr_none=("lr", "mean")).reset_index()


def report(long, title):
    g = gene_scores(long)
    rows = []
    for vid in g["variant"].unique():
        s = g[g.variant == vid]; k, r, c = s.category.eq("known"), s.category.eq("random"), s.category.eq("candidate")
        for col in ("mean_p", "win_rate", "lr_none"):
            rows.append({"variant": vid, "score": col, "AUC k/rand": auc(s.loc[k, col], s.loc[r, col]), "AUC k/rest": auc(s.loc[k, col], s.loc[~k, col]),
                         "AUC cand/rand": auc(s.loc[c, col], s.loc[r, col])})
    print(f"\n===== {title}")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    pos = long.groupby(["variant", "position"])["p"].mean().unstack().round(3)
    print("位置ごとの平均確率（均等なら 1/6 ≈ 0.167 前後。内容と無関係な番号の癖を見る）:\n" + pos.to_string())
    nn = long.drop_duplicates(["round", "group", "variant"]).groupby("variant")["p_none"].mean().round(3)
    print("「該当なし」の平均確率:\n" + nn.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", nargs="+", default=["achondroplasia", "ra", "prostate_cancer", "scz", "cystinuria"])
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--max-genes", type=int, default=None)
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    reg = json.load(open(os.path.join(ROOT, "data", "diseases.json"), encoding="utf-8"))
    if a.report_only:
        for key in a.disease:
            report(pd.read_csv(os.path.join(ROOT, "outputs", f"{reg[key]['gene_prefix']}_set100_rankvar.csv")), key)
        return
    model = a.model or sorted(glob.glob(os.path.expanduser("~/llm/models/**/*txgemma*.gguf"), recursive=True))[0]
    print("model:", model, flush=True)
    eng = Engine(model)
    for key in a.disease:
        report(run_disease(eng, key, reg, a.rounds, a.max_genes, os.path.basename(model), a.variants or VIDS), key)


if __name__ == "__main__":
    main()
