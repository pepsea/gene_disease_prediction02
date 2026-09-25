"""同じ中身（M1 の3条件）を、答えさせ方だけ変えて比べる検証スクリプト（GGUF 直接駆動）。

  scale : 1遺伝子ずつ「3条件にどのくらい当てはまるか」を 0〜9 で答えさせ、数字の期待値を点数にする。
          逆向き（0 = 当てはまる）でも聞くが、実機（txgemma-9b）では逆向きの指示が無視されて順位が逆転した（AUC 0.29〜0.39）。
          評価には正向きの scale_norm を使う（scale は両向きの平均で、参考として残す）。
  pair  : 2遺伝子を並べ「どちらがよく当てはまるか」を 1/2 で答えさせる。左右を入れ替えた2回を平均し、勝つ確率の平均を点数にする。
  worst : 5遺伝子（「該当なし」なし）から「候補リストから1つ外すならどれか」を選ばせる（best-worst scaling の worst 側）。選ばれた確率の平均の符号を反転して点数にする。
          グループ分けは rank_variants.py と同じ乱数・同じ手順なので、best 側（M1r）と組み合わせられる。

数字は「Answer: 」（末尾の空白まで）の直後で読む（Gemma は空白付きの数字トークンを持たないため。rank_variants.py 参照）。

使い方:
  python scripts/ask_modes.py --disease achondroplasia ra prostate_cancer scz cystinuria
  python scripts/ask_modes.py --disease ra --max-genes 10 --modes scale      # 試運転
出力: outputs/<疾患>_set100_askmodes.csv（遺伝子 × 方式の点数）
"""
import argparse, glob, json, math, os, random, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from yesno_question_variants import ROOT, STRICT_NOTE as YN_NOTE, load_genes
from rank_variants import STRICT_NOTE as RANK_NOTE

CRITERIA = ("Consider these three criteria:\n"
            "(a) A hypothesis can be formulated that a therapeutic drug targeting the gene would treat {disease} or the symptoms of {disease}.\n"
            "(b) Even if the gene is not part of the pathway that causes {disease}, activating or inhibiting it could counteract or compensate "
            "for the abnormal process described above, for example through a parallel or opposing pathway in the same cells.\n"
            "(c) The gene's own specific role — its substrate, ligand, signalling pathway, cell type or circuit — matches the mechanism described "
            "above precisely, rather than a related but distinct one (e.g. a different molecule, cell type, tissue or subcellular compartment); "
            "a similar-sounding but distinct role, even in the same gene family, does not count.\n")
ROLE = "You are an expert in drug discovery and human disease biology.\n"
GROUP_SIZE, PAIR_ROUNDS, WORST_ROUNDS = 5, 4, 8


def disease_block(disease, info):
    return f"Disease: {disease}\nTarget symptoms and the organ, cell and functional abnormalities behind them:\n" + "\n".join(f"- {b}" for b in info) + "\n"


def scale_prefix(disease, info, reverse):
    lo, hi = ("the gene clearly meets at least one criterion", "the gene meets none of the criteria") if reverse else \
             ("the gene meets none of the criteria", "the gene clearly meets at least one criterion")
    return (ROLE + YN_NOTE + disease_block(disease, info) +
            f"Rate each gene on a scale from 0 to 9, where 0 means {lo} and 9 means {hi}. Answer with a single digit only.\n\n")


def pair_prefix(disease, info):
    return (ROLE + "You will be shown two candidate genes for one disease, and asked which ONE number is the better answer.\n" + RANK_NOTE +
            disease_block(disease, info) + "Answer with a single number, 1 or 2, only. No words, no explanation.\n\n")


def worst_prefix(disease, info):
    return (ROLE + f"You will be shown a numbered list of {GROUP_SIZE} candidate genes for one disease, and asked which ONE number is the "
            "best answer to a question.\n" + RANK_NOTE + disease_block(disease, info) +
            f"Answer with a single number from 1 to {GROUP_SIZE} only. No words, no explanation.\n\n")


class Engine:
    def __init__(self, model_path, n_ctx=2048):
        from llama_cpp import Llama
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=-1, logits_all=False, verbose=False)
        self.digit = {n: self.tok(str(n)) for n in range(10)}
        assert all(len(v) == 1 for v in self.digit.values()), "数字が1トークンになっていません"
        self.states = {}

    def tok(self, text, bos=False):
        return self.llm.tokenize(text.encode("utf-8"), add_bos=bos, special=bos)

    def set_prefix(self, name, text):
        self.llm.reset(); self.llm.eval(self.tok(text, bos=True))
        self.states[name] = self.llm.save_state()

    def digits(self, state, text, options):
        """前置きの状態を復元 → text を処理 → options（数字）の中だけで正規化した確率。"""
        llm = self.llm
        llm.reset(); llm.load_state(self.states[state])
        llm.eval(self.tok(text))
        lg = np.ctypeslib.as_array(llm._ctx.get_logits(), shape=(llm.n_vocab(),)).astype(np.float64)
        l = np.array([lg[self.digit[o][0]] for o in options])
        p = np.exp(l - l.max()); return p / p.sum()


def groups(idx, rounds, size, rng):
    """rank_variants.py と同じ手順（シャッフル → 先頭から切り出し → 端数は他から補充）。"""
    for r in range(rounds):
        order = idx[:]; rng.shuffle(order)
        for start in range(0, len(order), size):
            chunk = order[start:start + size]
            if len(chunk) < size: chunk = chunk + rng.sample([i for i in idx if i not in chunk], size - len(chunk))
            yield r, chunk


def run_disease(eng, key, reg, max_genes, model_name, modes):
    D = reg[key]; disease, info = D["name"], D["info"][:5]
    genes = load_genes(D["gene_prefix"], max_genes)
    crit = CRITERIA.format(disease=disease)
    out = genes[["symbol", "category"]].copy()
    idx, t0 = list(genes.index), time.time()
    if "scale" in modes:
        for rev in (False, True):
            eng.set_prefix(f"scale{rev}", scale_prefix(disease, info, rev))
        q = f"Question: {crit}How well does this gene meet at least one of these criteria? Answer: "
        for rev, col in ((False, "scale_norm"), (True, "scale_rev")):
            ev = []
            for i in idx:
                p = eng.digits(f"scale{rev}", f"Gene: {genes.at[i, 'gene_label']}\n" + q, range(10))
                e = float((p * np.arange(10)).sum()); ev.append(9 - e if rev else e)
            out[col] = ev
        out["scale"] = (out["scale_norm"] + out["scale_rev"]) / 2
        print(f"  {key} scale done ({time.time() - t0:.0f}s)", flush=True)
    if "pair" in modes:
        eng.set_prefix("pair", pair_prefix(disease, info))
        q = f"Question: {crit}Which numbered gene better meets at least one of these criteria? Answer: "
        s, n = {i: 0.0 for i in idx}, {i: 0 for i in idx}
        for _, (a, b) in groups(idx, PAIR_ROUNDS, 2, random.Random(1)):
            la, lb = genes.at[a, "gene_label"], genes.at[b, "gene_label"]
            p_ab = eng.digits("pair", f"Candidate genes:\n1. {la}\n2. {lb}\n" + q, (1, 2))
            p_ba = eng.digits("pair", f"Candidate genes:\n1. {lb}\n2. {la}\n" + q, (1, 2))
            pa = (p_ab[0] + p_ba[1]) / 2
            s[a] += pa; s[b] += 1 - pa; n[a] += 1; n[b] += 1
        out["pair"] = [s[i] / n[i] for i in idx]
        print(f"  {key} pair done ({time.time() - t0:.0f}s)", flush=True)
    if "worst" in modes:
        eng.set_prefix("worst", worst_prefix(disease, info))
        q = (f"Question: If you had to remove ONE numbered gene above from a list of candidate drug targets for {disease} because it is "
             "the least relevant, which would you remove? Answer: ")                  # 比較用の聞き方（M1 の3条件は使わない）
        s, n = {i: 0.0 for i in idx}, {i: 0 for i in idx}
        for _, chunk in groups(idx, WORST_ROUNDS, GROUP_SIZE, random.Random(0)):
            block = "Candidate genes:\n" + "\n".join(f"{j + 1}. {genes.at[i, 'gene_label']}" for j, i in enumerate(chunk)) + "\n"
            p = eng.digits("worst", block + q, range(1, GROUP_SIZE + 1))
            for j, i in enumerate(chunk): s[i] += p[j]; n[i] += 1
        out["worst_p"] = [s[i] / n[i] for i in idx]
        out["worst"] = -out["worst_p"]
        print(f"  {key} worst done ({time.time() - t0:.0f}s)", flush=True)
    out["model"] = model_name; out["disease"] = disease
    path = os.path.join(ROOT, "outputs", f"{D['gene_prefix']}_set100_askmodes{'_test' if max_genes else ''}.csv")   # 試運転は別ファイル
    if os.path.exists(path) and max_genes is None:                 # 一部の方式だけ聞いたときは既存の列を残す
        old = pd.read_csv(path)
        keep = [c for c in old.columns if c not in out.columns]
        if keep: out = out.merge(old[["symbol"] + keep], on="symbol", how="left")
    out.to_csv(path, index=False)
    print(f"{key}: {len(genes)} genes, modes {modes} in {time.time() - t0:.0f}s -> {path}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", nargs="+", default=["achondroplasia", "ra", "prostate_cancer", "scz", "cystinuria"])
    ap.add_argument("--modes", nargs="+", default=["scale", "worst"])
    ap.add_argument("--max-genes", type=int, default=None)
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    reg = json.load(open(os.path.join(ROOT, "data", "diseases.json"), encoding="utf-8"))
    model = a.model or sorted(glob.glob(os.path.expanduser("~/llm/models/**/*txgemma*.gguf"), recursive=True))[0]
    print("model:", model, flush=True)
    eng = Engine(model)
    for key in a.disease:
        run_disease(eng, key, reg, a.max_genes, os.path.basename(model), a.modes)


if __name__ == "__main__":
    main()
