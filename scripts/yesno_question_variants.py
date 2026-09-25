"""Yes/No 採点を、何種類かの質問で独立に聞いて比べる検証スクリプト（GGUF 直接駆動）。

ノートブック 03 と同じ前置き（STRICT_NOTE 付き）・遺伝子名で、各質問を
「前置き + 遺伝子ブロック」の直後に 1 問ずつ独立に聞く（前の答えで条件付けしない）。
順序の癖は全遺伝子で両順（Yes or No / No or Yes）を聞き、対数オッズの平均で打ち消す。

使い方:
  python scripts/yesno_question_variants.py --disease scz achondroplasia
  python scripts/yesno_question_variants.py --disease ra --max-genes 10     # 試運転
出力: outputs/<疾患>_set100_qvariants.csv（遺伝子 × 質問 × 順序の p_yes）と、標準出力の AUC 表
"""
import argparse, glob, json, math, os, time
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# (id, 向き, 質問文)  向き -1 は「Yes = 標的ではない」なので 1 - p を採点に使う
QUESTIONS = [
    ("V0", +1, "Could activating or inhibiting this gene plausibly improve {disease} or its symptoms, i.e. is it a plausible drug target for {disease}?"),
    ("A1", +1, "Is this gene the direct target of a drug — approved, or currently in clinical trials — that is being developed "
               "specifically to treat {disease} (not some other, unrelated disease)?"),
    ("A2", +1, "Do human genetic variants in this gene, rare or common, cause {disease} or alter its risk or severity?"),
    ("B1", +1, "Does this gene's own specific role — its substrate, ligand, signalling pathway, cell type or circuit — match the "
               "mechanism described above precisely, rather than a related but distinct one (e.g. a different molecule, cell type, "
               "tissue or subcellular compartment)? Answer No for a similar-sounding but distinct role, even in the same gene family."),
    ("B2", +1, "In the affected cell types listed above, is this gene a principal driver of the abnormal function that produces the "
               "symptoms? Answer No if the gene is merely expressed there or has only a housekeeping or bystander role."),
    ("B3", +1, "Would inhibiting or activating this gene be expected to improve at least one of the symptoms listed above?"),
    ("B4", +1, "Does this gene directly interact with, or lie immediately upstream or downstream of, established targets or causal genes of {disease}?"),
    ("H1", +1, "Could a specific, biologically coherent therapeutic hypothesis be constructed for {disease} using this gene — i.e. a "
               "rationale for why modulating it might help — even if the exact mechanism differs from the primary one described above? "
               "Answer No only if you cannot articulate any coherent rationale connecting this gene to the disease at all."),
    ("H2", +1, "Would modulating (blocking or activating) this gene's protein product plausibly influence the abnormal function described "
               "above, even if this gene only receives or transmits a signal rather than being the original driver of that signal?"),
    ("H3", +1, "Can a hypothesis be formulated that a therapeutic drug targeting the gene {gene} would treat {disease} or the symptoms of {disease}?"),
    ("N1", -1, "Is this gene irrelevant to {disease}, such that inhibiting or activating it would have no effect on any of the symptoms listed above?"),
    # --- 追加（仮説・機構・KO/KI の説明） ---
    ("P1", +1, "Can you confidently propose the gene {gene} as a drug target for {disease}?"),
    ("P2", +1, "Can the disease mechanism of {disease} be explained by an abnormality of the gene {gene}?"),
    ("P3", +1, "Can you explain how knocking out or knocking in the gene {gene} could plausibly improve {disease} and the symptoms of {disease}?"),
    ("P4", +1, "Can {disease} and its symptoms be relieved by inhibiting or activating the gene {gene}?"),
    # --- レスキュー（取りこぼした既知遺伝子の救済）：R1 = 並行・拮抗経路による代償（NPR2 など）、R2 = 症状の一部を改善する仮説、R3 = 分子機構を正す仮説 ---
    ("R1", +1, "Even if this gene is not part of the pathway that causes {disease}, could activating or inhibiting it counteract or "
               "compensate for the abnormal process described above, for example through a parallel or opposing pathway in the same cells?"),
    ("R2", +1, "Can a hypothesis be formulated that activating or inhibiting the function of the gene {gene} would improve some of "
               "the symptoms caused by {disease}?"),
    ("R3", +1, "Can a hypothesis be formulated that activating or inhibiting the function of the gene {gene} would correct the "
               "molecular mechanism underlying {disease} and its symptoms?"),
    # --- 3問版の改訂（遺伝子を "the gene {gene}" に統一、B1 は病名を入れ、確立した別の機構も照合対象に。H3 は文面そのままなので聞き直さない） ---
    ("R1b", +1, "Even if the gene {gene} is not part of the pathway that causes {disease}, could activating or inhibiting it counteract or "
                "compensate for the abnormal process described above, for example through a parallel or opposing pathway in the same cells?"),
    ("B1b", +1, "Does the gene {gene}'s own specific role — its substrate, ligand, signalling pathway, cell type or circuit — precisely match "
                "a mechanism of {disease}, either the one described above or another well-established mechanism of {disease}, rather than "
                "a related but distinct one (e.g. a different molecule, cell type, tissue or subcellular compartment)? Answer No for a "
                "similar-sounding but distinct role, even in the same gene family."),
    # --- 疾患名も遺伝子名も入れず、前置きの条件（Disease 行と症状の箇条書き）を参照させる版。B1 はもともとこの形なので B1 のまま使う ---
    ("H3c", +1, "Can a hypothesis be formulated that a therapeutic drug targeting this gene would treat the disease described above "
                "or the symptoms listed above?"),
    ("R1c", +1, "Even if this gene is not part of the pathway that causes the disease described above, could activating or inhibiting it "
                "counteract or compensate for the abnormal process described above, for example through a parallel or opposing pathway "
                "in the same cells?"),
    # --- 3条件をまとめて「どれか1つに当てはまるか」を1問で聞く（条件の文面は H3 / R1 / B1 の旧版をそのまま平叙文に） ---
    ("M1", +1, "Consider these three criteria:\n"
               "(a) A hypothesis can be formulated that a therapeutic drug targeting the gene {gene} would treat {disease} or the symptoms of {disease}.\n"
               "(b) Even if this gene is not part of the pathway that causes {disease}, activating or inhibiting it could counteract or compensate "
               "for the abnormal process described above, for example through a parallel or opposing pathway in the same cells.\n"
               "(c) This gene's own specific role — its substrate, ligand, signalling pathway, cell type or circuit — matches the mechanism described "
               "above precisely, rather than a related but distinct one (e.g. a different molecule, cell type, tissue or subcellular compartment); "
               "a similar-sounding but distinct role, even in the same gene family, does not count.\n"
               "Does this gene meet at least one of these criteria?"),
    # --- M1 の条件 (a) を「症状の改善」に広げた版（(b)(c) は M1 と同じ文面） ---
    ("M2", +1, "Consider these three criteria:\n"
               "(a) A hypothesis can be constructed that inhibiting or activating the gene {gene} would treat {disease} or improve at least "
               "one of the symptoms listed above.\n"
               "(b) Even if this gene is not part of the pathway that causes {disease}, activating or inhibiting it could counteract or compensate "
               "for the abnormal process described above, for example through a parallel or opposing pathway in the same cells.\n"
               "(c) This gene's own specific role — its substrate, ligand, signalling pathway, cell type or circuit — matches the mechanism described "
               "above precisely, rather than a related but distinct one (e.g. a different molecule, cell type, tissue or subcellular compartment); "
               "a similar-sounding but distinct role, even in the same gene family, does not count.\n"
               "Does this gene meet at least one of these criteria?"),
    # --- M2 の表記をそろえた版（ユーザー案、文法のみ修正）：(a)(b)(c) すべて遺伝子名 {gene}、(c) に病名も入れる ---
    ("M3", +1, "Consider these three criteria:\n"
               "(a) A hypothesis can be constructed that inhibiting or activating {gene} would treat {disease} or improve at least "
               "one of the symptoms listed above.\n"
               "(b) Even if {gene} is not part of the pathway that causes {disease}, activating or inhibiting it could counteract or compensate "
               "for the abnormal process described above, for example through a parallel or opposing pathway in the same cells.\n"
               "(c) {gene}'s own specific role — its substrate, ligand, signalling pathway, cell type or circuit — matches the mechanism of {disease} described "
               "above precisely, rather than a related but distinct one (e.g. a different molecule, cell type, tissue or subcellular compartment); "
               "a similar-sounding but distinct role, even in the same gene family, does not count.\n"
               "Does this gene meet at least one of these criteria?"),
    # --- M3 のコンセプト（(a) 治療・症状改善の仮説 / (b) 代償経路 / (c) 機構の一致 の OR）を保ったまま文章だけ変えた3版 ---
    # M3s：短くした版（各条件を1文に）
    ("M3s", +1, "Consider these three criteria:\n"
                "(a) Inhibiting or activating {gene} could plausibly treat {disease} or improve at least one of the symptoms listed above.\n"
                "(b) Even outside the causal pathway, modulating {gene} could counteract the abnormal process described above through a "
                "parallel or opposing pathway.\n"
                "(c) {gene}'s own substrate, ligand, pathway, cell type or circuit precisely matches the mechanism of {disease} described above "
                "(a similar but distinct role, even in the same gene family, does not count).\n"
                "Does this gene meet at least one of these criteria?"),
    # M3d：説明を詳しくした版（各条件に判断の目安を足す。病気に固有の例は入れない）
    ("M3d", +1, "Consider these three criteria:\n"
                "(a) A hypothesis can be constructed that inhibiting or activating {gene} would treat {disease} or improve at least one of the "
                "symptoms listed above. The hypothesis must state a direction (inhibit or activate) and a plausible biological route from {gene} "
                "to the disease or symptom; a vague association is not enough.\n"
                "(b) Even if {gene} is not part of the pathway that causes {disease}, activating or inhibiting it could counteract or compensate "
                "for the abnormal process described above, for example by boosting a parallel pathway that restores the lost function, or by "
                "dampening the output of an over-active process in the same cells.\n"
                "(c) {gene}'s own specific role — its substrate, ligand, signalling pathway, cell type or circuit — matches the mechanism of "
                "{disease} described above precisely, rather than a related but distinct one (e.g. a different molecule, cell type, tissue or "
                "subcellular compartment). Judge by what {gene} itself does, not by its name or gene family; a similar-sounding but distinct role, "
                "even in the same gene family, does not count.\n"
                "Answer Yes if at least one criterion is clearly met, and No if none is. Does this gene meet at least one of these criteria?"),
    # M3w：言い方を変えた版（「治療の入り口となる3つの道筋」として問う）
    ("M3w", +1, "A gene can be a sound therapeutic entry point for {disease} through any of three routes:\n"
                "(a) Direct benefit: modulating {gene} (inhibiting or activating it) is expected to treat {disease} or relieve at least one of "
                "the symptoms listed above.\n"
                "(b) Compensation: {gene} lies outside the causal pathway, but modulating it could offset the abnormal process described above "
                "through a parallel or opposing pathway in the same cells.\n"
                "(c) Mechanistic match: {gene}'s own substrate, ligand, signalling pathway, cell type or circuit is exactly the mechanism of "
                "{disease} described above, not a similar-sounding one in the same gene family or in a different tissue or compartment.\n"
                "Is {gene} a sound therapeutic entry point for {disease} through at least one of these routes?"),
    # --- 短く、かつ言い方を変えた3版（コンセプトは M3 と同じ：(a) 治療・症状改善 / (b) 代償経路 / (c) 機構の一致 の OR） ---
    # M3r1：「治療標的になる3つの道筋」の枠組みを短く
    ("M3r1", +1, "Three routes can make a gene a therapeutic target for {disease}:\n"
                 "(a) Direct: inhibiting or activating {gene} could treat {disease} or relieve a symptom listed above.\n"
                 "(b) Compensatory: outside the causal pathway, modulating {gene} could offset the abnormal process described above via a "
                 "parallel or opposing pathway.\n"
                 "(c) Mechanistic: {gene}'s own substrate, ligand, pathway, cell type or circuit exactly matches the mechanism of {disease} "
                 "described above (a similar role in the same gene family does not count).\n"
                 "Does this gene fit at least one route?"),
    # M3r2：「次のどれかが正しければ Yes」と答え方を先に言う
    ("M3r2", +1, "Answer Yes if any of the following is true, otherwise No.\n"
                 "(a) Inhibiting or activating {gene} could plausibly treat {disease} or improve a symptom listed above.\n"
                 "(b) Although {gene} is not in the causal pathway, modulating it could counteract the abnormal process described above "
                 "through a parallel or opposing pathway.\n"
                 "(c) {gene}'s own substrate, ligand, pathway, cell type or circuit precisely matches the mechanism of {disease} described "
                 "above; a similar but distinct role, even in the same gene family, does not count.\n"
                 "Is at least one of these true for this gene?"),
    # M3r3：箇条書きをやめて1文の問いにまとめる
    ("M3r3", +1, "Could inhibiting or activating {gene} plausibly treat {disease} or relieve a symptom listed above — either directly, "
                 "by counteracting the abnormal process through a parallel or opposing pathway, or because {gene}'s own substrate, ligand, "
                 "pathway, cell type or circuit precisely matches the mechanism of {disease} described above (not merely a related gene "
                 "in the same family)?"),
    # --- 癖の分析から作った版：M4 = M3s の (b) だけ this gene に戻す（知名度による Yes を抑える）、M4L = M4 に短いラベルを付ける ---
    ("M4", +1, "Consider these three criteria:\n"
               "(a) Inhibiting or activating {gene} could plausibly treat {disease} or improve at least one of the symptoms listed above.\n"
               "(b) Even outside the causal pathway, modulating this gene could counteract the abnormal process described above through a "
               "parallel or opposing pathway.\n"
               "(c) {gene}'s own substrate, ligand, pathway, cell type or circuit precisely matches the mechanism of {disease} described above "
               "(a similar but distinct role, even in the same gene family, does not count).\n"
               "Does this gene meet at least one of these criteria?"),
    ("M4L", +1, "Consider these three criteria:\n"
                "(a) Direct benefit: inhibiting or activating {gene} could plausibly treat {disease} or improve at least one of the symptoms "
                "listed above.\n"
                "(b) Compensation: even outside the causal pathway, modulating this gene could counteract the abnormal process described above "
                "through a parallel or opposing pathway.\n"
                "(c) Mechanistic match: {gene}'s own substrate, ligand, pathway, cell type or circuit precisely matches the mechanism of {disease} "
                "described above (a similar but distinct role, even in the same gene family, does not count).\n"
                "Does this gene meet at least one of these criteria?"),
    ("C1", +1, "Setting the disease aside:is this a well-studied human gene, the subject of a large number of published research papers?"),
]
QIDS = [q[0] for q in QUESTIONS]
MIRROR = {"N1": "B3"}   # 否定形 → 対になる肯定形（N2〜N7 の検証結果は outputs/*_qvariants_neg.csv）
SIGN = {q[0]: q[1] for q in QUESTIONS}

STRICT_NOTE = ("Note: the vast majority of human genes are NOT drug targets for any given disease. "
               "Answer Yes only when there is clear evidence or a clear mechanistic link; otherwise answer No. "
               "Membership in the same gene family, superfamily or protein class as a true disease gene (e.g. being "
               "another member of the same transporter, channel, receptor or enzyme family) is NOT sufficient evidence "
               "by itself. Judge each gene on whether ITS OWN specific substrate, ligand, cargo or interaction partner "
               "matches the mechanism described above, not on family resemblance alone.\n")


def prefix_text(disease, info, order):
    options = "Answer each question with Yes or No." if order == "yes_first" else "Answer each question with No or Yes."
    bullets = "\n".join(f"- {b}" for b in info)
    return ("You are an expert in drug discovery and human disease biology.\n" + STRICT_NOTE +
            f"Disease: {disease}\nTarget symptoms and the organ, cell and functional abnormalities behind them:\n{bullets}\n"
            f"{options}\n\n")


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def load_genes(prefix, max_genes):
    g = pd.read_csv(os.path.join(ROOT, "data", "genes", f"{prefix}_set100.tsv"), sep="\t", dtype=str).fillna("")
    if max_genes: g = g.head(max_genes).copy()
    g["gene_label"] = [f"{r['symbol']} ({(r['protein_name_uniprot'] or r['gene_name']).split('|')[0]})"
                       if (r["protein_name_uniprot"] or r["gene_name"]) else r["symbol"] for _, r in g.iterrows()]
    return g


class Engine:
    def __init__(self, model_path, n_ctx=2048):
        from llama_cpp import Llama
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=-1, logits_all=False, verbose=False)
        self.yes = self.ids(["Yes", " Yes", "yes", " yes", "YES", " YES"])
        self.no = self.ids(["No", " No", "no", " no", "NO", " NO"])

    def tok(self, text, bos=False):
        return self.llm.tokenize(text.encode("utf-8"), add_bos=bos, special=bos)

    def ids(self, spellings):
        out = []
        for s in spellings:
            t = self.tok(s)
            if len(t) == 1 and t[0] not in out: out.append(t[0])
        return out

    def last_logprobs(self):
        llm = self.llm
        lg = np.ctypeslib.as_array(llm._ctx.get_logits(), shape=(llm.n_vocab(),)).astype(np.float64)
        lg -= lg.max()
        return lg - math.log(np.exp(lg).sum())

    def p_yes(self, lp):
        lse = lambda v: max(v) + math.log(sum(math.exp(x - max(v)) for x in v))
        ly, ln = lse([lp[i] for i in self.yes]), lse([lp[i] for i in self.no])
        return 1 / (1 + math.exp(ln - ly))

    def set_prefix(self, text):
        self.llm.reset(); self.llm.eval(self.tok(text, bos=True))
        self.state = (self.llm.save_state(), self.llm.n_tokens)

    def score(self, gene_label, question_lines):
        """前置き → 遺伝子ブロック の位置に毎回巻き戻して、各質問を独立に聞く。"""
        llm = self.llm
        llm.reset(); llm.load_state(self.state[0])
        llm.eval(self.tok(f"Gene: {gene_label}\n"))
        base = llm.n_tokens
        out = {}
        for qid, line in question_lines.items():
            llm.n_tokens = base                                   # eval() の先頭で base 以降の KV が消される
            llm.eval(self.tok(line))
            out[qid] = self.p_yes(self.last_logprobs())
        return out


def run_disease(eng, key, reg, max_genes, model_name, qids=None):
    D = reg[key]; disease, info = D["name"], D["info"][:5]
    genes = load_genes(D["gene_prefix"], max_genes)
    rows, t0 = [], time.time()
    for order in ("yes_first", "no_first"):
        eng.set_prefix(prefix_text(disease, info, order))
        for i, g in genes.iterrows():
            lines = {qid: f"{qid}. {text.format(disease=disease, gene=g['symbol'])} Answer:" for qid, _, text in QUESTIONS
                     if qids is None or qid in qids}
            p = eng.score(g["gene_label"], lines)
            rows += [{"symbol": g["symbol"], "category": g["category"], "note": g.get("note", ""), "order": order,
                      "qid": q, "p_yes": round(v, 6)} for q, v in p.items()]
            if (i + 1) % 25 == 0: print(f"  {key} {order} {i+1}/{len(genes)} ({time.time()-t0:.0f}s)", flush=True)
    long = pd.DataFrame(rows); long["model"] = model_name; long["disease"] = disease
    out = os.path.join(ROOT, "outputs", f"{D['gene_prefix']}_set100_qvariants.csv")
    if qids is not None and os.path.exists(out):                     # 一部の質問だけ聞いたときは、既存の CSV の同じ質問を置き換えて足す
        old = pd.read_csv(out)
        long = pd.concat([old[~old["qid"].isin(long["qid"].unique())], long], ignore_index=True)
    long.to_csv(out, index=False)
    print(f"{key}: {len(genes)} genes x {len(qids or QIDS)} questions x 2 orders in {time.time()-t0:.0f}s -> {out}", flush=True)
    return long


def summarize(long):
    """両順の対数オッズ平均 → 向きを揃えた採点 s（大きいほど標的らしい）。"""
    w = long.pivot_table(index=["symbol", "category", "qid"], columns="order", values="p_yes").reset_index()
    w["delta"] = w["yes_first"] - w["no_first"]
    w["lo"] = [(logit(a) + logit(b)) / 2 for a, b in zip(w["yes_first"], w["no_first"])]
    w["s"] = w["lo"] * w["qid"].map(SIGN)
    return w


def report(long, title):
    w = summarize(long)
    wide = w.pivot_table(index=["symbol", "category"], columns="qid", values="s").reset_index()
    for name, cols in {"mean_A1-B4": ["A1", "A2", "B1", "B2", "B3", "B4"], "mean_B1-B4": ["B1", "B2", "B3", "B4"],
                       "V0+B3+N1": ["V0", "B3", "N1"]}.items():
        wide[name] = wide[cols].mean(axis=1)
    k, c, r = (wide[wide.category == x] for x in ("known", "candidate", "random"))
    rest = pd.concat([c, r])
    have = [q for q in QIDS if q in wide.columns]
    cols = have + ["mean_A1-B4", "mean_B1-B4", "V0+B3+N1"]
    dlt = w.groupby("qid")["delta"].mean()
    p_avg = w.assign(p=1 / (1 + np.exp(-w["lo"]))).pivot_table(index="qid", columns="category", values="p", aggfunc="median")
    tbl = pd.DataFrame({
        "AUC k/rand": {q: auc(k[q], r[q]) for q in cols},
        "AUC k/rest": {q: auc(k[q], rest[q]) for q in cols},
        "AUC cand/rand": {q: auc(c[q], r[q]) for q in cols},
        "rho_vs_C1": {q: wide[q].corr(wide["C1"], method="spearman") for q in cols},
        "order_delta": {q: dlt.get(q, np.nan) for q in cols},
        "p_med_known": {q: p_avg.loc[q, "known"] if q in p_avg.index else np.nan for q in cols},
        "p_med_rand": {q: p_avg.loc[q, "random"] if q in p_avg.index else np.nan for q in cols},
    })
    print(f"\n===== {title}  (known {len(k)}, candidate {len(c)}, random {len(r)})")
    print(tbl.round(3).to_string())
    corr = wide[have].corr(method="spearman").round(2)
    print("Spearman between questions:\n" + corr.to_string())
    pw = w.assign(p=1 / (1 + np.exp(-w["lo"]))).pivot_table(index=["symbol", "category"], columns="qid", values="p").reset_index()
    mir = []
    for n, pos in MIRROR.items():                                     # 肯定形と否定形の食い違い：両方 Yes（p≥0.5）なら矛盾
        if n not in pw.columns or pos not in pw.columns: continue
        both = (pw[n] >= 0.5) & (pw[pos] >= 0.5)
        mir.append({"neg": n, "pos": pos, "rho(s_neg, s_pos)": wide[n].corr(wide[pos], method="spearman"),
                    "both_yes_known": both[pw.category == "known"].mean(), "both_yes_all": both.mean(),
                    "p_neg_med_known": pw.loc[pw.category == "known", n].median(), "p_neg_med_rand": pw.loc[pw.category == "random", n].median()})
    if mir: print("否定形と肯定形の対（both_yes = 両方に Yes と答えた割合 = 矛盾）:\n" + pd.DataFrame(mir).round(3).to_string(index=False))
    return wide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", nargs="+", default=["scz", "achondroplasia", "ra", "cystinuria", "prostate_cancer"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-genes", type=int, default=None)
    ap.add_argument("--report-only", action="store_true", help="既存の CSV から表だけ出す")
    ap.add_argument("--qids", nargs="+", default=None, help="この質問だけ聞いて既存の CSV に足す（例 --qids N2 N3）")
    a = ap.parse_args()
    reg = json.load(open(os.path.join(ROOT, "data", "diseases.json"), encoding="utf-8"))
    if a.report_only:
        for key in a.disease:
            report(pd.read_csv(os.path.join(ROOT, "outputs", f"{reg[key]['gene_prefix']}_set100_qvariants.csv")), key)
        return
    model = a.model or sorted(glob.glob(os.path.expanduser("~/llm/models/**/*txgemma*.gguf"), recursive=True))[0]
    print("model:", model, flush=True)
    eng = Engine(model)
    for key in a.disease:
        report(run_disease(eng, key, reg, a.max_genes, os.path.basename(model), a.qids), key)


if __name__ == "__main__":
    main()
