"""ノートブック 06 の流れ（2段階ランキング＋理由のラベル）を検証するスクリプト（GGUF 直接駆動）。

段階1：M3s と M3sF（機能情報つき）の対数オッズの平均で全遺伝子を採点する（outputs/<疾患>_set100_qvariants.csv の M3s・M3sF を使う）。
段階2：段階1の上位 TOP_K を、5択（M2r）× スイス式 × Bradley-Terry で並べ直す（swiss.py と同じ組み方・推定）。
        最終順位：上位 TOP_K の中は「段階1の順位と段階2の順位の悪い方」（同点は平均）で並べ、残りは段階1の順に後ろへ。
段階3：最終上位 N_LABEL に、H3（直接効く）・R1（代償経路）・B1（機構が一致）のどれで Yes だったかのラベルを付ける（順位には使わない）。

使い方: python scripts/pipeline06.py --disease achondroplasia ra prostate_cancer scz cystinuria
出力: outputs/<疾患>_set100_pipeline06.csv
"""
import argparse, glob, json, math, os, random, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from yesno_question_variants import ROOT, load_genes, logit
from rank_variants import Engine, VARIANTS, prefix_text
from rank_bt import luce
from swiss import random_groups, swiss_groups

VARIANT = "M2r"
LABELS = {"H3": "直接効く", "R1": "代償経路", "B1": "機構が一致"}


def stage1(key, prefix):
    qv = pd.read_csv(os.path.join(ROOT, "outputs", f"{prefix}_set100_qvariants.csv"))
    qv = qv[qv["qid"].isin(["M3s", "M3sF", "H3", "R1", "B1"])].assign(lo=lambda d: d["p_yes"].map(logit))
    w = qv.groupby(["symbol", "qid"])["lo"].mean().unstack()                      # 両順の対数オッズ平均
    w["stage1"] = (w["M3s"] + w["M3sF"]) / 2
    return w


def stage2(eng, genes, top, disease, info, rounds=8, random_rounds=2, jitter=0.3, seed=0):
    """上位 top（genes の行番号）だけでスイス式トーナメントを行い、Bradley-Terry の強さを返す。"""
    eng.set_prefix(prefix_text(disease, info))
    line = f"Question: {VARIANTS[VARIANT].format(disease=disease)} Answer: "
    rng, idx, rows = random.Random(seed), list(top), []
    def fit():
        items = idx + ["NONE"]; ix = {s: k for k, s in enumerate(items)}; groups = []
        for _, d in pd.DataFrame(rows).groupby(["round", "group"]):
            d = d.sort_values("position")
            groups.append(([ix[i] for i in d["gi"]] + [ix["NONE"]], np.r_[d["p"].values, d["p_none"].iloc[0]]))
        return luce(groups, len(items))
    for r in range(rounds):
        if r < random_rounds: groups = random_groups(idx, rng)
        else:
            s = fit(); groups = swiss_groups(idx, {i: s[k] for k, i in enumerate(idx)}, rng, jitter)
        for gi, chunk in enumerate(groups):
            p = eng.score([genes.at[i, "gene_label"] for i in chunk], {VARIANT: line})[VARIANT]
            for slot, i in enumerate(chunk):
                rows.append({"round": r, "group": gi, "position": slot + 1, "gi": i, "p": round(float(p[slot]), 6), "p_none": round(float(p[-1]), 6)})
    s = fit()
    return pd.Series(s[:-1], index=idx), float(s[-1])


def run_disease(eng, key, reg, top_k, n_label, model_name):
    D = reg[key]; disease, info = D["name"], D["info"][:5]
    genes = load_genes(D["gene_prefix"], None)
    w = stage1(key, D["gene_prefix"]).reindex(genes["symbol"]).reset_index(drop=True)
    t = genes[["symbol", "category"]].join(w)
    t["stage1_rank"] = t["stage1"].rank(ascending=False, method="first").astype(int)
    top = t.sort_values("stage1_rank").head(top_k).index.tolist()
    t0 = time.time()
    bt, none = stage2(eng, genes, top, disease, info)
    t["stage2_bt"] = bt; t["stage2_rank"] = t["stage2_bt"].rank(ascending=False, method="first")
    t["stage2_above_none"] = t["stage2_bt"] > none
    s1r = t["stage1_rank"].where(t.index.isin(top))
    key_final = np.where(t.index.isin(top), np.maximum(s1r, t["stage2_rank"]) + (s1r + t["stage2_rank"]) / 1000, t["stage1_rank"] + 1000)
    t["final_rank"] = pd.Series(key_final).rank(method="first").astype(int)
    for q in LABELS: t[f"yes_{q}"] = t[q] > 0                                     # 両順の対数オッズ平均が 0 より上＝Yes
    t["reasons"] = ["・".join(LABELS[q] for q in LABELS if r[f"yes_{q}"]) or "（どれも No）" if r["final_rank"] <= n_label else ""
                    for _, r in t.iterrows()]
    t["model"] = model_name; t["disease"] = disease
    out = os.path.join(ROOT, "outputs", f"{D['gene_prefix']}_set100_pipeline06.csv")
    t.sort_values("final_rank").to_csv(out, index=False)
    print(f"{key}: stage2 top {top_k} in {time.time() - t0:.0f}s -> {out}", flush=True)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", nargs="+", default=["achondroplasia", "ra", "prostate_cancer", "scz", "cystinuria"])
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--n-label", type=int, default=20)
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    reg = json.load(open(os.path.join(ROOT, "data", "diseases.json"), encoding="utf-8"))
    model = a.model or sorted(glob.glob(os.path.expanduser("~/llm/models/**/*txgemma*.gguf"), recursive=True))[0]
    print("model:", model, flush=True)
    eng = Engine(model)
    for key in a.disease:
        run_disease(eng, key, reg, a.top_k, a.n_label, os.path.basename(model))


if __name__ == "__main__":
    main()
