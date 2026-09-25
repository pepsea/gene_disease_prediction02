"""レスキュー質問（R1〜R3）の効果を見る：取りこぼした既知遺伝子の順位がどれだけ上がったか、代わりにダミーも上がっていないか。

使い方: python scripts/rescue_report.py
入力: outputs/<疾患>_set100_qvariants.csv（yesno_question_variants.py の出力）
"""
import math, os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DISEASES = ["achondroplasia", "ra", "prostate_cancer", "scz", "cystinuria"]
BASE = ["B3", "B4", "H2", "H3", "P3", "P4", "C1"]
RESCUE = ["R1", "R2", "R3"]
COMBOS = {"base(B4+H2+P3)": ["B4", "H2", "P3"], "base+max(R)": None, "base+R1+R2+R3": ["B4", "H2", "P3", "R1", "R2", "R3"]}


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def wide_scores(key):
    d = pd.read_csv(os.path.join(ROOT, "outputs", f"{key}_set100_qvariants.csv"))
    d["lo"] = d["p_yes"].map(logit)
    w = d.groupby(["symbol", "category", "qid"])["lo"].mean().unstack().reset_index()   # 両順の対数オッズ平均
    have_r = [q for q in RESCUE if q in w.columns]
    w["base(B4+H2+P3)"] = w[["B4", "H2", "P3"]].mean(axis=1)
    if have_r:
        w["base+max(R)"] = np.maximum(w["base(B4+H2+P3)"], w[have_r].max(axis=1))       # 理由のどれか1つで拾えれば良い（OR 的）
        if {"R1", "R2", "R3"} <= set(have_r): w["base+R1+R2+R3"] = w[["B4", "H2", "P3", "R1", "R2", "R3"]].mean(axis=1)
    return w, have_r


def main():
    for key in DISEASES:
        w, have_r = wide_scores(key)
        cols = [c for c in BASE + have_r + list(COMBOS) if c in w.columns]
        k, r = w.category.eq("known"), w.category.eq("random")
        rank = {c: w[c].rank(ascending=False, method="min") for c in cols}
        print(f"\n===== {key}  (known {k.sum()}, random {r.sum()}, n={len(w)})")
        t = pd.DataFrame({"AUC k/rand": {c: auc(w.loc[k, c], w.loc[r, c]) for c in cols},
                          "AUC k/rest": {c: auc(w.loc[k, c], w.loc[~k, c]) for c in cols},
                          "rho_vs_C1": {c: w[c].corr(w["C1"], method="spearman") for c in cols},
                          "rand_median_p": {c: 1 / (1 + math.exp(-w.loc[r, c].median())) for c in cols}})
        print(t.round(3).to_string())
        kr = pd.DataFrame({c: rank[c][k].astype(int) for c in cols}); kr.insert(0, "symbol", w.loc[k, "symbol"])
        print("既知遺伝子の順位（100中、小さいほど良い）:\n" + kr.sort_values("base(B4+H2+P3)", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
