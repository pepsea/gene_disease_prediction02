"""ノートブック 06 の評価指標：候補 vs ダミー、知名度（C1）をそろえた AUC、ダミーの Yes 率、既知 vs その他、同族ダミーとの AUC、取りこぼしの順位。
使い方: python scripts/eval06.py --disease ra --scores M3s M3sF
"""
import argparse, os, re, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from rescue_report import auc, wide_scores

fam = lambda s: (re.match(r"[A-Z]+\d*", s) or re.match(r".*", s)).group(0)
MISSED = ["NPR2", "GH1", "FOLH1", "DHFR", "DHODH", "MS4A1", "CD80", "CD86", "NR3C1", "CHRM1", "CHRM4", "HTR7", "SLC7A9", "TNFSF11"]


def strat_auc(s_pos, b_pos, s_neg, b_neg):
    """知名度をそろえた AUC：C1 の5分位ごとに、同じ層の中だけで正例と負例を比べる。"""
    num = den = 0.0
    for b in np.unique(np.r_[b_pos, b_neg]):
        p, n = s_pos[b_pos == b], s_neg[b_neg == b]
        if len(p) and len(n): num += (p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum(); den += len(p) * len(n)
    return num / den if den else np.nan


def evaluate(w, cols):
    bins = pd.qcut(w["C1"].rank(method="first"), 5, labels=False).values
    kn, rd, cd = (w.category.eq(c).values for c in ("known", "random", "candidate"))
    same = rd & w.symbol.map(fam).isin({fam(s) for s in w.symbol[kn]}).values
    rows = []
    for m in cols:
        s = w[m].values; rk = pd.Series(s).rank(ascending=False, method="min").values
        rows.append({"score": m, "候補 vs ダミー": auc(s[cd], s[rd]), "候補 vs ダミー（知名度をそろえる）": strat_auc(s[cd], bins[cd], s[rd], bins[rd]),
                     "ダミーの Yes 率": float(np.median(1 / (1 + np.exp(-s[rd])))) if m[0] == "M" else np.nan,
                     "既知 vs その他": auc(s[kn], s[~kn]), "既知 vs ダミー（知名度をそろえる）": strat_auc(s[kn], bins[kn], s[rd], bins[rd]),
                     "同族ダミーとの AUC": auc(s[kn], s[same]) if same.sum() >= 3 else np.nan,
                     "取りこぼしの順位": {g: int(r) for g, r, k in zip(w.symbol, rk, kn) if k and g in MISSED}})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--disease", nargs="+", required=True); ap.add_argument("--scores", nargs="+", required=True)
    a = ap.parse_args(); pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
    for k in a.disease:
        print("=====", k); print(evaluate(wide_scores(k)[0], a.scores).round(3).to_string(index=False))
