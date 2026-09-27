"""ノートブック 06 の検証まとめ図：M3s・M3sF・その平均（06 の点数）・段階2との統合・知名度の残差を、5疾患で比べる（plotly.js 埋め込み）。
使い方: python scripts/plot06_summary.py
入力: outputs/<疾患>_set100_qvariants.csv（M3s, M3sF, C1）と outputs/<疾患>_set100_pipeline06.csv（pipeline06.py）
出力: outputs/ranking06_summary.html
"""
import os, sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
sys.path.insert(0, os.path.dirname(__file__))
from rescue_report import ROOT, wide_scores
from eval06 import evaluate

DISEASES = [("achondroplasia", "軟骨無形成症"), ("ra", "関節リウマチ"), ("prostate_cancer", "前立腺がん"), ("scz", "統合失調症"), ("cystinuria", "シスチン尿症")]
METHODS = [("resid", "知名度の残差（C1 で補正）", "#e6b8b8"), ("M3s", "M3s（03）", "#c9c8c2"), ("M3sF", "M3sF（機能情報つき）", "#9ec0ec"),
           ("stage1", "M3s と M3sF の平均（06）", "#2a78d6"), ("final", "段階1と段階2の悪い方", "#8a5cd1")]
COLS = ["候補 vs ダミー", "候補 vs ダミー（知名度をそろえる）", "既知 vs その他", "既知 vs ダミー（知名度をそろえる）"]
LAYOUT = dict(template="plotly_white", font=dict(family="Hiragino Sans, Noto Sans JP, sans-serif", size=12))

rows = []
for k, n in DISEASES:
    w = wide_scores(k)[0]
    b1, b0 = np.polyfit(w["C1"], w["M3s"], 1); w["resid"] = w["M3s"] - (b0 + b1 * w["C1"])
    p = pd.read_csv(os.path.join(ROOT, "outputs", f"{k}_set100_pipeline06.csv")).set_index("symbol")
    w["stage1"] = w["symbol"].map(p["stage1"]); w["final"] = -w["symbol"].map(p["final_rank"])
    rows.append(evaluate(w, [m for m, *_ in METHODS]).assign(疾患=n))
T = pd.concat(rows)
print(T.groupby("score")[COLS].mean().round(3))
xs = [n for _, n in DISEASES] + ["平均"]
f = make_subplots(rows=1, cols=4, subplot_titles=[c.replace("（", "<br>（") for c in COLS], shared_yaxes=True, horizontal_spacing=0.03)
for j, c in enumerate(COLS, start=1):
    for m, lab, color in METHODS:
        s = T[T["score"] == m].set_index("疾患")[c]; y = list(s.reindex(xs[:-1])) + [s.mean()]
        f.add_trace(go.Bar(x=xs, y=y, name=lab, marker_color=color, legendgroup=m, showlegend=(j == 1),
                           hovertemplate=f"<b>{lab}</b><br>%{{x}}<br>{c} %{{y:.3f}}<extra></extra>"), row=1, col=j)
f.update_yaxes(range=[0.4, 1.02])
f.update_layout(**LAYOUT, barmode="group", height=500, margin=dict(t=110),
                title="ノートブック 06 の検証（AUC、5疾患）：濃い青＝06 の点数（M3s と機能情報つき M3sF の平均）")
out = os.path.join(ROOT, "outputs", "ranking06_summary.html")
open(out, "w", encoding="utf-8").write("<html><head><meta charset='utf-8'><title>06 の検証</title></head><body style='max-width:1300px;margin:auto'>"
                                        + f.to_html(full_html=False, include_plotlyjs=True) + "</body></html>")
print("saved:", out)
