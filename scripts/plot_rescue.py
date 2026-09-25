"""レスキュー質問（R1〜R3）の結果を plotly で図にする（ホバーで遺伝子名）。

使い方: python scripts/plot_rescue.py
出力: outputs/rescue_charts.html
図: (1) 既知遺伝子の順位ヒートマップ（疾患 × 質問）
    (2) 質問ごとの AUC（既知 vs ダミー） × ダミーの Yes 率（右下ほど「何にでも Yes」、左上ほど良い）
    (3) 疾患ごとの全遺伝子の散布（質問ごとの対数オッズ、取りこぼした既知遺伝子に名前）
"""
import os, sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))
from rescue_report import ROOT, auc, wide_scores

DISEASES = [("achondroplasia", "軟骨無形成症"), ("ra", "関節リウマチ"), ("prostate_cancer", "前立腺がん"),
            ("scz", "統合失調症"), ("cystinuria", "シスチン尿症")]
QS = ["B3", "B4", "H2", "H3", "P3", "P4", "C1", "R1", "R2", "R3"]
LABEL = {"B3": "B3 摂動で症状改善", "B4": "B4 既知標的の近く", "H2": "H2 信号の受け手でも可", "H3": "H3 治療仮説",
         "P3": "P3 KO/KI で改善", "P4": "P4 阻害/活性化で緩和", "C1": "C1 有名さ（対照）",
         "R1": "R1 並行・拮抗経路で代償", "R2": "R2 症状の一部を改善する仮説", "R3": "R3 分子機構を改善する仮説"}
CAT = [("known", "既知", "#2a78d6", "circle"), ("candidate", "候補", "#eb6834", "square"), ("random", "ダミー", "#1baf7a", "diamond")]
DCOL = {"achondroplasia": "#2a78d6", "ra": "#eb6834", "prostate_cancer": "#1baf7a", "scz": "#8a5cd1", "cystinuria": "#d64545"}
MISSED_TOP = 20                                                     # base(B4+H2+P3) でこれより下の既知遺伝子を「取りこぼし」として名前を出す
LAYOUT = dict(template="plotly_white", font=dict(family="Hiragino Sans, Noto Sans JP, sans-serif", size=12))


def load():
    data = {}
    for key, name in DISEASES:
        w, _ = wide_scores(key)
        data[key] = w
    return data


def fig_rank_heatmap(data):
    rows, ylab, hover = [], [], []
    for key, name in DISEASES:
        w = data[key]; k = w.category.eq("known")
        ranks = {q: w[q].rank(ascending=False, method="min") for q in QS if q in w.columns}
        order = w.loc[k].assign(b=w["base(B4+H2+P3)"]).sort_values("b").index    # 取りこぼし（base の低い順）を上に
        for i in order:
            sym = w.at[i, "symbol"]
            rows.append([ranks[q][i] if q in ranks else np.nan for q in QS])
            ylab.append(f"{name}｜{sym}")
            hover.append([f"{sym}（{name}）<br>{LABEL[q]}<br>順位 {int(ranks[q][i]) if q in ranks else '—'} / {len(w)}" for q in QS])
    z = np.array(rows, float)
    fig = go.Figure(go.Heatmap(z=z, x=[LABEL[q] for q in QS], y=ylab, text=np.where(np.isnan(z), "", np.nan_to_num(z).astype(int).astype(str)),
                               texttemplate="%{text}", hovertext=hover, hoverinfo="text", colorscale="RdYlGn_r", zmin=1, zmax=100,
                               colorbar=dict(title="順位")))
    fig.update_layout(**LAYOUT, title="既知遺伝子の順位（100中、小さいほど良い。緑＝上位）— 各疾患の上側が取りこぼし",
                      height=28 * len(ylab) + 260, yaxis=dict(autorange="reversed"), xaxis=dict(side="top", tickangle=-30), margin=dict(l=200, t=230), title_y=0.99)
    for x0 in (6.5,):                                              # C1 と R1 の間に区切り
        fig.add_vline(x=x0, line=dict(color="#0b0b0b", width=2))
    return fig


def fig_auc_vs_yes(data):
    fig = go.Figure()
    for key, name in DISEASES:
        w = data[key]; k, r = w.category.eq("known"), w.category.eq("random")
        qs = [q for q in QS if q in w.columns and q != "C1"]
        x = [1 / (1 + np.exp(-w.loc[r, q].median())) for q in qs]
        y = [auc(w.loc[k, q], w.loc[r, q]) for q in qs]
        sym = ["star" if q.startswith("R") else "circle" for q in qs]
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers+text", name=name, text=qs, textposition="top center",
                                 marker=dict(color=DCOL[key], size=[15 if q.startswith("R") else 9 for q in qs], symbol=sym,
                                             line=dict(color="white", width=1)),
                                 customdata=[[LABEL[q], name] for q in qs],
                                 hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>AUC 既知 vs ダミー %{y:.3f}<br>ダミーの Yes 率（中央値） %{x:.2f}<extra></extra>"))
    fig.update_layout(**LAYOUT, title="質問の良し悪し：左上ほど良い（既知を上に置き、ダミーには No）　★ = レスキュー質問 R1〜R3",
                      xaxis=dict(title="ダミーの Yes 率（中央値）", range=[-0.03, 1.03]), yaxis=dict(title="AUC 既知 vs ダミー", range=[0.45, 1.03]),
                      height=620)
    return fig


def fig_strips(data, key, name):
    w = data[key]
    qs = [q for q in QS if q in w.columns]
    k = w.category.eq("known")
    missed = set(w.loc[k & (w["base(B4+H2+P3)"].rank(ascending=False) > MISSED_TOP), "symbol"])
    fig = go.Figure()
    rng = np.random.default_rng(0)
    for cat, jp, col, mk in CAT:
        sub = w[w.category.eq(cat)]
        for j, q in enumerate(qs):
            jit = rng.uniform(-0.28, 0.28, len(sub))
            fig.add_trace(go.Scatter(x=j + jit, y=sub[q], mode="markers", name=jp, legendgroup=cat, showlegend=(j == 0),
                                     marker=dict(color=col, symbol=mk, size=9 if cat == "known" else 6, opacity=0.9 if cat == "known" else 0.55,
                                                 line=dict(color="white", width=0.5)),
                                     customdata=np.c_[sub["symbol"], [jp] * len(sub), [LABEL[q]] * len(sub)],
                                     hovertemplate="<b>%{customdata[0]}</b>（%{customdata[1]}）<br>%{customdata[2]}<br>対数オッズ %{y:.2f}<extra></extra>"))
    for sym in sorted(missed):                                        # 取りこぼした既知遺伝子は線でつないで名前を出す
        row = w[w.symbol == sym].iloc[0]
        fig.add_trace(go.Scatter(x=list(range(len(qs))), y=[row[q] for q in qs], mode="lines+markers+text", name=f"取りこぼし: {sym}",
                                 text=[sym if q in ("B3", "R1", "R3") else "" for q in qs], textposition="middle right",
                                 line=dict(color="#0b0b0b", width=1.2, dash="dot"), marker=dict(color="#2a78d6", size=10, line=dict(color="#0b0b0b", width=1.5)),
                                 hovertemplate=f"<b>{sym}</b>（既知・取りこぼし）<br>%{{y:.2f}}<extra></extra>"))
    fig.add_vrect(x0=qs.index("R1") - 0.5, x1=len(qs) - 0.5, fillcolor="#fff3c4", opacity=0.45, line_width=0, layer="below")
    fig.update_layout(**LAYOUT, title=f"{name}：全遺伝子の採点（両順の対数オッズ平均。黄色＝レスキュー質問、点線＝取りこぼした既知遺伝子）",
                      xaxis=dict(tickmode="array", tickvals=list(range(len(qs))), ticktext=[LABEL[q] for q in qs], tickangle=-30),
                      yaxis=dict(title="対数オッズ（大きいほど Yes）"), height=560)
    return fig


def main():
    data = load()
    figs = [fig_rank_heatmap(data), fig_auc_vs_yes(data)] + [fig_strips(data, k, n) for k, n in DISEASES]
    out = os.path.join(ROOT, "outputs", "rescue_charts.html")
    html = "".join(f.to_html(full_html=False, include_plotlyjs=(True if i == 0 else False)) for i, f in enumerate(figs))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("<html><head><meta charset='utf-8'><title>レスキュー質問の検証</title></head><body style='max-width:1200px;margin:auto'>"
                 + html + "</body></html>")
    print("saved:", out)


if __name__ == "__main__":
    main()
