"""M1（3条件をまとめて「どれか1つに当てはまるか」を1問で聞く）と、3問（H3・R1・B1）を別々に聞いて平均する方式を比べる図（plotly、ホバーで遺伝子名）。

使い方: python scripts/plot_m1.py
出力: outputs/m1_charts.html
図: (1) AUC（既知 vs その他）とダミーの Yes 率  (2) 疾患ごとの散布図 M1 × 3問平均
    (3) 既知遺伝子の順位ヒートマップ  (4) 疾患ごとの全遺伝子の採点
"""
import os, re, sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))
from rescue_report import ROOT, auc, wide_scores

DISEASES = [("achondroplasia", "軟骨無形成症"), ("ra", "関節リウマチ"), ("prostate_cancer", "前立腺がん"),
            ("scz", "統合失調症"), ("cystinuria", "シスチン尿症")]
COLS = ["H3", "R1", "B1", "3問平均", "M1"]
LABEL = {"H3": "H3 治療仮説", "R1": "R1 代償", "B1": "B1 機構一致", "3問平均": "3問平均（別々に聞く）", "M1": "M1（1問で OR）"}
COLOR = {"H3": "#c9c8c2", "R1": "#b9d3f2", "B1": "#f6c3ad", "3問平均": "#2a78d6", "M1": "#eb6834"}
CAT = [("known", "既知", "#2a78d6", "circle"), ("candidate", "候補", "#eb6834", "square"), ("random", "ダミー", "#1baf7a", "diamond"),
       ("random_same", "ダミー（正解と同じファミリー）", "#eda100", "triangle-up")]
LAYOUT = dict(template="plotly_white", font=dict(family="Hiragino Sans, Noto Sans JP, sans-serif", size=12))
fam = lambda s: (re.match(r"[A-Z]+\d*", s) or re.match(r".*", s)).group(0)       # SLC7A9 -> SLC7（簡易）
sig = lambda x: 1 / (1 + np.exp(-x))


def load():
    data = {}
    for key, _ in DISEASES:
        w, _ = wide_scores(key)
        w["3問平均"] = w[["H3", "R1", "B1"]].mean(axis=1)
        kfam = {fam(s) for s in w.loc[w.category.eq("known"), "symbol"]}
        w["cat2"] = np.where(w.category.eq("random") & w.symbol.map(fam).isin(kfam), "random_same", w.category)
        data[key] = w
    return data


def fig_summary(data):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("AUC 既知 vs その他（候補＋ダミー）", "ダミーへの Yes 率（中央値）"), horizontal_spacing=0.08)
    names = [n for _, n in DISEASES]
    for q in COLS:
        a, y, extra = [], [], []
        for key, _ in DISEASES:
            w = data[key]; k, r, c = w.category.eq("known"), w.category.eq("random"), w.category.eq("candidate")
            a.append(auc(w.loc[k, q], w.loc[~k, q])); y.append(sig(w.loc[r, q].median()))
            extra.append([auc(w.loc[k, q], w.loc[r, q]), auc(w.loc[c, q], w.loc[r, q])])
        cd = np.array(extra)
        fig.add_trace(go.Bar(x=names, y=a, name=LABEL[q], marker_color=COLOR[q], legendgroup=q, customdata=cd,
                             hovertemplate=f"<b>{LABEL[q]}</b><br>%{{x}}<br>AUC 既知/その他 %{{y:.3f}}<br>AUC 既知/ダミー %{{customdata[0]:.3f}}"
                                           "<br>AUC 候補/ダミー %{customdata[1]:.3f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Bar(x=names, y=y, name=LABEL[q], marker_color=COLOR[q], legendgroup=q, showlegend=False,
                             hovertemplate=f"<b>{LABEL[q]}</b><br>%{{x}}<br>ダミーの Yes 率 %{{y:.2f}}<extra></extra>"), row=1, col=2)
    fig.update_yaxes(range=[0.4, 1.0], row=1, col=1); fig.update_yaxes(range=[0, 1.02], row=1, col=2)
    fig.update_layout(**LAYOUT, barmode="group", height=480, title="M1（1問で OR）と3問平均の比較 — 淡色は個別の質問")
    return fig


def fig_scatter(data):
    fig = make_subplots(rows=1, cols=5, subplot_titles=[n for _, n in DISEASES], horizontal_spacing=0.035)
    for j, (key, name) in enumerate(DISEASES, start=1):
        w = data[key]
        rho = w["M1"].corr(w["3問平均"], method="spearman")
        fig.layout.annotations[j - 1].text = f"{name}（ρ={rho:.2f}）"
        for cat, jp, col, mk in CAT:
            s = w[w.cat2.eq(cat)]
            if s.empty: continue
            fig.add_trace(go.Scatter(x=s["3問平均"], y=s["M1"], mode="markers", name=jp, legendgroup=cat, showlegend=(j == 1),
                                     marker=dict(color=col, symbol=mk, size=9 if cat in ("known", "random_same") else 6,
                                                 opacity=0.95 if cat in ("known", "random_same") else 0.55, line=dict(color="white", width=0.5)),
                                     customdata=np.c_[s["symbol"], [jp] * len(s)],
                                     hovertemplate="<b>%{customdata[0]}</b>（%{customdata[1]}）<br>3問平均 %{x:.2f}<br>M1 %{y:.2f}<extra></extra>"),
                          row=1, col=j)
        fig.update_xaxes(title_text="3問平均" if j == 3 else None, row=1, col=j)
    fig.update_yaxes(title_text="M1（対数オッズ）", row=1, col=1)
    fig.update_layout(**LAYOUT, height=420, title="遺伝子ごとの点数：横＝3問平均、縦＝M1（右上ほど標的らしい。ρ＝順位相関）")
    return fig


def fig_rank(data):
    rows, ylab = [], []
    for key, name in DISEASES:
        w = data[key]; k = w.category.eq("known")
        rk = {q: w[q].rank(ascending=False, method="min") for q in COLS}
        for i in w.index[k].to_series().sort_values(key=lambda s: rk["3問平均"][s], ascending=False):
            rows.append([rk[q][i] for q in COLS]); ylab.append(f"{name}｜{w.at[i, 'symbol']}")
    z = np.array(rows, float)
    d = z[:, 4] - z[:, 3]                                              # M1 − 3問平均（負 = M1 で上がった）
    fig = make_subplots(rows=1, cols=2, column_widths=[0.82, 0.18], horizontal_spacing=0.02, shared_yaxes=True,
                        subplot_titles=("順位（100中）", "M1 − 3問平均"))
    fig.add_trace(go.Heatmap(z=z, x=[LABEL[q] for q in COLS], y=ylab, text=z.astype(int).astype(str), texttemplate="%{text}",
                             colorscale="RdYlGn_r", zmin=1, zmax=100, colorbar=dict(title="順位", x=0.8),
                             hovertemplate="%{y}<br>%{x}<br>順位 %{z:.0f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Heatmap(z=d[:, None], x=["差"], y=ylab, text=[[f"{v:+.0f}"] for v in d], texttemplate="%{text}",
                             colorscale="RdBu_r", zmid=0, zmin=-40, zmax=40, showscale=False,
                             hovertemplate="%{y}<br>M1 − 3問平均 %{z:+.0f}（負＝M1 で上がった）<extra></extra>"), row=1, col=2)
    fig.update_yaxes(autorange="reversed")
    fig.add_vline(x=2.5, line=dict(color="#0b0b0b", width=1.5), row=1, col=1)
    fig.update_layout(**LAYOUT, height=24 * len(ylab) + 200, margin=dict(l=190),
                      title="既知遺伝子の順位：3問平均 vs M1（緑＝上位。右端の青＝M1 で上がった、赤＝下がった）")
    return fig


def fig_strip(data, key, name):
    w = data[key]; qs = COLS
    fig, rng = go.Figure(), np.random.default_rng(0)
    for cat, jp, col, mk in CAT:
        s = w[w.cat2.eq(cat)]
        if s.empty: continue
        for j, q in enumerate(qs):
            fig.add_trace(go.Scatter(x=j + rng.uniform(-0.28, 0.28, len(s)), y=s[q], mode="markers", name=jp, legendgroup=cat, showlegend=(j == 0),
                                     marker=dict(color=col, symbol=mk, size=10 if cat == "known" else 6,
                                                 opacity=0.95 if cat in ("known", "random_same") else 0.5, line=dict(color="white", width=0.5)),
                                     customdata=np.c_[s["symbol"], [jp] * len(s), [LABEL[q]] * len(s)],
                                     hovertemplate="<b>%{customdata[0]}</b>（%{customdata[1]}）<br>%{customdata[2]}<br>対数オッズ %{y:.2f}<extra></extra>"))
    fig.add_vrect(x0=2.5, x1=4.5, fillcolor="#fff3c4", opacity=0.45, line_width=0, layer="below")
    fig.update_layout(**LAYOUT, title=f"{name}：全遺伝子の採点（両順の対数オッズ平均。黄色＝合成スコア）", height=440,
                      xaxis=dict(tickmode="array", tickvals=list(range(len(qs))), ticktext=[LABEL[q] for q in qs]),
                      yaxis=dict(title="対数オッズ（大きいほど Yes）"))
    return fig


def main():
    data = load()
    figs = [fig_summary(data), fig_scatter(data), fig_rank(data)] + [fig_strip(data, k, n) for k, n in DISEASES]
    out = os.path.join(ROOT, "outputs", "m1_charts.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("<html><head><meta charset='utf-8'><title>M1 の検証</title></head><body style='max-width:1300px;margin:auto'>"
                 + "".join(f.to_html(full_html=False, include_plotlyjs=(True if i == 0 else False)) for i, f in enumerate(figs)) + "</body></html>")
    print("saved:", out)
    return data, figs


if __name__ == "__main__":
    main()
