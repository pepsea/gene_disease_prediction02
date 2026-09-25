"""M1 と、条件 (a) を「症状の改善」に広げた M2 を、Yes/No 版と強制選択版（M1r / M2r）の両方で比べる図（plotly、ホバーで遺伝子名）。

使い方: python scripts/plot_m2.py
入力: outputs/<疾患>_set100_qvariants.csv（M1, M2）と outputs/<疾患>_set100_rankvar.csv（M1r, M2r）
出力: outputs/m2_charts.html
図: (1) AUC 3種（疾患ごと＋平均）  (2) ダミーの Yes 率・同族ダミーとの AUC  (3) 既知遺伝子の順位  (4) M1 × M2 の散布図
"""
import os, re, sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))
from rescue_report import ROOT, auc, wide_scores
from rank_variants import gene_scores

DISEASES = [("achondroplasia", "軟骨無形成症"), ("ra", "関節リウマチ"), ("prostate_cancer", "前立腺がん"),
            ("scz", "統合失調症"), ("cystinuria", "シスチン尿症")]
METHODS = ["M1", "M2", "M3", "M3s", "M3d", "M3w", "M1r", "M2r"]
LABEL = {"M1": "Yes/No M1（旧）", "M2": "Yes/No M2（現行）", "M3": "Yes/No M3（M2 の表記を遺伝子名・病名にそろえた版）",
         "M3s": "Yes/No M3s（M3 を短く）", "M3d": "Yes/No M3d（M3 を詳しく）", "M3w": "Yes/No M3w（M3 の言い方を変更）", "M1r": "選択式 M1r", "M2r": "選択式 M2r"}
COLOR = {"M1": "#9ec0ec", "M2": "#2a78d6", "M3": "#123f7a", "M3s": "#1baf7a", "M3d": "#8a5cd1", "M3w": "#eda100", "M1r": "#f6c3ad", "M2r": "#eb6834"}
CAT = [("known", "既知", "#2a78d6", "circle"), ("candidate", "候補", "#eb6834", "square"), ("random", "ダミー", "#1baf7a", "diamond"),
       ("random_same", "ダミー（正解と同じファミリー）", "#eda100", "triangle-up")]
LAYOUT = dict(template="plotly_white", font=dict(family="Hiragino Sans, Noto Sans JP, sans-serif", size=12))
fam = lambda s: (re.match(r"[A-Z]+\d*", s) or re.match(r".*", s)).group(0)


def load():
    data = {}
    for key, _ in DISEASES:
        w, _ = wide_scores(key)
        w = w[["symbol", "category"] + [m for m in ("M1", "M2", "M3", "M3s", "M3d", "M3w") if m in w.columns]].copy()
        g = gene_scores(pd.read_csv(os.path.join(ROOT, "outputs", f"{key}_set100_rankvar.csv")))
        r = g[g.variant.isin(["M1r", "M2r"])].pivot_table(index="symbol", columns="variant", values="mean_p")
        w = w.merge(r, left_on="symbol", right_index=True, how="left")
        kfam = {fam(s) for s in w.loc[w.category.eq("known"), "symbol"]}
        w["cat2"] = np.where(w.category.eq("random") & w.symbol.map(fam).isin(kfam), "random_same", w.category)
        data[key] = w
    return data


def table(data):
    rows = []
    for key, name in DISEASES:
        w = data[key]; k, r, c, rs = (w.category.eq("known"), w.category.eq("random"), w.category.eq("candidate"), w.cat2.eq("random_same"))
        for m in METHODS:
            rows.append({"疾患": name, "方式": m, "既知 vs その他": auc(w.loc[k, m], w.loc[~k, m]), "既知 vs ダミー": auc(w.loc[k, m], w.loc[r, m]),
                         "候補 vs ダミー": auc(w.loc[c, m], w.loc[r, m]),
                         "既知 vs 同族ダミー": auc(w.loc[k, m], w.loc[rs, m]) if rs.sum() >= 3 else np.nan,
                         "ダミーの Yes 率": 1 / (1 + np.exp(-w.loc[r, m].median())) if not m.endswith("r") else np.nan})
    return pd.DataFrame(rows)


def fig_auc(t):
    cols = ("既知 vs その他", "既知 vs ダミー", "候補 vs ダミー")
    fig = make_subplots(rows=1, cols=3, subplot_titles=cols, shared_yaxes=True, horizontal_spacing=0.03)
    names = [n for _, n in DISEASES] + ["平均"]
    for j, col in enumerate(cols, start=1):
        for m in METHODS:
            s = t[t["方式"] == m].set_index("疾患")[col]
            y = list(s.reindex(names[:-1])) + [s.mean()]
            fig.add_trace(go.Bar(x=names, y=y, name=LABEL[m], marker_color=COLOR[m], legendgroup=m, showlegend=(j == 1),
                                 hovertemplate=f"<b>{LABEL[m]}</b><br>%{{x}}<br>{col} %{{y:.3f}}<extra></extra>"), row=1, col=j)
    fig.update_yaxes(range=[0.5, 1.0])
    fig.update_layout(**LAYOUT, barmode="group", height=480, title="AUC：M1（旧）・M2（現行）・M3（表記をそろえた版）と選択式")
    return fig


def fig_side(t):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ダミーの Yes 率（Yes/No 版、中央値）", "シスチン尿症：既知 vs 正解と同じファミリーのダミー"))
    for m in [m for m in METHODS if not m.endswith("r")]:
        s = t[t["方式"] == m]
        fig.add_trace(go.Bar(x=s["疾患"], y=s["ダミーの Yes 率"], name=LABEL[m], marker_color=COLOR[m], legendgroup=m,
                             hovertemplate=f"<b>{LABEL[m]}</b><br>%{{x}}<br>ダミーの Yes 率 %{{y:.2f}}<extra></extra>"), row=1, col=1)
    s = t[t["疾患"] == "シスチン尿症"]
    fig.add_trace(go.Bar(x=[LABEL[m] for m in s["方式"]], y=s["既知 vs 同族ダミー"], marker_color=[COLOR[m] for m in s["方式"]], showlegend=False,
                         hovertemplate="%{x}<br>AUC %{y:.3f}<extra></extra>"), row=1, col=2)
    fig.update_yaxes(range=[0, 1.02])
    fig.update_layout(**LAYOUT, barmode="group", height=420, title="副作用のチェック：広げた条件がダミーや同族の遺伝子まで拾っていないか")
    return fig


def fig_rank(data):
    rows, ylab = [], []
    for key, name in DISEASES:
        w = data[key]; k = w.category.eq("known")
        rk = {m: w[m].rank(ascending=False, method="min") for m in METHODS}
        for i in w.index[k].to_series().sort_values(key=lambda s: rk["M1"][s], ascending=False):
            rows.append([rk[m][i] for m in METHODS]); ylab.append(f"{name}｜{w.at[i, 'symbol']}")
    z = np.array(rows, float)
    ix = {m: METHODS.index(m) for m in METHODS}
    DIFF = [("M2", "M1"), ("M3", "M2"), ("M3s", "M3"), ("M3d", "M3"), ("M3w", "M3")]
    d = np.c_[tuple(z[:, ix[a]] - z[:, ix[b]] for a, b in DIFF)]   # 負 = 左の方式で上がった
    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], shared_yaxes=True, horizontal_spacing=0.02,
                        subplot_titles=("順位（100中）", "差（負＝右側の方式で上がった）"))
    fig.add_trace(go.Heatmap(z=z, x=[LABEL[m] for m in METHODS], y=ylab, zmin=1, zmax=100, colorscale="RdYlGn_r",
                             text=z.astype(int).astype(str), texttemplate="%{text}", colorbar=dict(title="順位", x=1.02),
                             hovertemplate="%{y}<br>%{x}<br>順位 %{z:.0f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Heatmap(z=d, x=[f"{a} − {b}" for a, b in DIFF], y=ylab, zmid=0, zmin=-40, zmax=40, colorscale="RdBu_r", showscale=False,
                             text=[[f"{v:+.0f}" for v in r] for r in d], texttemplate="%{text}",
                             hovertemplate="%{y}<br>%{x} %{z:+.0f}<extra></extra>"), row=1, col=2)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(**LAYOUT, height=24 * len(ylab) + 200, margin=dict(l=190),
                      title="既知遺伝子の順位（緑＝上位）。右の差：青＝右側の方式で順位が上がった、赤＝下がった")
    return fig


def fig_scatter(data):
    fig = make_subplots(rows=1, cols=5, subplot_titles=[n for _, n in DISEASES], horizontal_spacing=0.035)
    for j, (key, name) in enumerate(DISEASES, start=1):
        w = data[key]
        fig.layout.annotations[j - 1].text = f"{name}（ρ={w['M1'].corr(w['M2'], method='spearman'):.2f}）"
        for cat, jp, col, mk in CAT:
            s = w[w.cat2.eq(cat)]
            if s.empty: continue
            fig.add_trace(go.Scatter(x=s["M1"], y=s["M2"], mode="markers", name=jp, legendgroup=cat, showlegend=(j == 1),
                                     marker=dict(color=col, symbol=mk, size=9 if cat in ("known", "random_same") else 6,
                                                 opacity=0.95 if cat in ("known", "random_same") else 0.55, line=dict(color="white", width=0.5)),
                                     customdata=np.c_[s["symbol"], [jp] * len(s)],
                                     hovertemplate="<b>%{customdata[0]}</b>（%{customdata[1]}）<br>M1 %{x:.2f}<br>M2 %{y:.2f}<extra></extra>"), row=1, col=j)
        lo, hi = float(np.nanmin(w[["M1", "M2"]].values)), float(np.nanmax(w[["M1", "M2"]].values))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#c3c2b7", dash="dot"), showlegend=False, hoverinfo="skip"), row=1, col=j)
    fig.update_xaxes(title_text="M1（対数オッズ）", row=1, col=3); fig.update_yaxes(title_text="M2（対数オッズ）", row=1, col=1)
    fig.update_layout(**LAYOUT, height=420, title="遺伝子ごとの点数（Yes/No）：横＝M1、縦＝M2。点線より上＝M2 で上がった")
    return fig


def main():
    data = load(); t = table(data)
    pd.set_option("display.width", 200)
    print(t.round(3).to_string(index=False))
    print("5疾患平均:\n" + t.groupby("方式")[["既知 vs その他", "既知 vs ダミー", "候補 vs ダミー", "ダミーの Yes 率"]].mean().round(3).to_string())
    figs = [fig_auc(t), fig_side(t), fig_rank(data), fig_scatter(data)]
    out = os.path.join(ROOT, "outputs", "m2_charts.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("<html><head><meta charset='utf-8'><title>M1 と M2 の比較</title></head><body style='max-width:1300px;margin:auto'>"
                 + "".join(f.to_html(full_html=False, include_plotlyjs=(True if i == 0 else False)) for i, f in enumerate(figs)) + "</body></html>")
    print("saved:", out)
    return data, t, figs


if __name__ == "__main__":
    main()
