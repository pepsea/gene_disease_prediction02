"""M3 と、その文章だけ変えた3版（M3s 短く / M3d 詳しく / M3w 言い方を変更）を比べる図。終わった疾患から順に描ける。

使い方: python scripts/plot_m3v.py                 # 結果がそろっている疾患をすべて
        python scripts/plot_m3v.py --disease ra     # 1疾患だけ
出力: outputs/m3v_charts.html（全体）/ outputs/m3v_<疾患>.html（1疾患）
図: (1) AUC 3種とダミーの Yes 率  (2) 既知遺伝子の順位  (3) 全遺伝子の点数の分布
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))
from rescue_report import ROOT, auc, wide_scores

DISEASES = {"achondroplasia": "軟骨無形成症", "ra": "関節リウマチ", "prostate_cancer": "前立腺がん", "scz": "統合失調症", "cystinuria": "シスチン尿症"}
METHODS = ["M2", "M3", "M3s", "M3d", "M3w"]                        # --methods で差し替え可（先頭2つは参考、3つ目以降を M3 と比べる）
LABEL = {"M2": "M2（現行ノートブック）", "M3": "M3（基準）", "M3s": "M3s 短く", "M3d": "M3d 詳しく", "M3w": "M3w 言い方を変更",
         "M3r1": "M3r1 短く＋道筋", "M3r2": "M3r2 短く＋答え方を先に", "M3r3": "M3r3 短く＋1文", "M4": "M4 M3s＋(b) this gene", "M4L": "M4L M4＋ラベル"}
COLOR = {"M2": "#c9c8c2", "M3": "#123f7a", "M3s": "#1baf7a", "M3d": "#8a5cd1", "M3w": "#eda100", "M3r1": "#d64545", "M3r2": "#2a78d6", "M3r3": "#8a5cd1", "M4": "#d64545", "M4L": "#eda100"}
CAT = [("known", "既知", "#2a78d6", "circle"), ("candidate", "候補", "#eb6834", "square"), ("random", "ダミー", "#1baf7a", "diamond")]
LAYOUT = dict(template="plotly_white", font=dict(family="Hiragino Sans, Noto Sans JP, sans-serif", size=12))
sig = lambda x: 1 / (1 + np.exp(-x))


def ready():
    out = {}
    for key in DISEASES:
        w, _ = wide_scores(key)
        if all(m in w.columns for m in METHODS): out[key] = w
    return out


def table(data):
    rows = []
    for key, w in data.items():
        k, r, c = w.category.eq("known"), w.category.eq("random"), w.category.eq("candidate")
        for m in METHODS:
            rows.append({"疾患": DISEASES[key], "方式": m, "既知 vs その他": auc(w.loc[k, m], w.loc[~k, m]), "既知 vs ダミー": auc(w.loc[k, m], w.loc[r, m]),
                         "候補 vs ダミー": auc(w.loc[c, m], w.loc[r, m]), "ダミーの Yes 率": sig(w.loc[r, m].median())})
    return pd.DataFrame(rows)


def fig_summary(t):
    cols = ["既知 vs その他", "既知 vs ダミー", "候補 vs ダミー", "ダミーの Yes 率"]
    fig = make_subplots(rows=1, cols=4, subplot_titles=[f"AUC {c}" if "Yes" not in c else c + "（低いほど良い）" for c in cols], horizontal_spacing=0.05)
    names = list(dict.fromkeys(t["疾患"])) + (["平均"] if t["疾患"].nunique() > 1 else [])
    for j, col in enumerate(cols, start=1):
        for m in METHODS:
            s = t[t["方式"] == m].set_index("疾患")[col]
            y = list(s.reindex(names[:len(s)])) + ([s.mean()] if len(names) > len(s) else [])
            fig.add_trace(go.Bar(x=names, y=y, name=LABEL[m], marker_color=COLOR[m], legendgroup=m, showlegend=(j == 1),
                                 text=[f"{v:.3f}" if "Yes" not in col else f"{v:.2f}" for v in y], textposition="outside", textfont=dict(size=9),
                                 hovertemplate=f"<b>{LABEL[m]}</b><br>%{{x}}<br>{col} %{{y:.3f}}<extra></extra>"), row=1, col=j)
        fig.update_yaxes(range=[0.5, 1.03] if "Yes" not in col else [0, 1.05], row=1, col=j)
    fig.update_layout(**LAYOUT, barmode="group", height=460, title="M3 の文章を変えた3版の比較（AUC は高いほど、ダミーの Yes 率は低いほど良い）")
    return fig


def fig_rank(data):
    rows, ylab = [], []
    for key, w in data.items():
        k = w.category.eq("known")
        rk = {m: w[m].rank(ascending=False, method="min") for m in METHODS}
        for i in w.index[k].to_series().sort_values(key=lambda s: rk[METHODS[1]][s], ascending=False):
            rows.append([rk[m][i] for m in METHODS]); ylab.append(f"{DISEASES[key]}｜{w.at[i, 'symbol']}")
    z = np.array(rows, float); d = z[:, 2:] - z[:, [1]]                  # M3s/M3d/M3w − M3（負 = 上がった）
    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], shared_yaxes=True, horizontal_spacing=0.02,
                        subplot_titles=("順位（100中）", f"{METHODS[1]} との差（負＝上がった）"))
    fig.add_trace(go.Heatmap(z=z, x=[LABEL[m] for m in METHODS], y=ylab, zmin=1, zmax=100, colorscale="RdYlGn_r",
                             text=z.astype(int).astype(str), texttemplate="%{text}", colorbar=dict(title="順位", x=1.02),
                             hovertemplate="%{y}<br>%{x}<br>順位 %{z:.0f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Heatmap(z=d, x=[f"{m} − {METHODS[1]}" for m in METHODS[2:]], y=ylab, zmid=0, zmin=-30, zmax=30, colorscale="RdBu_r", showscale=False,
                             text=[[f"{v:+.0f}" for v in r] for r in d], texttemplate="%{text}",
                             hovertemplate="%{y}<br>%{x} %{z:+.0f}<extra></extra>"), row=1, col=2)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(**LAYOUT, height=26 * len(ylab) + 200, margin=dict(l=190), title=f"既知遺伝子の順位（緑＝上位）。右：青＝{METHODS[1]} より上がった、赤＝下がった")
    return fig


def fig_strip(data):
    keys = list(data)
    fig = make_subplots(rows=len(keys), cols=1, subplot_titles=[DISEASES[k] for k in keys], vertical_spacing=0.08 if len(keys) > 1 else 0.1)
    rng = np.random.default_rng(0)
    for i, key in enumerate(keys, start=1):
        w = data[key]
        for cat, jp, col, mk in CAT:
            s = w[w.category.eq(cat)]
            for j, m in enumerate(METHODS):
                fig.add_trace(go.Scatter(x=j + rng.uniform(-0.28, 0.28, len(s)), y=s[m], mode="markers", name=jp, legendgroup=cat,
                                         showlegend=(i == 1 and j == 0),
                                         marker=dict(color=col, symbol=mk, size=9 if cat == "known" else 6, opacity=0.95 if cat == "known" else 0.5,
                                                     line=dict(color="white", width=0.5)),
                                         customdata=np.c_[s["symbol"], [jp] * len(s), [LABEL[m]] * len(s)],
                                         hovertemplate="<b>%{customdata[0]}</b>（%{customdata[1]}）<br>%{customdata[2]}<br>対数オッズ %{y:.2f}<extra></extra>"),
                              row=i, col=1)
        fig.update_xaxes(tickmode="array", tickvals=list(range(len(METHODS))), ticktext=[LABEL[m] for m in METHODS], row=i, col=1)
        fig.add_hline(y=0, line=dict(color="#c3c2b7", dash="dot"), row=i, col=1)
    fig.update_layout(**LAYOUT, height=380 * len(keys), title="全遺伝子の点数（両順の対数オッズ平均。点線より上＝Yes 寄り）")
    return fig


def main():
    global METHODS
    ap = argparse.ArgumentParser(); ap.add_argument("--disease", default=None)
    ap.add_argument("--methods", nargs="+", default=None); ap.add_argument("--tag", default="m3v"); a = ap.parse_args()
    if a.methods: METHODS = a.methods
    data = ready()
    if a.disease: data = {a.disease: data[a.disease]}
    t = table(data)
    pd.set_option("display.width", 200)
    print(t.round(3).to_string(index=False))
    if len(data) > 1: print("平均:\n" + t.groupby("方式")[["既知 vs その他", "既知 vs ダミー", "候補 vs ダミー", "ダミーの Yes 率"]].mean().round(3).to_string())
    figs = [fig_summary(t), fig_rank(data), fig_strip(data)]
    name = f"{a.tag}_{a.disease}.html" if a.disease else f"{a.tag}_charts.html"
    out = os.path.join(ROOT, "outputs", name)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("<html><head><meta charset='utf-8'><title>M3 の文章比較</title></head><body style='max-width:1300px;margin:auto'>"
                 + "".join(f.to_html(full_html=False, include_plotlyjs=(True if i == 0 else False)) for i, f in enumerate(figs)) + "</body></html>")
    print("saved:", out)
    return data, t, figs


if __name__ == "__main__":
    main()
