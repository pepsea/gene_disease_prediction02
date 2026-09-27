"""機能情報つき（M3sF）と M3s を比べる図（ノートブック 06 の検証用。plotly.js 埋め込み）。

使い方: python scripts/plot06.py --disease ra            # 1疾患
        python scripts/plot06.py                         # 結果がそろっている疾患すべて
出力: outputs/func_<疾患>.html / outputs/func_charts.html
図: (1) 評価指標の比較  (2) 遺伝子ごとの点数 M3s × M3sF  (3) 既知遺伝子の順位
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))
from rescue_report import ROOT, wide_scores
from eval06 import evaluate

DISEASES = [("achondroplasia", "軟骨無形成症"), ("ra", "関節リウマチ"), ("prostate_cancer", "前立腺がん"), ("scz", "統合失調症"), ("cystinuria", "シスチン尿症")]
M = [("M3s", "M3s（現行）", "#c9c8c2"), ("M3sF", "M3sF（機能情報つき）", "#2a78d6")]
CAT = [("known", "既知", "#2a78d6", "circle"), ("candidate", "候補", "#eb6834", "square"), ("random", "ダミー", "#1baf7a", "diamond")]
LAYOUT = dict(template="plotly_white", font=dict(family="Hiragino Sans, Noto Sans JP, sans-serif", size=12))
COLS = ["候補 vs ダミー", "候補 vs ダミー（知名度をそろえる）", "ダミーの Yes 率", "既知 vs その他", "既知 vs ダミー（知名度をそろえる）"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--disease", default=None); a = ap.parse_args()
    data = {}
    for k, n in DISEASES:
        if a.disease and k != a.disease: continue
        w = wide_scores(k)[0]
        if "M3sF" in w.columns: data[k] = w
    names = dict(DISEASES)
    T = pd.concat([evaluate(w, [m for m, *_ in M]).assign(疾患=names[k]) for k, w in data.items()])
    print(T.drop(columns=["取りこぼしの順位"]).round(3).to_string(index=False))
    if len(data) > 1: print("平均:\n" + T.groupby("score")[COLS].mean().round(3).to_string())

    xs = [names[k] for k in data] + (["平均"] if len(data) > 1 else [])
    f1 = make_subplots(rows=1, cols=len(COLS), subplot_titles=[c.replace("（知名度をそろえる）", "<br>（知名度をそろえる）") + ("（低いほど良い）" if "Yes" in c else "") for c in COLS],
                       horizontal_spacing=0.04)
    for j, c in enumerate(COLS, start=1):
        for m, lab, color in M:
            s = T[T["score"] == m].set_index("疾患")[c]; y = list(s.reindex(xs[:len(s)])) + ([s.mean()] if len(xs) > len(s) else [])
            f1.add_trace(go.Bar(x=xs, y=y, name=lab, marker_color=color, legendgroup=m, showlegend=(j == 1),
                                text=[f"{v:.2f}" for v in y], textposition="outside", textfont=dict(size=9),
                                hovertemplate=f"<b>{lab}</b><br>%{{x}}<br>{c} %{{y:.3f}}<extra></extra>"), row=1, col=j)
        f1.update_yaxes(range=[0, 1.1] if "Yes" in c else [0.4, 1.1], row=1, col=j)
    f1.update_layout(**LAYOUT, barmode="group", height=460, margin=dict(t=110),
                     title="機能情報（UniProt の機能・GO・Reactome、疾患への言及は除去）をプロンプトに入れた効果")

    f2 = make_subplots(rows=1, cols=len(data), subplot_titles=[names[k] for k in data], horizontal_spacing=0.04)
    for j, (k, w) in enumerate(data.items(), start=1):
        for cat, jp, col, mk in CAT:
            s = w[w.category.eq(cat)]
            f2.add_trace(go.Scatter(x=s["M3s"], y=s["M3sF"], mode="markers", name=jp, legendgroup=cat, showlegend=(j == 1),
                                    marker=dict(color=col, symbol=mk, size=9 if cat == "known" else 6, opacity=0.9 if cat == "known" else 0.55,
                                                line=dict(color="white", width=0.5)),
                                    customdata=np.c_[s["symbol"], [jp] * len(s)],
                                    hovertemplate="<b>%{customdata[0]}</b>（%{customdata[1]}）<br>M3s %{x:.2f}<br>M3sF %{y:.2f}<extra></extra>"), row=1, col=j)
        lo, hi = float(np.nanmin(w[["M3s", "M3sF"]].values)), float(np.nanmax(w[["M3s", "M3sF"]].values))
        f2.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#c3c2b7", dash="dot"), showlegend=False, hoverinfo="skip"), row=1, col=j)
        f2.add_hline(y=0, line=dict(color="#e6e5e0"), row=1, col=j); f2.add_vline(x=0, line=dict(color="#e6e5e0"), row=1, col=j)
    f2.update_xaxes(title_text="M3s（対数オッズ）"); f2.update_yaxes(title_text="M3sF（対数オッズ）", row=1, col=1)
    f2.update_layout(**LAYOUT, height=440, width=max(500, 380 * len(data)),
                     title="遺伝子ごとの点数：点線より上＝機能情報で上がった。0 より上＝Yes 寄り")

    zr, ylab = [], []
    for k, w in data.items():
        rk = {m: w[m].rank(ascending=False, method="min") for m, *_ in M}
        for i in w.index[w.category.eq("known")].to_series().sort_values(key=lambda s: rk["M3s"][s], ascending=False):
            zr.append([rk[m][i] for m, *_ in M] + [rk["M3sF"][i] - rk["M3s"][i]]); ylab.append(f"{names[k]}｜{w.at[i, 'symbol']}")
    z = np.array(zr, float)
    f3 = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3], shared_yaxes=True, horizontal_spacing=0.02, subplot_titles=("順位（100中）", "差（負＝上がった）"))
    f3.add_trace(go.Heatmap(z=z[:, :2], x=[lab for _, lab, _ in M], y=ylab, zmin=1, zmax=100, colorscale="RdYlGn_r", colorbar=dict(title="順位", x=1.02),
                            text=z[:, :2].astype(int).astype(str), texttemplate="%{text}", hovertemplate="%{y}<br>%{x}<br>順位 %{z:.0f}<extra></extra>"), row=1, col=1)
    f3.add_trace(go.Heatmap(z=z[:, 2:], x=["M3sF − M3s"], y=ylab, zmid=0, zmin=-30, zmax=30, colorscale="RdBu_r", showscale=False,
                            text=[[f"{v:+.0f}"] for v in z[:, 2]], texttemplate="%{text}", hovertemplate="%{y}<br>%{z:+.0f}<extra></extra>"), row=1, col=2)
    f3.update_yaxes(autorange="reversed")
    f3.update_layout(**LAYOUT, height=26 * len(ylab) + 200, margin=dict(l=190), title="既知遺伝子の順位（緑＝上位、右の青＝機能情報で上がった）")

    out = os.path.join(ROOT, "outputs", f"func_{a.disease}.html" if a.disease else "func_charts.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("<html><head><meta charset='utf-8'><title>機能情報の効果</title></head><body style='max-width:1300px;margin:auto'>"
                 + "".join(f.to_html(full_html=False, include_plotlyjs=(True if i == 0 else False)) for i, f in enumerate((f1, f2, f3))) + "</body></html>")
    print("saved:", out)
    return (f1, f2, f3)


if __name__ == "__main__":
    main()
