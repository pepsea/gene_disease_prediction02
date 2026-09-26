"""ノートブック 03（Yes/No の M3s）と 04（スイス式 × Bradley-Terry）の点数を統合したときの比較図（plotly.js 埋め込み）。

使い方: python scripts/plot_combined.py
入力: outputs/<疾患>_set100_qvariants.csv（M3s）と outputs/<疾患>_set100_swiss.csv（luce）
出力: outputs/combined_charts.html と outputs/<疾患>_set100_combined.csv
統合の仕方: z平均（標準化した点数の平均）/ 順位平均 / 順位の悪い方（両方で上位＝AND 的）/ 順位の良い方（どちらかで上位＝OR 的）
"""
import os, sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))
from rescue_report import ROOT, auc, wide_scores

DISEASES = [("achondroplasia", "軟骨無形成症"), ("ra", "関節リウマチ"), ("prostate_cancer", "前立腺がん"), ("scz", "統合失調症"), ("cystinuria", "シスチン尿症")]
METHODS = [("m03", "03 のみ（Yes/No M3s）", "#c9c8c2"), ("m04", "04 のみ（スイス式 × BT）", "#9ec0ec"), ("zmean", "統合：z平均", "#1baf7a"),
           ("rmean", "統合：順位平均", "#eda100"), ("rworst", "統合：順位の悪い方（両方で上位）", "#2a78d6"), ("rbest", "統合：順位の良い方（どちらかで上位）", "#8a5cd1")]
LAYOUT = dict(template="plotly_white", font=dict(family="Hiragino Sans, Noto Sans JP, sans-serif", size=12))
z = lambda x: (x - x.mean()) / x.std()


def combine(key):
    t = wide_scores(key)[0].set_index("symbol")[["category", "M3s"]].rename(columns={"M3s": "m03"})
    t = t.join(pd.read_csv(os.path.join(ROOT, "outputs", f"{key}_set100_swiss.csv")).set_index("symbol")["luce"].rename("m04"))
    r03, r04 = t["m03"].rank(ascending=False), t["m04"].rank(ascending=False)
    t["zmean"] = (z(t["m03"]) + z(t["m04"])) / 2
    t["rmean"] = -(r03 + r04) / 2
    t["rworst"] = -np.maximum(r03, r04) - (r03 + r04) / 1000            # 同点は順位平均で分ける
    t["rbest"] = -np.minimum(r03, r04) - (r03 + r04) / 1000
    for m, *_ in METHODS: t[m + "_rank"] = t[m].rank(ascending=False, method="min").astype(int)
    t.reset_index().to_csv(os.path.join(ROOT, "outputs", f"{key}_set100_combined.csv"), index=False)
    return t.reset_index()


def main():
    data = {k: combine(k) for k, _ in DISEASES}; names = dict(DISEASES)
    rows = []
    for k, t in data.items():
        kn, rd, cd = t.category.eq("known"), t.category.eq("random"), t.category.eq("candidate")
        for m, *_ in METHODS:
            rk = t[m + "_rank"]
            rows.append({"疾患": names[k], "方式": m, "既知 vs その他": auc(t.loc[kn, m], t.loc[~kn, m]), "既知 vs ダミー": auc(t.loc[kn, m], t.loc[rd, m]),
                         "候補 vs ダミー": auc(t.loc[cd, m], t.loc[rd, m]), "既知の平均順位": rk[kn].mean(), "上位10件中の既知": int((rk <= 10)[kn].sum())})
    T = pd.DataFrame(rows)
    print(T.groupby("方式").mean(numeric_only=True).round(3))
    cols = ["既知 vs その他", "既知 vs ダミー", "候補 vs ダミー", "既知の平均順位"]
    xs = [n for _, n in DISEASES] + ["平均"]
    f1 = make_subplots(rows=1, cols=4, subplot_titles=[("AUC " + c if "AUC" not in c and "順位" not in c else c) + ("（小さいほど良い）" if "順位" in c else "") for c in cols],
                       horizontal_spacing=0.05)
    for j, c in enumerate(cols, start=1):
        for m, lab, color in METHODS:
            s = T[T["方式"] == m].set_index("疾患")[c]; y = list(s.reindex(xs[:-1])) + [s.mean()]
            f1.add_trace(go.Bar(x=xs, y=y, name=lab, marker_color=color, legendgroup=m, showlegend=(j == 1),
                                hovertemplate=f"<b>{lab}</b><br>%{{x}}<br>{c} %{{y:.3f}}<extra></extra>"), row=1, col=j)
        if j <= 3: f1.update_yaxes(range=[0.5, 1.02], row=1, col=j)
    f1.update_layout(**LAYOUT, barmode="group", height=480, title="03 と 04 を統合したスコアの比較（淡色＝単独、濃色＝統合）")

    f2 = make_subplots(rows=1, cols=5, subplot_titles=[n for _, n in DISEASES], horizontal_spacing=0.035)
    CAT = [("known", "既知", "#2a78d6", "circle"), ("candidate", "候補", "#eb6834", "square"), ("random", "ダミー", "#1baf7a", "diamond")]
    for j, (k, name) in enumerate(DISEASES, start=1):
        t = data[k]
        f2.layout.annotations[j - 1].text = f"{name}（ρ={t['m03'].corr(t['m04'], method='spearman'):.2f}）"
        for cat, jp, col, mk in CAT:
            s = t[t.category.eq(cat)]
            f2.add_trace(go.Scatter(x=s["m03_rank"], y=s["m04_rank"], mode="markers", name=jp, legendgroup=cat, showlegend=(j == 1),
                                    marker=dict(color=col, symbol=mk, size=9 if cat == "known" else 6, opacity=0.9 if cat == "known" else 0.5,
                                                line=dict(color="white", width=0.5)),
                                    customdata=np.c_[s["symbol"], [jp] * len(s), s["rworst_rank"]],
                                    hovertemplate="<b>%{customdata[0]}</b>（%{customdata[1]}）<br>03 の順位 %{x}<br>04 の順位 %{y}"
                                                  "<br>統合（悪い方）%{customdata[2]}位<extra></extra>"), row=1, col=j)
        f2.update_xaxes(autorange="reversed", row=1, col=j); f2.update_yaxes(autorange="reversed", row=1, col=j)
    f2.update_xaxes(title_text="03 の順位", row=1, col=3); f2.update_yaxes(title_text="04 の順位", row=1, col=1)
    f2.update_layout(**LAYOUT, height=420, title="03 と 04 の順位（右上＝両方で上位。ρ＝順位相関）")

    zr, ylab = [], []
    for k, t in data.items():
        for _, r in t[t.category.eq("known")].sort_values("m03_rank", ascending=False).iterrows():
            zr.append([r[m + "_rank"] for m, *_ in METHODS]); ylab.append(f"{names[k]}｜{r['symbol']}")
    zz = np.array(zr, float)
    f3 = go.Figure(go.Heatmap(z=zz, x=[lab for _, lab, _ in METHODS], y=ylab, zmin=1, zmax=100, colorscale="RdYlGn_r",
                              text=zz.astype(int).astype(str), texttemplate="%{text}", colorbar=dict(title="順位"),
                              hovertemplate="%{y}<br>%{x}<br>順位 %{z:.0f}<extra></extra>"))
    f3.update_layout(**LAYOUT, height=24 * len(ylab) + 200, margin=dict(l=190, t=140), yaxis=dict(autorange="reversed"),
                     xaxis=dict(side="top", tickangle=-20), title="既知遺伝子の順位（100中、緑＝上位）")
    out = os.path.join(ROOT, "outputs", "combined_charts.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("<html><head><meta charset='utf-8'><title>03 と 04 の統合</title></head><body style='max-width:1300px;margin:auto'>"
                 + "".join(f.to_html(full_html=False, include_plotlyjs=(True if i == 0 else False)) for i, f in enumerate((f1, f2, f3))) + "</body></html>")
    print("saved:", out)
    return (f1, f2, f3)


if __name__ == "__main__":
    main()
