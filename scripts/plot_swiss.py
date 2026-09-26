"""スイス式（swiss.py）とランダムな組み合わせ（rank_variants.py → rank_bt.py）を、同じ質問数（8ラウンド）で比べる図。

使い方: python scripts/plot_swiss.py [--disease ra]     # 省略時は結果がそろっている疾患すべて
出力: outputs/swiss_charts.html（または outputs/swiss_<疾患>.html）。plotly.js は埋め込み。
図: (1) AUC と上位の指標  (2) 既知遺伝子の順位  (3) ランダム vs スイスの強さの散布図  (4) ラウンドごとの「同じ組の強さのばらつき」
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))
from rescue_report import ROOT, auc
from tournament import spearman

DISEASES = [("achondroplasia", "軟骨無形成症"), ("ra", "関節リウマチ"), ("prostate_cancer", "前立腺がん"), ("scz", "統合失調症"), ("cystinuria", "シスチン尿症")]
METHODS = [("rand_mean_p", "ランダム × mean_p（04 の現行）", "#c9c8c2"), ("rand_luce", "ランダム × Bradley-Terry", "#9ec0ec"),
           ("swiss_luce", "スイス式 × Bradley-Terry", "#2a78d6"), ("swiss_elo", "スイス式 × Elo", "#8a5cd1")]
CAT = [("known", "既知", "#2a78d6", "circle"), ("candidate", "候補", "#eb6834", "square"), ("random", "ダミー", "#1baf7a", "diamond")]
LAYOUT = dict(template="plotly_white", font=dict(family="Hiragino Sans, Noto Sans JP, sans-serif", size=12))


def load(key):
    p = os.path.join(ROOT, "outputs", f"{key}_set100_swiss.csv")
    if not os.path.exists(p): return None
    s = pd.read_csv(p).set_index("symbol")
    r = pd.read_csv(os.path.join(ROOT, "outputs", f"{key}_set100_rank_bt.csv")).set_index("symbol")
    t = pd.DataFrame({"category": s["category"], "rand_mean_p": r["mean_p"], "rand_luce": r["luce"], "swiss_luce": s["luce"], "swiss_elo": s["elo"]})
    for m, *_ in METHODS: t[m + "_rank"] = t[m].rank(ascending=False, method="min").astype(int)
    return t.reset_index()


def metrics(t):
    kn, rd, cd = t.category.eq("known"), t.category.eq("random"), t.category.eq("candidate")
    out = []
    for m, lab, _ in METHODS:
        rk = t[m + "_rank"]
        top30 = rk <= 30
        out.append({"方式": m, "既知 vs その他": auc(t.loc[kn, m], t.loc[~kn, m]), "既知 vs ダミー": auc(t.loc[kn, m], t.loc[rd, m]),
                    "候補 vs ダミー": auc(t.loc[cd, m], t.loc[rd, m]),
                    "上位30内の既知 vs その他": auc(t.loc[kn & top30, m], t.loc[~kn & top30, m]),
                    "既知の平均順位": rk[kn].mean(), "上位10件中の既知": int((rk <= 10)[kn].sum())})
    return pd.DataFrame(out)


def spread(key):
    """ラウンドごとに、同じグループ内の強さ（最終推定）の標準偏差の平均。スイス式なら小さくなる（近い強さどうしで組んでいる）。"""
    L = pd.read_csv(os.path.join(ROOT, "outputs", f"{key}_set100_swiss_long.csv"))
    s = pd.read_csv(os.path.join(ROOT, "outputs", f"{key}_set100_swiss.csv")).set_index("symbol")["luce"]
    L["s"] = L["symbol"].map(s)
    return L.groupby(["round", "group"])["s"].std().groupby("round").mean()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--disease", default=None); a = ap.parse_args()
    keys = [(k, n) for k, n in DISEASES if (a.disease is None or k == a.disease)]
    data = {k: t for k, _ in keys if (t := load(k)) is not None}
    names = {k: n for k, n in DISEASES}
    T = pd.concat([metrics(t).assign(疾患=names[k]) for k, t in data.items()])
    pd.set_option("display.width", 220)
    print(T.round(3).to_string(index=False))
    if len(data) > 1: print("平均:\n" + T.groupby("方式").mean(numeric_only=True).round(3).to_string())

    cols = ["既知 vs その他", "上位30内の既知 vs その他", "既知の平均順位", "上位10件中の既知"]
    xs = [names[k] for k in data] + (["平均"] if len(data) > 1 else [])
    f1 = make_subplots(rows=1, cols=4, subplot_titles=[c + ("（小さいほど良い）" if c == "既知の平均順位" else "") for c in cols], horizontal_spacing=0.05)
    for j, c in enumerate(cols, start=1):
        for m, lab, color in METHODS:
            s = T[T["方式"] == m].set_index("疾患")[c]; y = list(s.reindex(xs[:len(s)])) + ([s.mean()] if len(xs) > len(s) else [])
            f1.add_trace(go.Bar(x=xs, y=y, name=lab, marker_color=color, legendgroup=m, showlegend=(j == 1),
                                hovertemplate=f"<b>{lab}</b><br>%{{x}}<br>{c} %{{y:.3f}}<extra></extra>"), row=1, col=j)
    f1.update_yaxes(range=[0.4, 1.02], row=1, col=1); f1.update_yaxes(range=[0.2, 1.02], row=1, col=2)
    f1.update_layout(**LAYOUT, barmode="group", height=460, title="スイス式とランダムの比較（同じ8ラウンド＝同じ質問数）")

    zr, ylab = [], []
    for k, t in data.items():
        for _, r in t[t.category.eq("known")].sort_values("rand_mean_p_rank", ascending=False).iterrows():
            zr.append([r[m + "_rank"] for m, *_ in METHODS]); ylab.append(f"{names[k]}｜{r['symbol']}")
    z = np.array(zr, float)
    f2 = go.Figure(go.Heatmap(z=z, x=[lab for _, lab, _ in METHODS], y=ylab, zmin=1, zmax=100, colorscale="RdYlGn_r",
                              text=z.astype(int).astype(str), texttemplate="%{text}", colorbar=dict(title="順位"),
                              hovertemplate="%{y}<br>%{x}<br>順位 %{z:.0f}<extra></extra>"))
    f2.update_layout(**LAYOUT, height=24 * len(ylab) + 200, margin=dict(l=190, t=120), yaxis=dict(autorange="reversed"),
                     xaxis=dict(side="top"), title="既知遺伝子の順位（100中、緑＝上位）")

    f3 = make_subplots(rows=1, cols=len(data), subplot_titles=[names[k] for k in data], horizontal_spacing=0.035)
    for j, (k, t) in enumerate(data.items(), start=1):
        f3.layout.annotations[j - 1].text = f"{names[k]}（ρ={spearman(t['rand_luce'], t['swiss_luce']):.2f}）"
        for cat, jp, col, mk in CAT:
            s = t[t.category.eq(cat)]
            f3.add_trace(go.Scatter(x=s["rand_luce"], y=s["swiss_luce"], mode="markers", name=jp, legendgroup=cat, showlegend=(j == 1),
                                    marker=dict(color=col, symbol=mk, size=9 if cat == "known" else 6, opacity=0.9 if cat == "known" else 0.55,
                                                line=dict(color="white", width=0.5)),
                                    customdata=np.c_[s["symbol"], [jp] * len(s), s["rand_luce_rank"], s["swiss_luce_rank"]],
                                    hovertemplate="<b>%{customdata[0]}</b>（%{customdata[1]}）<br>ランダム %{x:.2f}（%{customdata[2]}位）"
                                                  "<br>スイス %{y:.2f}（%{customdata[3]}位）<extra></extra>"), row=1, col=j)
    f3.update_xaxes(title_text="ランダム × BT", row=1, col=(len(data) + 1) // 2); f3.update_yaxes(title_text="スイス式 × BT", row=1, col=1)
    f3.update_layout(**LAYOUT, height=420, title="遺伝子ごとの強さ：横＝ランダムな組み合わせ、縦＝スイス式（どちらも Bradley-Terry）")

    f4 = go.Figure()
    for k in data:
        sp = spread(k)
        f4.add_trace(go.Scatter(x=[f"R{r + 1}" for r in sp.index], y=sp.values, mode="lines+markers", name=names[k],
                                hovertemplate=f"{names[k]}<br>%{{x}}<br>同じ組の強さのばらつき %{{y:.2f}}<extra></extra>"))
    f4.add_vrect(x0=-0.5, x1=1.5, fillcolor="#f1f0ec", opacity=0.6, line_width=0, annotation_text="ランダム", annotation_position="top left")
    f4.update_layout(**LAYOUT, height=380, yaxis_title="同じグループ内の強さの標準偏差（平均）",
                     title="ラウンドごとの組み方の確認：3ラウンド目から強さの近い遺伝子どうしで組んでいれば、ばらつきが下がる")

    out = os.path.join(ROOT, "outputs", f"swiss_{a.disease}.html" if a.disease else "swiss_charts.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("<html><head><meta charset='utf-8'><title>スイス式トーナメント</title></head><body style='max-width:1300px;margin:auto'>"
                 + "".join(f.to_html(full_html=False, include_plotlyjs=(True if i == 0 else False)) for i, f in enumerate((f1, f2, f3, f4))) + "</body></html>")
    print("saved:", out)
    return (f1, f2, f3, f4)


if __name__ == "__main__":
    main()
