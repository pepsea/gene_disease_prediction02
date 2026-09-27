"""方向の推定 ステップ1の図（plotly.js 埋め込み）。
使い方: python scripts/plot_direction_step1.py
入力: outputs/direction/step1_known37.csv（direction_step1.py）
出力: outputs/direction/step1_charts.html
図: (1) 言い回し × 機能情報ごとの当たり具合（阻害・活性化の正解だけで、差の符号の正解率と AUC）
    (2) 遺伝子ごとの散布図（横＝阻害の対数オッズ、縦＝活性化の対数オッズ。色＝正解の方向）
    (3) 遺伝子ごとの差（阻害 − 活性化）のヒートマップ
"""
import os, sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LAYOUT = dict(template="plotly_white", font=dict(family="Hiragino Sans, Noto Sans JP, sans-serif", size=12))
COL = {"inhibit": "#d64545", "activate": "#2a78d6", "neither": "#8a8984"}
JP = {"inhibit": "阻害が正解", "activate": "活性化が正解", "neither": "どちらでもない（目印・除去・薬なし）"}


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0: return np.nan
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def main():
    w = pd.read_csv(os.path.join(ROOT, "outputs", "direction", "step1_known37.csv"))
    w["setting"] = w["variant"] + "・機能情報" + w["function"].map({"no": "なし", "yes": "あり"})
    rows = []
    for st, d in w.groupby("setting"):
        dd = d[d["net_effect"].isin(["inhibit", "activate"])]
        correct = np.where(dd["net_effect"] == "inhibit", dd["diff"] > 0, dd["diff"] < 0)
        rows.append({"設定": st, "正解率（全体）": correct.mean(),
                     "阻害の正解率": (dd.loc[dd.net_effect == "inhibit", "diff"] > 0).mean(),
                     "活性化の正解率": (dd.loc[dd.net_effect == "activate", "diff"] < 0).mean(),
                     "AUC（差で阻害と活性化を分ける）": auc(dd.loc[dd.net_effect == "inhibit", "diff"], dd.loc[dd.net_effect == "activate", "diff"]),
                     "どちらでもない：2問の高い方の平均": d.loc[d.net_effect == "neither", ["inhibit", "activate"]].max(axis=1).mean(),
                     "方向あり：2問の高い方の平均": dd[["inhibit", "activate"]].max(axis=1).mean()})
    T = pd.DataFrame(rows); pd.set_option("display.width", 220)
    print(T.round(3).to_string(index=False))

    cols = ["正解率（全体）", "阻害の正解率", "活性化の正解率", "AUC（差で阻害と活性化を分ける）"]
    f1 = make_subplots(rows=1, cols=4, subplot_titles=cols, shared_yaxes=True, horizontal_spacing=0.03)
    for j, c in enumerate(cols, start=1):
        f1.add_trace(go.Bar(x=T["設定"], y=T[c], showlegend=False, text=[f"{v:.2f}" for v in T[c]], textposition="outside",
                            marker_color=["#9ec0ec" if "なし" in s else "#2a78d6" for s in T["設定"]],
                            hovertemplate="%{x}<br>" + c + " %{y:.3f}<extra></extra>"), row=1, col=j)
        f1.add_hline(y=0.5, line=dict(color="#c3c2b7", dash="dot"), row=1, col=j)
    f1.update_yaxes(range=[0, 1.1]); f1.update_xaxes(tickangle=-40)
    n_i, n_a = int((w.net_effect == "inhibit").sum() / w.setting.nunique()), int((w.net_effect == "activate").sum() / w.setting.nunique())
    f1.update_layout(**LAYOUT, height=460, title=f"方向の当たり具合（阻害が正解 {n_i} 遺伝子・活性化が正解 {n_a} 遺伝子。淡色＝機能情報なし、濃色＝あり。点線＝偶然）")

    sets = list(T["設定"])
    f2 = make_subplots(rows=2, cols=3, subplot_titles=sets, horizontal_spacing=0.06, vertical_spacing=0.12)
    for k, st in enumerate(sets):
        d = w[w.setting == st]; r, cc = k % 2 + 1, k // 2 + 1
        for ne, col in COL.items():
            x = d[d.net_effect == ne]
            f2.add_trace(go.Scatter(x=x["inhibit"], y=x["activate"], mode="markers+text", text=x["symbol"], textposition="top center", textfont=dict(size=8),
                                    name=JP[ne], legendgroup=ne, showlegend=(k == 0), marker=dict(color=col, size=8),
                                    customdata=np.c_[x["symbol"], x["disease"], x["diff"]],
                                    hovertemplate="<b>%{customdata[0]}</b>（%{customdata[1]}）<br>阻害 %{x:.2f}<br>活性化 %{y:.2f}<br>差 %{customdata[2]:+.2f}<extra></extra>"),
                          row=r, col=cc)
        lo, hi = float(w[["inhibit", "activate"]].min().min()), float(w[["inhibit", "activate"]].max().max())
        f2.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#c3c2b7", dash="dot"), showlegend=False, hoverinfo="skip"), row=r, col=cc)
    f2.update_xaxes(title_text="阻害の対数オッズ"); f2.update_yaxes(title_text="活性化の対数オッズ")
    f2.update_layout(**LAYOUT, height=900, title="遺伝子ごとの答え：点線より右下＝阻害寄り、左上＝活性化寄り（赤＝阻害が正解、青＝活性化が正解、灰＝どちらでもない）")

    order = w[w.setting == sets[0]].sort_values(["net_effect", "disease", "symbol"])
    ylab = [f"{r.symbol}（{JP[r.net_effect][:4]}）" for r in order.itertuples()]
    Z = np.array([[w[(w.setting == st) & (w.symbol == r.symbol) & (w.disease == r.disease)]["diff"].iloc[0] for st in sets] for r in order.itertuples()])
    f3 = go.Figure(go.Heatmap(z=Z, x=sets, y=ylab, zmid=0, colorscale="RdBu_r", text=np.round(Z, 1), texttemplate="%{text}",
                              colorbar=dict(title="阻害 − 活性化"), hovertemplate="%{y}<br>%{x}<br>差 %{z:+.2f}<extra></extra>"))
    f3.update_layout(**LAYOUT, height=24 * len(ylab) + 200, margin=dict(l=220), yaxis=dict(autorange="reversed"), xaxis=dict(side="top", tickangle=-30),
                     title="遺伝子ごとの差（阻害 − 活性化の対数オッズ）：赤＝阻害寄り、青＝活性化寄り")

    out = os.path.join(ROOT, "outputs", "direction", "step1_charts.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("<html><head><meta charset='utf-8'><title>方向の推定 ステップ1</title></head><body style='max-width:1300px;margin:auto'>"
                 + "".join(f.to_html(full_html=False, include_plotlyjs=(True if i == 0 else False)) for i, f in enumerate((f1, f2, f3))) + "</body></html>")
    print("saved:", out)
    return T, (f1, f2, f3)


if __name__ == "__main__":
    main()
