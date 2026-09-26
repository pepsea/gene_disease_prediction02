"""tournament.py の結果を疾患ごとに図にする（plotly、ホバーで遺伝子名。plotly.js は HTML に埋め込む）。

使い方: python scripts/plot_tournament.py --disease ra
出力: outputs/tournament_<疾患>.html
図: (1) AUC などの比較（M3s の並び vs 総当たり）  (2) 強さの棒グラフ  (3) 順位の入れ替わり  (4) 対戦の勝率ヒートマップ
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))
from rescue_report import ROOT, auc

NAMES = {"achondroplasia": "軟骨無形成症", "ra": "関節リウマチ", "prostate_cancer": "前立腺がん", "scz": "統合失調症", "cystinuria": "シスチン尿症"}
COL = {"known": "#2a78d6", "candidate": "#eb6834", "random": "#1baf7a"}
LAYOUT = dict(template="plotly_white", font=dict(family="Hiragino Sans, Noto Sans JP, sans-serif", size=12))


def summary(t):
    kn = t.category.eq("known")
    rows = []
    for name, col, rk in (("M3s の並び", "m3s_logodds", "m3s_rank"), ("総当たり（Bradley-Terry）", "bt", "bt_rank"), ("総当たり（Elo）", "elo", None)):
        r = t[col].rank(ascending=False, method="min") if rk is None else t[rk]
        rows.append({"方式": name, "AUC 既知 vs その他（上位内）": auc(t.loc[kn, col], t.loc[~kn, col]),
                     "既知の平均順位": r[kn].mean(), "上位10件中の既知": int((r <= 10)[kn].sum())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--disease", required=True); a = ap.parse_args()
    key = a.disease; name = NAMES.get(key, key)
    t = pd.read_csv(os.path.join(ROOT, "outputs", f"{key}_set100_tournament.csv")).sort_values("bt_rank").reset_index(drop=True)
    pr = pd.read_csv(os.path.join(ROOT, "outputs", f"{key}_set100_tournament_pairs.csv"))
    s = summary(t); n = len(t)
    first_pos = float(np.mean(list(pr["p_a_first"]) + list(1 - pr["p_a_second"])))
    print(s.round(3).to_string(index=False)); print(f"1番が選ばれる確率の平均 {first_pos:.3f}")

    f1 = make_subplots(rows=1, cols=3, subplot_titles=list(s.columns[1:]), horizontal_spacing=0.08)
    for j, col in enumerate(s.columns[1:], start=1):
        f1.add_trace(go.Bar(x=s["方式"], y=s[col], marker_color=["#c9c8c2", "#2a78d6", "#8a5cd1"], showlegend=False,
                            text=[f"{v:.3f}" if j == 1 else f"{v:.1f}" if j == 2 else f"{v:.0f}" for v in s[col]], textposition="outside",
                            hovertemplate="%{x}<br>" + col + " %{y:.3f}<extra></extra>"), row=1, col=j)
    f1.update_yaxes(range=[0, 1.1], row=1, col=1)
    f1.update_layout(**LAYOUT, height=400, title=f"{name}：上位{n}（M3s で絞り込み）の中で既知遺伝子をどれだけ上に置けたか　｜　1番が選ばれる確率 {first_pos:.2f}（両順の平均で打ち消し済み）")

    f2 = go.Figure(go.Bar(x=t["symbol"], y=t["bt"], marker_color=[COL.get(c, "#888") for c in t["category"]],
                          customdata=t[["category", "m3s_rank", "win_rate", "elo"]].values,
                          hovertemplate="<b>%{x}</b>（%{customdata[0]}）<br>強さ log π %{y:.2f}<br>M3s の順位 %{customdata[1]}"
                                        "<br>勝率 %{customdata[2]:.2f}<br>Elo %{customdata[3]:.0f}<extra></extra>"))
    f2.update_layout(**LAYOUT, height=420, xaxis_tickangle=-60, yaxis_title="強さ log π",
                     title="Bradley-Terry の強さ（強い順）。青＝既知、橙＝候補、緑＝ダミー")

    f3 = go.Figure()
    for _, r in t.iterrows():
        f3.add_trace(go.Scatter(x=[0, 1], y=[r["m3s_rank"], r["bt_rank"]], mode="lines+markers+text", text=[r["symbol"], r["symbol"]],
                                textposition=["middle left", "middle right"], showlegend=False, marker=dict(size=7),
                                line=dict(color=COL.get(r["category"], "#888"), width=3 if r["category"] == "known" else 1),
                                hovertemplate=f"<b>{r['symbol']}</b>（{r['category']}）<br>順位 %{{y}}<extra></extra>"))
    f3.update_layout(**LAYOUT, width=700, height=900, title="順位の入れ替わり：左＝M3s、右＝総当たり（太線＝既知）",
                     xaxis=dict(tickvals=[0, 1], ticktext=["M3s", "総当たり"], range=[-0.45, 1.45]), yaxis=dict(autorange="reversed", title="順位"))

    order = t["symbol"].tolist(); pos = {g: i for i, g in enumerate(order)}
    M = np.full((n, n), np.nan)
    for r in pr.itertuples():
        w = (r.p_a_first + r.p_a_second) / 2; M[pos[r.a], pos[r.b]] = w; M[pos[r.b], pos[r.a]] = 1 - w
    lab = [f"{g}{'★' if c == 'known' else ''}" for g, c in zip(order, t["category"])]
    f4 = go.Figure(go.Heatmap(z=M, x=lab, y=lab, zmin=0, zmax=1, colorscale="RdBu", colorbar=dict(title="行が勝つ確率"),
                              hovertemplate="%{y} が %{x} に勝つ確率 %{z:.2f}<extra></extra>"))
    f4.update_layout(**LAYOUT, width=850, height=800, yaxis=dict(autorange="reversed"), xaxis_tickangle=-60,
                     title="対戦の勝率（両順の平均、強い順に並べた。★＝既知）")

    out = os.path.join(ROOT, "outputs", f"tournament_{key}.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("<html><head><meta charset='utf-8'><title>総当たり</title></head><body style='max-width:1200px;margin:auto'>"
                 + "".join(f.to_html(full_html=False, include_plotlyjs=(True if i == 0 else False)) for i, f in enumerate((f1, f2, f3, f4))) + "</body></html>")
    print("saved:", out)


if __name__ == "__main__":
    main()
