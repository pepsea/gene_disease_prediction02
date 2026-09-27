"""段階1の比較図：埋め込み（bge-m3）とリランカーで、正解の遺伝子が全遺伝子中の何位に来るか。
使い方: python scripts/plot_rerank.py --disease ra --run outputs/genome/ra/20260926-1415
入力: <run>/stage1.csv（埋め込みの順位）、<run>/stage6.csv（Open Targets の臨床段階）、outputs/rerank/<疾患>_rerank.csv
出力: outputs/rerank/<疾患>_rerank_charts.html
図: (1) 既知の標的の順位（遺伝子ごと、横軸は対数）
    (2) 上位K件に正解が何割入るか（既知、Open Targets の第2相以上）
"""
import argparse, os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LAYOUT = dict(template="plotly_white", font=dict(family="Hiragino Sans, Noto Sans JP, sans-serif", size=12))
NAMES = {"rank1": "埋め込み bge-m3（今の段階1）", "rank_bge_plain": "bge-reranker・疾患の説明そのまま",
         "rank_bge_target": "bge-reranker・治療標的を探す言い方", "rank_qwen06": "Qwen3-Reranker-0.6B（指示文付き）", "rank_qwen06_m3s": "Qwen3-Reranker-0.6B（M3s の3条件）",
         "rank_qwen4": "Qwen3-Reranker-4B（指示文付き）"}
COLS = {"rank1": "#8a8984", "rank_bge_plain": "#9ec0ec", "rank_bge_target": "#2a78d6", "rank_qwen06": "#d64545", "rank_qwen06_m3s": "#e8912d", "rank_qwen4": "#7a2e8c"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--disease", default="ra"); ap.add_argument("--run", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "pipeline07.yaml"), encoding="utf-8"))
    D = {d["key"]: d for d in yaml.safe_load(open(os.path.join(ROOT, cfg["diseases_file"]), encoding="utf-8"))["diseases"]}[a.disease]
    t = pd.read_csv(os.path.join(ROOT, "outputs", "rerank", f"{a.disease}_rerank.csv")).merge(
        pd.read_csv(os.path.join(ROOT, a.run, "stage1.csv"))[["symbol", "rank1"]], on="symbol")
    methods = [m for m in NAMES if m in t.columns]
    known = set(pd.read_csv(os.path.join(ROOT, D["known_file"]), sep="\t")["symbol"])
    s6 = pd.read_csv(os.path.join(ROOT, a.run, "stage6.csv"))
    ot = set(s6.loc[s6.ot_max_stage.isin(["PHASE_2", "PHASE_2_3", "PHASE_3", "APPROVAL"]), "symbol"])
    N, keep = len(t), cfg["stage1"]["keep"]

    k = t[t.symbol.isin(known)].sort_values("rank1")
    print(k[["symbol"] + methods].to_string(index=False))
    Ks = [500, 1000, 2000, 5000]
    rows = [{"方法": NAMES[m], **{f"既知@{K}": int((k[m] <= K).sum()) for K in Ks},
             **{f"OT第2相以上@{K}": round(float((t[t.symbol.isin(ot)][m] <= K).mean()), 3) for K in Ks}} for m in methods]
    T = pd.DataFrame(rows); pd.set_option("display.width", 250); print(T.to_string(index=False))

    f1 = go.Figure()
    for m in methods:
        f1.add_trace(go.Scatter(x=k[m], y=k["symbol"], mode="markers", name=NAMES[m], marker=dict(size=11, color=COLS[m], line=dict(width=1, color="white")),
                                hovertemplate="%{y}<br>" + NAMES[m] + "<br>%{x} 位<extra></extra>"))
    for K, lab in ((keep, f"今の線 {keep}位"), (2000, "2000位")):
        f1.add_vline(x=K, line=dict(color="#c3c2b7", dash="dot"), annotation_text=lab, annotation_position="top")
    f1.update_xaxes(type="log", title=f"全 {N} 遺伝子中の順位（対数、左ほど上位）", range=[0, np.log10(N) + 0.05])
    f1.update_yaxes(autorange="reversed", title="")
    f1.update_layout(**LAYOUT, height=40 * len(k) + 200, legend=dict(orientation="h", y=-0.15),
                     title=f"{D['name']}：既知の標的 {len(k)} 個の順位（上から、今の段階1で上位の順）")

    f2 = make_subplots(rows=1, cols=2, subplot_titles=[f"既知の標的（{len(k)} 個）", f"Open Targets 第2相以上の標的（{len(ot & set(t.symbol))} 個）"])
    Kx = np.unique(np.logspace(1, np.log10(N), 80).astype(int))
    for j, truth in enumerate((known, ot), start=1):
        r = t[t.symbol.isin(truth)]
        for m in methods:
            f2.add_trace(go.Scatter(x=Kx, y=[(r[m] <= K).mean() for K in Kx], mode="lines", name=NAMES[m], legendgroup=m, showlegend=(j == 1),
                                    line=dict(color=COLS[m], width=2.5), hovertemplate="上位 %{x} 件に %{y:.0%}<extra>" + NAMES[m] + "</extra>"), row=1, col=j)
        f2.add_trace(go.Scatter(x=Kx, y=Kx / N, mode="lines", name="ランダム", showlegend=(j == 1), line=dict(color="#c3c2b7", dash="dot")), row=1, col=j)
        f2.add_vline(x=keep, line=dict(color="#c3c2b7", dash="dot"), row=1, col=j)
    f2.update_xaxes(type="log", title="上位K件（対数）"); f2.update_yaxes(title="正解のうち入った割合", range=[0, 1.02], tickformat=".0%")
    f2.update_layout(**LAYOUT, height=480, legend=dict(orientation="h", y=-0.25), title="上位K件に正解が何割入るか（縦の点線＝今の段階1の線）")

    out = os.path.join(ROOT, "outputs", "rerank", f"{a.disease}_rerank_charts.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("<html><head><meta charset='utf-8'><title>段階1の比較</title></head><body style='max-width:1200px;margin:auto'>"
                 + f1.to_html(full_html=False, include_plotlyjs=True) + f2.to_html(full_html=False, include_plotlyjs=False)
                 + "<h3>表：上位K件に入った数・割合</h3>" + T.to_html(index=False) + "</body></html>")
    print("saved:", out)


if __name__ == "__main__":
    main()
