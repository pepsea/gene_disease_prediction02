"""yesno_question_variants.py の結果を図にする（plotly、ホバーで遺伝子名）。

使い方: python scripts/plot_qvariants.py --disease scz
出力: outputs/<疾患>_set100_qvariants_charts.html（と、kaleido があれば同名の .png）
図: (1) 質問ごとの AUC（そのまま / 有名さを除く） (2) 分類ごとの p_yes 分布
    (3) notebook 03 の「答えを引き継ぐ聞き方」との AUC 比較 (4) 有名さ（C1）× H3 の散布図
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))
import yesno_question_variants as qv

ROOT = qv.ROOT
CAT = [("known", "正解", "#2a78d6", "circle"), ("candidate", "可能性", "#eb6834", "square"), ("random", "ダミー", "#1baf7a", "diamond")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SHORT = {"V0": "V0 改善する可能性（nb02）", "A1": "A1 開発中の薬の標的", "A2": "A2 遺伝学", "B1": "B1 機構が正確に一致",
         "B2": "B2 主要な担い手", "B3": "B3 摂動で症状改善", "B4": "B4 既知標的の近く", "H1": "H1 広い治療仮説",
         "H2": "H2 信号の受け手でも可", "H3": "H3 仮説を立てられるか（追加）", "N1": "N1 否定形（無関係か）",
         "P1": "P1 自信を持って提案できるか", "P2": "P2 機構を異常で説明できるか", "P3": "P3 KO/KI で改善を説明できるか", "P4": "P4 阻害/活性化で回復できるか",
         "C1": "C1 有名さ（対照）"}


def tables(key):
    reg = json.load(open(os.path.join(ROOT, "data", "diseases.json"), encoding="utf-8"))
    D = reg[key]
    long = pd.read_csv(os.path.join(ROOT, "outputs", f"{D['gene_prefix']}_set100_qvariants.csv"))
    w = qv.summarize(long)
    w["p"] = 1 / (1 + np.exp(-w["lo"]))                                  # 両順平均の p_yes（表示用。向きは反転しない）
    wide = w.pivot_table(index=["symbol", "category"], columns="qid", values="s").reset_index()
    k = wide["category"] == "known"
    rc = wide["C1"].rank()
    auc_raw, auc_adj = {}, {}
    for q in qv.QIDS:
        auc_raw[q] = qv.auc(wide.loc[k, q], wide.loc[~k, q])
        rq = wide[q].rank(); r = rq - np.polyval(np.polyfit(rc, rq, 1), rc)   # 有名さ（C1 の順位）で説明できる分を引く
        auc_adj[q] = qv.auc(r[k], r[~k])
    six = os.path.join(ROOT, "outputs", f"{D['gene_prefix']}_set100_six.csv")
    chained = None
    if os.path.exists(six):
        old = pd.read_csv(six); ko = old["category"] == "known"
        chained = {q: qv.auc(old.loc[ko, f"p_{q}"], old.loc[~ko, f"p_{q}"]) for q in qv.QIDS if f"p_{q}" in old.columns}
    names = pd.read_csv(os.path.join(ROOT, "data", "genes", f"{D['gene_prefix']}_set100.tsv"), sep="\t", dtype=str).fillna("")
    names = dict(zip(names["symbol"], names["protein_name_uniprot"].where(names["protein_name_uniprot"] != "", names["gene_name"])))
    return D["name"], w, auc_raw, auc_adj, chained, names


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--disease", default="scz"); a = ap.parse_args()
    disease, w, auc_raw, auc_adj, chained, names = tables(a.disease)
    order = sorted([q for q in qv.QIDS if q != "C1"], key=lambda q: auc_adj[q]) + ["C1"]   # 下から上へ：有名さを除いた AUC の昇順、対照は最上段
    fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.2, vertical_spacing=0.16,
                        subplot_titles=("① 正解を上に置けるか（AUC：正解 vs それ以外）", "② 分類ごとの Yes 確率（両順平均）",
                                        "③ 答えを引き継ぐ聞き方（nb03）→ 独立に聞く", "④ 有名さ（C1）と H3 の関係"))
    # ① AUC：そのまま / 有名さを除く（同じ 0〜1 の尺度なので1軸）
    fig.add_bar(y=[SHORT[q] for q in order], x=[auc_raw[q] for q in order], orientation="h", name="AUC（そのまま）",
                marker=dict(color="#2a78d6", cornerradius=4), legendgroup="auc", legend="legend",
                hovertemplate="%{y}<br>AUC = %{x:.3f}<extra>そのまま</extra>", row=1, col=1)
    fig.add_bar(y=[SHORT[q] for q in order], x=[auc_adj[q] for q in order], orientation="h", name="AUC（有名さを除く）",
                marker=dict(color="#eb6834", cornerradius=4), legendgroup="auc", legend="legend",
                hovertemplate="%{y}<br>AUC = %{x:.3f}<extra>有名さを除く</extra>", row=1, col=1)
    fig.add_vline(x=0.5, line=dict(color=MUTED, width=1, dash="dot"), row=1, col=1)
    # ② 分類ごとの p_yes（質問ごとに3分類を横にずらして点を打つ）
    qs = [q for q in qv.QIDS]
    rng = np.random.default_rng(0)
    for j, (cat, jp, color, sym) in enumerate(CAT):
        d = w[w["category"] == cat].copy()
        d["x"] = d["qid"].map({q: i for i, q in enumerate(qs)}) + (j - 1) * 0.26 + rng.uniform(-0.07, 0.07, len(d))
        fig.add_scatter(x=d["x"], y=d["p"], mode="markers", name=f"{jp}（{cat}）", legendgroup=cat, legend="legend2",
                        marker=dict(color=color, symbol=sym, size=6, opacity=0.7, line=dict(color="#fcfcfb", width=1)),
                        customdata=np.stack([d["symbol"], d["symbol"].map(names).fillna(""), d["qid"].map(SHORT)], axis=1),
                        hovertemplate="<b>%{customdata[0]}</b> %{customdata[1]}<br>%{customdata[2]}<br>p_yes = %{y:.3f}<extra>" + jp + "</extra>",
                        row=1, col=2)
    fig.update_xaxes(tickvals=list(range(len(qs))), ticktext=qs, row=1, col=2)
    # ③ 引き継ぐ vs 独立（ダンベル）
    if chained:
        cq = [q for q in qv.QIDS if q in chained]
        for i, q in enumerate(cq):
            fig.add_scatter(x=[chained[q], auc_raw[q]], y=[SHORT[q]] * 2, mode="lines", line=dict(color=GRID, width=3),
                            showlegend=False, hoverinfo="skip", row=2, col=1)
        fig.add_scatter(x=[chained[q] for q in cq], y=[SHORT[q] for q in cq], mode="markers", name="引き継ぐ（nb03）",
                        marker=dict(color="#52514e", symbol="circle-open", size=11, line=dict(width=2)), legend="legend3",
                        hovertemplate="%{y}<br>引き継ぐ: AUC = %{x:.3f}<extra></extra>", row=2, col=1)
        fig.add_scatter(x=[auc_raw[q] for q in cq], y=[SHORT[q] for q in cq], mode="markers", name="独立に聞く（今回）",
                        marker=dict(color="#2a78d6", symbol="circle", size=11, line=dict(color="#fcfcfb", width=2)), legend="legend3",
                        hovertemplate="%{y}<br>独立: AUC = %{x:.3f}<extra></extra>", row=2, col=1)
        fig.add_vline(x=0.5, line=dict(color=MUTED, width=1, dash="dot"), row=2, col=1)
    else:
        xd, yd = fig.get_subplot(2, 1).xaxis.domain, fig.get_subplot(2, 1).yaxis.domain   # ③ の枠の中央（紙面座標）
        fig.add_annotation(x=sum(xd) / 2, y=sum(yd) / 2, xref="paper", yref="paper", showarrow=False, font=dict(size=13, color=MUTED),
                           text="この疾患には notebook 03（答えを引き継ぐ聞き方）の結果<br>outputs/*_six.csv が無いため、比較できません")
        fig.update_xaxes(visible=False, row=2, col=1); fig.update_yaxes(visible=False, row=2, col=1)
    # ④ 有名さ × H3（対数オッズ）
    pv = w.pivot_table(index=["symbol", "category"], columns="qid", values="lo").reset_index()
    for cat, jp, color, sym in CAT:
        d = pv[pv["category"] == cat]
        fig.add_scatter(x=d["C1"], y=d["H3"], mode="markers", name=jp, legendgroup=cat, showlegend=False,
                        marker=dict(color=color, symbol=sym, size=9, opacity=0.8, line=dict(color="#fcfcfb", width=1)),
                        customdata=np.stack([d["symbol"], d["symbol"].map(names).fillna("")], axis=1),
                        hovertemplate="<b>%{customdata[0]}</b> %{customdata[1]}<br>有名さ C1 = %{x:.1f}<br>H3 = %{y:.1f}<extra>" + jp + "</extra>",
                        row=2, col=2)
    for n, (_, r) in enumerate(pv[pv["category"] == "random"].nlargest(2, "H3").iterrows()):   # H3 が高いダミーだけ名前を出す（選択的なラベル）
        fig.add_annotation(x=r["C1"], y=r["H3"], text=r["symbol"], showarrow=False, xshift=-6, yshift=8 - 16 * n, xanchor="right",
                           font=dict(size=10, color=INK), row=2, col=2)
    fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dot"), row=2, col=2)

    ax = dict(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=MUTED, size=11), title_font=dict(color=MUTED, size=12))
    fig.update_xaxes(**ax); fig.update_yaxes(**ax)
    fig.update_xaxes(title_text="AUC（点線 = でたらめの 0.5）", range=[0, 1.02], row=1, col=1)
    fig.update_yaxes(title_text="p_yes", range=[-0.03, 1.03], row=1, col=2)
    fig.update_xaxes(title_text="質問", row=1, col=2)
    fig.update_xaxes(title_text="AUC（正解 vs それ以外、点線 = 0.5）", range=[0.3, 1.02], row=2, col=1)
    fig.update_xaxes(title_text="C1「よく研究された遺伝子か」の対数オッズ", row=2, col=2)
    fig.update_yaxes(title_text="H3 の対数オッズ", row=2, col=2)
    fig.update_layout(
        title=dict(text=f"{disease}：Yes/No {len(qv.QIDS)}問の比較（txgemma-9b、各問を独立に聞き、両順で平均）", font=dict(size=17, color=INK), x=0.01, y=0.985),
        template="none", paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb", barmode="group", bargap=0.3, bargroupgap=0.1,
        font=dict(family="Hiragino Sans, Noto Sans CJK JP, sans-serif", color=INK), width=1400, height=1050, hovermode="closest",
        margin=dict(l=190, r=30, t=150, b=60),
        legend=dict(x=0.0, y=1.095, orientation="h", font=dict(size=11)),
        legend2=dict(x=0.6, y=1.095, orientation="h", font=dict(size=11)),
        legend3=dict(x=0.0, y=0.49, orientation="h", font=dict(size=11)))
    for an in fig.layout.annotations[:4]: an.font = dict(size=13, color=INK)
    out = os.path.join(ROOT, "outputs", f"{os.path.basename(a.disease)}_set100_qvariants_charts.html")
    fig.write_html(out, include_plotlyjs=True); print("wrote", out)
    try:
        fig.write_image(out.replace(".html", ".png"), scale=1); print("wrote", out.replace(".html", ".png"))
    except Exception as e:
        print("png skipped:", e)


if __name__ == "__main__":
    main()
