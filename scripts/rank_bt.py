"""ノートブック 04（5遺伝子＋該当なしから1つ選ぶ）の勝敗データから、Bradley-Terry（5択への拡張＝Luce の選択モデル）と Elo で強さを推定する。
モデルに新しく質問はしない。outputs/<疾患>_set100_rankvar.csv（rank_variants.py の出力）を使う。

使い方: python scripts/rank_bt.py [--variant M2r]
出力: outputs/rank_bt_charts.html（plotly.js 埋め込み）と outputs/<疾患>_set100_rank_bt.csv
図: (1) AUC の比較（mean_p / Elo / Luce-BT）  (2) 疾患ごとの散布図 mean_p × Luce-BT（点線＝「該当なし」の強さ）
    (3) 既知遺伝子の順位
"""
import argparse, itertools, os, sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))
from rescue_report import ROOT, auc
from rank_variants import gene_scores
from tournament import elo, spearman

DISEASES = [("achondroplasia", "軟骨無形成症"), ("ra", "関節リウマチ"), ("prostate_cancer", "前立腺がん"), ("scz", "統合失調症"), ("cystinuria", "シスチン尿症")]
METHODS = [("mean_p", "mean_p（04 の現行：選ばれた確率の平均）", "#c9c8c2"), ("elo", "Elo（5択を10組の対戦に分解）", "#8a5cd1"),
           ("luce", "Bradley-Terry（Luce の選択モデル）", "#2a78d6")]
CAT = [("known", "既知", "#2a78d6", "circle"), ("candidate", "候補", "#eb6834", "square"), ("random", "ダミー", "#1baf7a", "diamond")]
LAYOUT = dict(template="plotly_white", font=dict(family="Hiragino Sans, Noto Sans JP, sans-serif", size=12))


def luce(groups, n, iters=1000, prior=0.5):
    """Luce の選択モデル（Bradley-Terry を多者択一に広げたもの）の強さを MM 法（Hunter 2004）で推定する。
    groups: [(メンバーの番号の並び, 各メンバーが選ばれた確率)]。遺伝子 i が選ばれる確率 = π_i / Σ_{グループ内} π。
    各項目に「強さ1の仮想相手との prior 勝ち・prior 負け」を足して、全勝・全敗でも発散しないようにする。return: log π"""
    W, pi = np.full(n, prior), np.ones(n)
    for mem, p in groups:
        for m, q in zip(mem, p): W[m] += q
    for _ in range(iters):
        den = 2 * prior / (pi + 1)
        for mem, p in groups:
            den[mem] += 1 / pi[mem].sum()
        new = W / den; new /= np.exp(np.mean(np.log(new)))
        if np.max(np.abs(np.log(new) - np.log(pi))) < 1e-10: pi = new; break
        pi = new
    return np.log(pi)


def fit(key, variant):
    L = pd.read_csv(os.path.join(ROOT, "outputs", f"{key}_set100_rankvar.csv"))
    L = L[L["variant"] == variant]
    syms = sorted(L["symbol"].unique()); items = syms + ["NONE"]; ix = {s: i for i, s in enumerate(items)}
    groups, pairs = [], []
    for _, d in L.groupby(["round", "group"]):
        d = d.sort_values("position")
        mem = [ix[s] for s in d["symbol"]] + [ix["NONE"]]; p = np.r_[d["p"].values, d["p_none"].iloc[0]]
        groups.append((mem, p))
        for a, b in itertools.combinations(range(len(mem)), 2):      # Elo 用：a と b だけを比べたときの勝つ確率 p_a / (p_a + p_b)
            if p[a] + p[b] > 0: pairs.append((mem[a], mem[b], p[a] / (p[a] + p[b])))
    s_luce, s_elo = luce(groups, len(items)), elo(len(items), pairs)
    g = gene_scores(L).set_index("symbol").reindex(syms)
    out = pd.DataFrame({"symbol": syms, "category": g["category"].values, "mean_p": g["mean_p"].values,
                        "luce": s_luce[:-1], "elo": s_elo[:-1]})
    out["luce_above_none"] = out["luce"] > s_luce[-1]
    for m, *_ in METHODS: out[f"{m}_rank"] = out[m].rank(ascending=False, method="min").astype(int)
    out.to_csv(os.path.join(ROOT, "outputs", f"{key}_set100_rank_bt.csv"), index=False)
    return out, float(s_luce[-1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="M2r"); a = ap.parse_args()
    res = {k: fit(k, a.variant) for k, _ in DISEASES}
    rows = []
    for k, name in DISEASES:
        t = res[k][0]; kn, rd, cd = t.category.eq("known"), t.category.eq("random"), t.category.eq("candidate")
        for m, lab, _ in METHODS:
            rows.append({"疾患": name, "方式": m, "既知 vs その他": auc(t.loc[kn, m], t.loc[~kn, m]),
                         "既知 vs ダミー": auc(t.loc[kn, m], t.loc[rd, m]), "候補 vs ダミー": auc(t.loc[cd, m], t.loc[rd, m])})
    T = pd.DataFrame(rows)
    print(T.groupby("方式")[["既知 vs その他", "既知 vs ダミー", "候補 vs ダミー"]].mean().round(3))

    # (1) AUC
    cols = ("既知 vs その他", "既知 vs ダミー", "候補 vs ダミー")
    f1 = make_subplots(rows=1, cols=3, subplot_titles=[f"AUC {c}" for c in cols], shared_yaxes=True, horizontal_spacing=0.03)
    names = [n for _, n in DISEASES] + ["平均"]
    for j, c in enumerate(cols, start=1):
        for m, lab, color in METHODS:
            s = T[T["方式"] == m].set_index("疾患")[c]; y = list(s.reindex(names[:-1])) + [s.mean()]
            f1.add_trace(go.Bar(x=names, y=y, name=lab, marker_color=color, legendgroup=m, showlegend=(j == 1),
                                text=[f"{v:.3f}" for v in y], textposition="outside", textfont=dict(size=9),
                                hovertemplate=f"<b>{lab}</b><br>%{{x}}<br>{c} %{{y:.3f}}<extra></extra>"), row=1, col=j)
    f1.update_yaxes(range=[0.5, 1.05])
    f1.update_layout(**LAYOUT, barmode="group", height=480, title=f"ノートブック 04（{a.variant}）の勝敗データから推定した強さの AUC（モデルへの追加の質問なし）")

    # (2) 散布図 mean_p × Luce-BT
    f2 = make_subplots(rows=1, cols=5, subplot_titles=[n for _, n in DISEASES], horizontal_spacing=0.035)
    for j, (k, name) in enumerate(DISEASES, start=1):
        t, none = res[k]
        f2.layout.annotations[j - 1].text = f"{name}（ρ={spearman(t['mean_p'], t['luce']):.2f}）"
        for cat, jp, col, mk in CAT:
            s = t[t.category.eq(cat)]
            f2.add_trace(go.Scatter(x=s["mean_p"], y=s["luce"], mode="markers", name=jp, legendgroup=cat, showlegend=(j == 1),
                                    marker=dict(color=col, symbol=mk, size=9 if cat == "known" else 6, opacity=0.9 if cat == "known" else 0.55,
                                                line=dict(color="white", width=0.5)),
                                    customdata=np.c_[s["symbol"], [jp] * len(s), s["luce_rank"], s["mean_p_rank"]],
                                    hovertemplate="<b>%{customdata[0]}</b>（%{customdata[1]}）<br>mean_p %{x:.3f}（%{customdata[3]}位）"
                                                  "<br>BT 強さ %{y:.2f}（%{customdata[2]}位）<extra></extra>"), row=1, col=j)
        f2.add_hline(y=none, line=dict(color="#0b0b0b", dash="dot"), row=1, col=j)
    f2.update_xaxes(title_text="mean_p", row=1, col=3); f2.update_yaxes(title_text="Bradley-Terry の強さ log π", row=1, col=1)
    f2.update_layout(**LAYOUT, height=440, title="遺伝子ごとの点数：横＝mean_p、縦＝Bradley-Terry（点線＝「該当なし」の強さ。点線より上＝該当なしより選ばれやすい）")

    # (3) 既知遺伝子の順位
    zr, ylab = [], []
    for k, name in DISEASES:
        t = res[k][0]; kk = t[t.category.eq("known")].sort_values("mean_p_rank", ascending=False)
        for _, r in kk.iterrows():
            zr.append([r[f"{m}_rank"] for m, *_ in METHODS]); ylab.append(f"{name}｜{r['symbol']}")
    z = np.array(zr, float)
    f3 = go.Figure(go.Heatmap(z=z, x=[lab for _, lab, _ in METHODS], y=ylab, zmin=1, zmax=100, colorscale="RdYlGn_r",
                              text=z.astype(int).astype(str), texttemplate="%{text}", colorbar=dict(title="順位"),
                              hovertemplate="%{y}<br>%{x}<br>順位 %{z:.0f}<extra></extra>"))
    f3.update_layout(**LAYOUT, height=24 * len(ylab) + 220, margin=dict(l=190, t=160), yaxis=dict(autorange="reversed"),
                     xaxis=dict(side="top"), title="既知遺伝子の順位（100中、緑＝上位）", title_y=0.99)

    out = os.path.join(ROOT, "outputs", "rank_bt_charts.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("<html><head><meta charset='utf-8'><title>04 の Bradley-Terry</title></head><body style='max-width:1300px;margin:auto'>"
                 + "".join(f.to_html(full_html=False, include_plotlyjs=(True if i == 0 else False)) for i, f in enumerate((f1, f2, f3))) + "</body></html>")
    print("saved:", out)
    return (f1, f2, f3)


if __name__ == "__main__":
    main()
