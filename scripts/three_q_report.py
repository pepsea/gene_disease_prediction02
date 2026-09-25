"""3問版（H3 + R1 + B1）と改訂版（H3 + R1b + B1b）を比べる：表（標準出力）と plotly の図。

使い方: python scripts/three_q_report.py
入力: outputs/<疾患>_set100_qvariants.csv（yesno_question_variants.py の出力）
出力: outputs/three_q_charts.html
"""
import os, re, sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))
from rescue_report import ROOT, auc, wide_scores

DISEASES = [("achondroplasia", "軟骨無形成症"), ("ra", "関節リウマチ"), ("prostate_cancer", "前立腺がん"),
            ("scz", "統合失調症"), ("cystinuria", "シスチン尿症")]
SETS = {"旧 H3+R1+B1": ["H3", "R1", "B1"], "新 H3+R1b+B1b": ["H3", "R1b", "B1b"], "条件参照 H3c+R1c+B1": ["H3c", "R1c", "B1"]}
QS = ["H3", "H3c", "R1", "R1b", "R1c", "B1", "B1b"]
LABEL = {"H3": "H3 治療仮説", "R1": "R1 代償（旧）", "R1b": "R1b 代償（新）", "B1": "B1 機構一致（旧）", "B1b": "B1b 機構一致（新）",
         "旧 H3+R1+B1": "旧 3問合成", "新 H3+R1b+B1b": "新 3問合成", "H3c": "H3c 治療仮説（条件参照）", "R1c": "R1c 代償（条件参照）", "条件参照 H3c+R1c+B1": "条件参照 3問合成"}
CAT = [("known", "既知", "#2a78d6", "circle"), ("candidate", "候補", "#eb6834", "square"), ("random", "ダミー", "#1baf7a", "diamond"),
       ("random_same", "ダミー（正解と同じファミリー）", "#eda100", "triangle-up")]
LAYOUT = dict(template="plotly_white", font=dict(family="Hiragino Sans, Noto Sans JP, sans-serif", size=12))
fam = lambda s: (re.match(r"[A-Z]+\d*", s) or re.match(r".*", s)).group(0)    # SLC7A9 -> SLC7（簡易）


def load():
    data = {}
    for key, _ in DISEASES:
        w, _ = wide_scores(key)
        for name, cols in SETS.items():
            if all(c in w.columns for c in cols): w[name] = w[cols].mean(axis=1)
        kfam = {fam(s) for s in w.loc[w.category.eq("known"), "symbol"]}
        w["cat2"] = np.where(w.category.eq("random") & w.symbol.map(fam).isin(kfam), "random_same", w.category)
        data[key] = w
    return data


def metrics(data):
    rows, ranks = [], []
    for key, name in DISEASES:
        w = data[key]; k, r = w.category.eq("known"), w.category.eq("random"); c = w.category.eq("candidate")
        rs = w.cat2.eq("random_same")
        for q in [q for q in QS + list(SETS) if q in w.columns]:
            rows.append({"疾患": name, "質問": q, "AUC 既知/ダミー": auc(w.loc[k, q], w.loc[r, q]), "AUC 既知/その他": auc(w.loc[k, q], w.loc[~k, q]),
                         "AUC 候補/ダミー": auc(w.loc[c, q], w.loc[r, q]),
                         "AUC 既知/同族ダミー": auc(w.loc[k, q], w.loc[rs, q]) if rs.any() else np.nan,
                         "ダミーのYes率": 1 / (1 + np.exp(-w.loc[r, q].median()))})
            rk = w[q].rank(ascending=False, method="min")
            for i in w.index[k]: ranks.append({"疾患": name, "遺伝子": w.at[i, "symbol"], "質問": q, "順位": int(rk[i])})
    return pd.DataFrame(rows), pd.DataFrame(ranks)


def fig_rank(rk):
    cols = [q for q in QS + list(SETS) if q in rk["質問"].unique()]
    p = rk.pivot_table(index=["疾患", "遺伝子"], columns="質問", values="順位")[cols]
    order = [n for _, n in DISEASES]
    p = p.reset_index(); p["o"] = p["疾患"].map(order.index); p = p.sort_values(["o", cols[-1]], ascending=[True, False])
    y = [f"{a}｜{b}" for a, b in zip(p["疾患"], p["遺伝子"])]
    z = p[cols].to_numpy(float)
    fig = go.Figure(go.Heatmap(z=z, x=[LABEL[c] for c in cols], y=y, text=np.where(np.isnan(z), "", np.nan_to_num(z).astype(int).astype(str)), texttemplate="%{text}",
                               colorscale="RdYlGn_r", zmin=1, zmax=100, colorbar=dict(title="順位"),
                               hovertemplate="%{y}<br>%{x}<br>順位 %{z:.0f} / 100<extra></extra>"))
    fig.update_layout(**LAYOUT, title="既知遺伝子の順位（100中、小さいほど良い）：旧 → 新", height=26 * len(y) + 260,
                      yaxis=dict(autorange="reversed"), xaxis=dict(side="top", tickangle=-30), margin=dict(l=200, t=200), title_y=0.99)
    for x0 in (1.5, 4.5, 6.5): fig.add_vline(x=x0, line=dict(color="#0b0b0b", width=1.5))
    return fig


def fig_auc(m):
    fig = go.Figure()
    col = {"H3": "#8a8984", "R1": "#b9d3f2", "R1b": "#2a78d6", "B1": "#f6c3ad", "B1b": "#eb6834", "旧 H3+R1+B1": "#a8ddc6", "新 H3+R1b+B1b": "#1baf7a", "H3c": "#0b0b0b", "R1c": "#123f7a", "条件参照 H3c+R1c+B1": "#8a5cd1"}
    for q in [q for q in QS + list(SETS) if q in m["質問"].unique()]:
        s = m[m["質問"] == q]
        fig.add_trace(go.Bar(x=s["疾患"], y=s["AUC 既知/その他"], name=LABEL[q], marker_color=col[q],
                             customdata=np.c_[s["AUC 既知/ダミー"], s["ダミーのYes率"], s["AUC 候補/ダミー"]],
                             hovertemplate=f"<b>{LABEL[q]}</b><br>%{{x}}<br>AUC 既知/その他 %{{y:.3f}}<br>AUC 既知/ダミー %{{customdata[0]:.3f}}"
                                           "<br>AUC 候補/ダミー %{customdata[2]:.3f}<br>ダミーのYes率 %{customdata[1]:.2f}<extra></extra>"))
    fig.update_layout(**LAYOUT, barmode="group", title="AUC 既知 vs その他（候補＋ダミー）— 淡色＝旧、濃色＝新", yaxis=dict(range=[0.4, 1.0], title="AUC"), height=480)
    return fig


def fig_yes(m):
    fig = go.Figure()
    for q, dash in [("B1", "dot"), ("B1b", "solid"), ("R1", "dot"), ("R1b", "solid"), ("R1c", "dash"), ("H3", "dot"), ("H3c", "dash")]:
        s = m[m["質問"] == q]
        if s.empty: continue
        fig.add_trace(go.Scatter(x=s["疾患"], y=s["ダミーのYes率"], mode="lines+markers", name=LABEL[q],
                                 line=dict(dash=dash, color={"B": "#eb6834", "R": "#2a78d6", "H": "#0b0b0b"}[q[0]], width=2), marker=dict(size=9),
                                 hovertemplate=f"<b>{LABEL[q]}</b><br>%{{x}}<br>ダミーのYes率 %{{y:.2f}}<extra></extra>"))
    fig.update_layout(**LAYOUT, title="ダミーへの Yes 率（中央値）— 低いほど「何にでも Yes」ではない。点線＝旧、実線＝新",
                      yaxis=dict(range=[0, 1.02], title="ダミーの Yes 率"), height=420)
    return fig


def fig_strip(data, key, name):
    w = data[key]; qs = [q for q in QS + list(SETS) if q in w.columns]
    fig, rng = go.Figure(), np.random.default_rng(0)
    for cat, jp, col, mk in CAT:
        sub = w[w.cat2.eq(cat)]
        if sub.empty: continue
        for j, q in enumerate(qs):
            fig.add_trace(go.Scatter(x=j + rng.uniform(-0.28, 0.28, len(sub)), y=sub[q], mode="markers", name=jp, legendgroup=cat, showlegend=(j == 0),
                                     marker=dict(color=col, symbol=mk, size=10 if cat == "known" else 6, opacity=0.95 if cat in ("known", "random_same") else 0.5,
                                                 line=dict(color="white", width=0.5)),
                                     customdata=np.c_[sub["symbol"], [jp] * len(sub), [LABEL[q]] * len(sub)],
                                     hovertemplate="<b>%{customdata[0]}</b>（%{customdata[1]}）<br>%{customdata[2]}<br>対数オッズ %{y:.2f}<extra></extra>"))
    fig.update_layout(**LAYOUT, title=f"{name}：全遺伝子の採点（両順の対数オッズ平均）",
                      xaxis=dict(tickmode="array", tickvals=list(range(len(qs))), ticktext=[LABEL[q] for q in qs]),
                      yaxis=dict(title="対数オッズ（大きいほど Yes）"), height=480)
    return fig


def main():
    data = load()
    m, rk = metrics(data)
    pd.set_option("display.width", 200)
    print(m.round(3).to_string(index=False))
    print(rk.pivot_table(index=["疾患", "遺伝子"], columns="質問", values="順位")[[q for q in QS + list(SETS) if q in rk["質問"].unique()]].astype("Int64").to_string())
    avg = m.groupby("質問")[["AUC 既知/ダミー", "AUC 既知/その他", "AUC 候補/ダミー", "ダミーのYes率"]].mean().round(3)
    print("5疾患平均:\n" + avg.to_string())
    figs = [fig_auc(m), fig_yes(m), fig_rank(rk)] + [fig_strip(data, k, n) for k, n in DISEASES]
    out = os.path.join(ROOT, "outputs", "three_q_charts.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("<html><head><meta charset='utf-8'><title>3問版の検証</title></head><body style='max-width:1200px;margin:auto'>"
                 + "".join(f.to_html(full_html=False, include_plotlyjs=(True if i == 0 else False)) for i, f in enumerate(figs)) + "</body></html>")
    print("saved:", out)


if __name__ == "__main__":
    main()
