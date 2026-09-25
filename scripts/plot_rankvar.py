"""相対評価（強制選択）の質問パターン比較を、Yes/No の M1・3問平均と並べて図にする（plotly、ホバーで遺伝子名）。

使い方: python scripts/plot_rankvar.py
入力: outputs/<疾患>_set100_rankvar.csv（rank_variants.py）と outputs/<疾患>_set100_qvariants.csv（yesno_question_variants.py）
出力: outputs/rankvar_charts.html
図: (1) AUC ヒートマップ（方式 × 疾患、3種の AUC）  (2) 疾患ごとの AUC 棒グラフ
    (3) 番号（位置）ごとの平均確率＝番号の癖  (4) 既知遺伝子の順位  (5) Yes/No の M1 × 強制選択の散布図
"""
import os, sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))
from rescue_report import ROOT, auc, wide_scores
from rank_variants import VIDS, gene_scores

DISEASES = [("achondroplasia", "軟骨無形成症"), ("ra", "関節リウマチ"), ("prostate_cancer", "前立腺がん"),
            ("scz", "統合失調症"), ("cystinuria", "シスチン尿症")]
LAYOUT = dict(template="plotly_white", font=dict(family="Hiragino Sans, Noto Sans JP, sans-serif", size=12))
YN = {"YN M1": "Yes/No M1（1問）", "YN 3問平均": "Yes/No 3問平均", "評点 0-9": "評点 0〜9（1遺伝子ずつ、M1の3条件、正向きのみ）",
      "外す（最悪側）": "5遺伝子から外す1つ（最悪側）", "最良−最悪": "最良（pick1）− 最悪（外す）"}
CAT = [("known", "既知", "#2a78d6", "circle"), ("candidate", "候補", "#eb6834", "square"), ("random", "ダミー", "#1baf7a", "diamond")]


def load():
    """疾患ごとに、遺伝子 × 方式 のスコア表（強制選択は variant/score ごと、Yes/No は M1 と3問平均）。"""
    data, longs = {}, {}
    for key, _ in DISEASES:
        p = os.path.join(ROOT, "outputs", f"{key}_set100_rankvar.csv")
        if not os.path.exists(p): continue
        long = pd.read_csv(p); longs[key] = long
        g = gene_scores(long)
        w = g.pivot_table(index=["symbol", "category"], columns="variant", values=["mean_p", "win_rate", "lr_none"])
        w.columns = [f"{v} {s}" for s, v in w.columns]
        yn, _ = wide_scores(key)
        yn = yn.set_index(["symbol", "category"])
        w["YN M1"] = yn["M1"]; w["YN 3問平均"] = yn[["H3", "R1", "B1"]].mean(axis=1)
        am = os.path.join(ROOT, "outputs", f"{key}_set100_askmodes.csv")
        if os.path.exists(am):                                    # ask_modes.py の結果（評点・最悪側）
            a = pd.read_csv(am).set_index(["symbol", "category"])
            if "scale_norm" in a: w["評点 0-9"] = a["scale_norm"]      # 正向き（9 = 当てはまる）だけ。逆向きの目盛りはモデルが無視して逆順になるため使わない
            if "worst" in a:
                w["外す（最悪側）"] = a["worst"]
                if "pick1 mean_p" in w: w["最良−最悪"] = w["pick1 mean_p"] + a["worst"]   # worst は符号反転済み（= −選ばれた確率）
        data[key] = w.reset_index()
    return data, longs


def method_cols(score):
    return [f"{v} {score}" for v in VIDS] + list(YN)          # 黒線（len(VIDS) の位置）より後ろが「選択式（最良側）以外」の方式


def auc_table(data, score):
    rows = []
    for key, name in DISEASES:
        if key not in data: continue
        w = data[key]; k, r, c = w.category.eq("known"), w.category.eq("random"), w.category.eq("candidate")
        for m in method_cols(score):
            if m not in w: continue
            rows.append({"疾患": name, "方式": m, "既知 vs その他": auc(w.loc[k, m], w.loc[~k, m]),
                         "既知 vs ダミー": auc(w.loc[k, m], w.loc[r, m]), "候補 vs ダミー": auc(w.loc[c, m], w.loc[r, m])})
    return pd.DataFrame(rows)


def label(m):
    return YN.get(m, m.replace(" lr_none", "（対 該当なし）").replace(" mean_p", "（平均確率）").replace(" win_rate", "（1位率）"))


def fig_heat(data, score):
    t = auc_table(data, score)
    names = [n for k, n in DISEASES if k in data]
    fig = make_subplots(rows=1, cols=3, subplot_titles=("AUC 既知 vs その他", "AUC 既知 vs ダミー", "AUC 候補 vs ダミー"),
                        shared_yaxes=True, horizontal_spacing=0.03)
    for j, col in enumerate(("既知 vs その他", "既知 vs ダミー", "候補 vs ダミー"), start=1):
        p = t.pivot(index="方式", columns="疾患", values=col)[names]
        p["平均"] = p.mean(axis=1)
        p = p.loc[[m for m in method_cols(score) if m in p.index]]
        fig.add_trace(go.Heatmap(z=p.values, x=list(p.columns), y=[label(m) for m in p.index], zmin=0.5, zmax=1.0, colorscale="Blues",
                                 text=np.round(p.values, 3), texttemplate="%{text}", showscale=(j == 3),
                                 hovertemplate="%{y}<br>%{x}<br>" + col + " %{z:.3f}<extra></extra>"), row=1, col=j)
    fig.update_yaxes(autorange="reversed")
    for j in (1, 2, 3): fig.add_hline(y=len(VIDS) - 0.5, line=dict(color="#0b0b0b", width=2), row=1, col=j)
    fig.update_layout(**LAYOUT, height=34 * len(p) + 180, margin=dict(l=280),
                      title=f"AUC の比較（強制選択のスコア＝{score}）。黒線より下は選択式（最良側）以外の方式")
    return fig


def fig_bars(data, score):
    t = auc_table(data, score)
    fig = go.Figure()
    pal = ["#dcdbd5", "#c9c8c2", "#b3b2ab", "#b9a3e3", "#9ec0ec", "#8fd6b9",       # 04 既存文面・Yes/No 由来の選択肢版（淡色）
           "#8a5cd1", "#2a78d6", "#1baf7a", "#eda100",                           # 比較ならではの聞き方（濃色）
           "#eb6834", "#0b0b0b", "#d64545", "#7a7974", "#123f7a"]                # Yes/No・評点・最悪側など（斜線）
    for m, col in zip(method_cols(score), pal):
        s = t[t["方式"] == m]
        if s.empty: continue
        fig.add_trace(go.Bar(x=s["疾患"], y=s["既知 vs その他"], name=label(m), marker_color=col,
                             marker_pattern_shape="/" if m in YN else "",
                             customdata=np.c_[s["既知 vs ダミー"], s["候補 vs ダミー"]],
                             hovertemplate=f"<b>{label(m)}</b><br>%{{x}}<br>既知 vs その他 %{{y:.3f}}<br>既知 vs ダミー %{{customdata[0]:.3f}}"
                                           "<br>候補 vs ダミー %{customdata[1]:.3f}<extra></extra>"))
    fig.update_layout(**LAYOUT, barmode="group", height=600, yaxis=dict(range=[0.4, 1.0], title="AUC 既知 vs その他"),
                      title=f"疾患ごとの AUC（既知 vs その他）。淡色＝04既存文面・Yes/No 由来、濃色＝比較ならではの聞き方、斜線＝選択式（最良側）以外")
    return fig


def fig_position(longs):
    d = pd.concat([l.assign(key=k) for k, l in longs.items()])
    pos = d.groupby(["variant", "position"])["p"].mean().unstack()
    none = d.drop_duplicates(["key", "round", "group", "variant"]).groupby("variant")["p_none"].mean()
    fig = go.Figure()
    for v in VIDS:
        if v not in pos.index: continue
        fig.add_trace(go.Scatter(x=[f"{i}番" for i in pos.columns] + ["該当なし"], y=list(pos.loc[v]) + [none[v]], mode="lines+markers", name=v,
                                 hovertemplate=f"<b>{v}</b><br>%{{x}}<br>平均確率 %{{y:.3f}}<extra></extra>"))
    fig.add_hline(y=1 / 6, line=dict(color="#c3c2b7", dash="dot"), annotation_text="均等なら 1/6")
    fig.update_layout(**LAYOUT, height=420, yaxis=dict(title="平均確率（5疾患・全ラウンド）"),
                      title="番号（位置）ごとの平均確率：遺伝子はシャッフルしているので、偏りは内容と無関係な『番号の癖』")
    return fig


def fig_rank(data, score):
    cols = method_cols(score)
    rows, ylab = [], []
    for key, name in DISEASES:
        if key not in data: continue
        w = data[key]; k = w.category.eq("known")
        rk = {m: w[m].rank(ascending=False, method="min") for m in cols if m in w}
        for i in w.index[k].to_series().sort_values(key=lambda s: rk["YN M1"][s], ascending=False):
            rows.append([rk[m][i] if m in rk else np.nan for m in cols]); ylab.append(f"{name}｜{w.at[i, 'symbol']}")
    z = np.array(rows, float)
    fig = go.Figure(go.Heatmap(z=z, x=[label(m) for m in cols], y=ylab, zmin=1, zmax=100, colorscale="RdYlGn_r",
                               text=np.where(np.isnan(z), "", np.nan_to_num(z).astype(int).astype(str)), texttemplate="%{text}",
                               colorbar=dict(title="順位"), hovertemplate="%{y}<br>%{x}<br>順位 %{z:.0f}<extra></extra>"))
    fig.add_vline(x=len(VIDS) - 0.5, line=dict(color="#0b0b0b", width=2))
    fig.update_layout(**LAYOUT, height=24 * len(ylab) + 240, margin=dict(l=190, t=180), yaxis=dict(autorange="reversed"),
                      xaxis=dict(side="top", tickangle=-30), title=f"既知遺伝子の順位（100中、緑＝上位。強制選択は {score}）", title_y=0.99)
    return fig


def fig_scatter(data, best):
    keys = [k for k, _ in DISEASES if k in data]
    fig = make_subplots(rows=1, cols=len(keys), subplot_titles=[n for k, n in DISEASES if k in data], horizontal_spacing=0.035)
    for j, key in enumerate(keys, start=1):
        w = data[key]
        rho = w[best].corr(w["YN M1"], method="spearman")
        fig.layout.annotations[j - 1].text += f"（ρ={rho:.2f}）"
        for cat, jp, col, mk in CAT:
            s = w[w.category.eq(cat)]
            fig.add_trace(go.Scatter(x=s["YN M1"], y=s[best], mode="markers", name=jp, legendgroup=cat, showlegend=(j == 1),
                                     marker=dict(color=col, symbol=mk, size=9 if cat == "known" else 6, opacity=0.9 if cat == "known" else 0.55,
                                                 line=dict(color="white", width=0.5)),
                                     customdata=np.c_[s["symbol"], [jp] * len(s)],
                                     hovertemplate="<b>%{customdata[0]}</b>（%{customdata[1]}）<br>Yes/No M1 %{x:.2f}<br>" + best + " %{y:.2f}<extra></extra>"),
                          row=1, col=j)
    fig.update_xaxes(title_text="Yes/No M1（対数オッズ）", row=1, col=(len(keys) + 1) // 2)
    fig.update_yaxes(title_text=label(best), row=1, col=1)
    fig.update_layout(**LAYOUT, height=420, title=f"遺伝子ごとの点数：横＝Yes/No の M1、縦＝強制選択の {label(best)}（右上ほど標的らしい）")
    return fig


def main():
    data, longs = load()
    t = pd.concat([auc_table(data, s).assign(score=s) for s in ("mean_p", "win_rate", "lr_none")]).drop_duplicates(["疾患", "方式"])
    avg = t.groupby("方式")[["既知 vs その他", "既知 vs ダミー", "候補 vs ダミー"]].mean().round(3).sort_values("既知 vs その他", ascending=False)
    print(f"疾患: {list(data)}\n5疾患平均:\n{avg.to_string()}")
    best = [m for m in avg.index if m not in YN][0]            # 選択式（最良側）の中で最良
    score = best.split(" ")[1]
    print("強制選択で最良:", best)
    figs = [fig_heat(data, score), fig_bars(data, score), fig_position(longs), fig_rank(data, score), fig_scatter(data, best)]
    if score != "lr_none": figs.insert(1, fig_heat(data, "lr_none"))
    if score != "mean_p": figs.insert(1, fig_heat(data, "mean_p"))
    out = os.path.join(ROOT, "outputs", "rankvar_charts.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("<html><head><meta charset='utf-8'><title>強制選択の質問パターン比較</title></head><body style='max-width:1300px;margin:auto'>"
                 + "".join(f.to_html(full_html=False, include_plotlyjs=(True if i == 0 else False)) for i, f in enumerate(figs)) + "</body></html>")
    print("saved:", out)
    return data, longs, figs


if __name__ == "__main__":
    main()
