"""総当たり（2遺伝子の対戦）で上位の並びを精密にする検証スクリプト（GGUF 直接駆動）。ノートブック 05 の元。

1. M3s（Yes/No、ノートブック 03）の点数で上位 TOP_K 遺伝子に絞る（outputs/<疾患>_set100_qvariants.csv の M3s を使う）。
2. その中の全ペアについて「M3s の3条件のどれかによりよく当てはまるのはどちらか」を 1/2 で答えさせる。左右を入れ替えた2回を聞く。
3. 対戦結果（勝つ確率＝やわらかい勝ち数）から Bradley-Terry モデル（MM 法）と Elo レーティングで強さを推定する。

数字は「Answer: 」（末尾の空白まで）の直後で読む（Gemma は空白付きの数字トークンを持たないため）。

使い方:
  python scripts/tournament.py --disease achondroplasia ra prostate_cancer scz cystinuria
  python scripts/tournament.py --disease ra --top-k 10          # 試運転（45 ペア）
出力: outputs/<疾患>_set100_tournament.csv（遺伝子ごと）と outputs/<疾患>_set100_tournament_pairs.csv（対戦ごと）
"""
import argparse, glob, itertools, json, math, os, random, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from yesno_question_variants import ROOT, load_genes, logit
from rank_variants import STRICT_NOTE as RANK_NOTE

ROLE = "You are an expert in drug discovery and human disease biology.\n"
PAIR_QUESTION = ("Consider these three criteria:\n"
                 "(a) Inhibiting or activating the gene could plausibly treat {disease} or improve at least one of the symptoms listed above.\n"
                 "(b) Even outside the causal pathway, modulating the gene could counteract the abnormal process described above through a "
                 "parallel or opposing pathway.\n"
                 "(c) The gene's own substrate, ligand, pathway, cell type or circuit precisely matches the mechanism of {disease} described above "
                 "(a similar but distinct role, even in the same gene family, does not count).\n"
                 "Which of the two numbered genes above better meets at least one of these criteria?")


def pair_prefix(disease, info):
    bullets = "\n".join(f"- {b}" for b in info)
    return (ROLE + "You will be shown two candidate genes for one disease, and asked which ONE number is the better answer.\n" + RANK_NOTE +
            f"Disease: {disease}\nTarget symptoms and the organ, cell and functional abnormalities behind them:\n{bullets}\n"
            "Answer with a single number, 1 or 2, only. No words, no explanation.\n\n")


def pair_block(label1, label2):
    return f"Candidate genes:\n1. {label1}\n2. {label2}\n"


def question_line(disease):
    return f"Question: {PAIR_QUESTION.format(disease=disease)} Answer: "


def bradley_terry(n, games, iters=500, prior=0.5):
    """games: [(i, j, w)]（w = i が j に勝つ確率、1試合ぶん）。MM 法で強さ π を推定し log π を返す。
    prior：各遺伝子に「強さ1の仮想相手」との prior 勝ち・prior 負けを足して、全勝・全敗でも発散しないようにする。"""
    W = np.full(n, prior)                                          # やわらかい勝ち数（仮想相手からの prior 勝ちを含む）
    pairs = {}
    for i, j, w in games:
        W[i] += w; W[j] += 1 - w
        pairs[(i, j)] = pairs.get((i, j), 0) + 1
    pi = np.ones(n)
    for _ in range(iters):
        denom = 2 * prior / (pi + 1)                               # 仮想相手（π=1）との 2*prior 試合
        for (i, j), c in pairs.items():
            d = c / (pi[i] + pi[j]); denom[i] += d; denom[j] += d
        new = W / denom
        new /= math.exp(np.mean(np.log(new)))                      # 幾何平均を 1 に固定
        if np.max(np.abs(np.log(new) - np.log(pi))) < 1e-9: pi = new; break
        pi = new
    return np.log(pi)


def elo(n, games, k=24.0, shuffles=50, seed=0):
    """Elo レーティング（初期 1500）。対戦の順番で結果が変わるので、順番をシャッフルして平均する。"""
    rng, total = random.Random(seed), np.zeros(n)
    for _ in range(shuffles):
        r, g = np.full(n, 1500.0), games[:]; rng.shuffle(g)
        for i, j, w in g:
            e = 1 / (1 + 10 ** ((r[j] - r[i]) / 400))
            r[i] += k * (w - e); r[j] -= k * (w - e)
        total += r
    return total / shuffles


def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def run_disease(eng, key, reg, top_k, model_name):
    D = reg[key]; disease, info = D["name"], D["info"][:5]
    genes = load_genes(D["gene_prefix"], None).set_index("symbol")
    qv = pd.read_csv(os.path.join(ROOT, "outputs", f"{D['gene_prefix']}_set100_qvariants.csv"))
    qv = qv[qv["qid"] == "M3s"].assign(lo=lambda d: d["p_yes"].map(logit))
    m3s = qv.groupby("symbol")["lo"].mean().rename("m3s_logodds")
    top = m3s.sort_values(ascending=False).head(top_k).index.tolist()
    eng.set_prefix("pair", pair_prefix(disease, info))
    q = question_line(disease)
    rows, t0 = [], time.time()
    for a, b in itertools.combinations(range(len(top)), 2):
        la, lb = genes.at[top[a], "gene_label"], genes.at[top[b], "gene_label"]
        p_ab = eng.digits("pair", pair_block(la, lb) + q, (1, 2))   # a が1番
        p_ba = eng.digits("pair", pair_block(lb, la) + q, (1, 2))   # b が1番
        rows.append({"a": top[a], "b": top[b], "p_a_first": float(p_ab[0]), "p_a_second": float(p_ba[1])})
    print(f"  {key}: {len(rows)} pairs x 2 orders in {time.time() - t0:.0f}s", flush=True)
    pr = pd.DataFrame(rows)
    idx = {s: i for i, s in enumerate(top)}
    g_all = [(idx[r.a], idx[r.b], r.p_a_first) for r in pr.itertuples()] + [(idx[r.a], idx[r.b], r.p_a_second) for r in pr.itertuples()]
    half = np.random.default_rng(0).permutation(len(pr)) % 2              # 安定性：ペアを無作為に2分割（どちらも両順を含む）して BT を比べる
    g_h = [[g for h, r in zip(half, pr.itertuples()) if h == k for g in ((idx[r.a], idx[r.b], r.p_a_first), (idx[r.a], idx[r.b], r.p_a_second))]
           for k in (0, 1)]
    n = len(top)
    bt, el = bradley_terry(n, g_all), elo(n, g_all)
    win = np.zeros(n); cnt = np.zeros(n)
    for i, j, w in g_all: win[i] += w; win[j] += 1 - w; cnt[i] += 1; cnt[j] += 1
    out = pd.DataFrame({"symbol": top, "category": [genes.at[s, "category"] for s in top], "m3s_logodds": [m3s[s] for s in top],
                        "bt": bt, "elo": el, "win_rate": win / cnt})
    out["m3s_rank"] = np.arange(1, n + 1)
    out["bt_rank"] = out["bt"].rank(ascending=False, method="min").astype(int)
    out["model"] = model_name; out["disease"] = disease
    first_pos = float(np.mean(list(pr["p_a_first"]) + list(1 - pr["p_a_second"])))   # 1番に置かれた方が勝つ確率の平均
    stab = spearman(bradley_terry(n, g_h[0]), bradley_terry(n, g_h[1]))   # 左右で分けると位置の癖が片方ずつに乗るので、ペアで分ける
    base = os.path.join(ROOT, "outputs", f"{D['gene_prefix']}_set100_tournament")
    out.to_csv(base + ".csv", index=False); pr.assign(model=model_name, disease=disease).to_csv(base + "_pairs.csv", index=False)
    print(f"{key}: top {n}, first-position win prob {first_pos:.3f}, split-half (pairs) Spearman(BT) {stab:.3f} -> {base}.csv", flush=True)
    out.attrs.update(first_pos=first_pos, stability=stab)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", nargs="+", default=["achondroplasia", "ra", "prostate_cancer", "scz", "cystinuria"])
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    from ask_modes import Engine
    reg = json.load(open(os.path.join(ROOT, "data", "diseases.json"), encoding="utf-8"))
    model = a.model or sorted(glob.glob(os.path.expanduser("~/llm/models/**/*txgemma*.gguf"), recursive=True))[0]
    print("model:", model, flush=True)
    eng = Engine(model)
    for key in a.disease:
        run_disease(eng, key, reg, a.top_k, os.path.basename(model))


if __name__ == "__main__":
    main()
