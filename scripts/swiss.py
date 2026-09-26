"""ノートブック 04 の形式（5遺伝子＋該当なしから1つ選ぶ、質問 M2r）でスイス式トーナメントを試す検証スクリプト（GGUF 直接駆動）。

- 最初の RANDOM_ROUNDS ラウンドはランダムに5人組を作る（rank_variants.py と同じ乱数・同じ手順なので同じグループになる）。
- 残りのラウンドは、それまでの全データで推定した強さ（Luce の選択モデル＝多者択一の Bradley-Terry）が近い遺伝子どうしで5人組を作る。
  強さに少しだけ乱数を足してから並べ、毎回同じ組にならないようにする。
- 最後に全ラウンドのデータで強さを推定する。ランダムに 8 ラウンド組んだ場合（rank_variants.py）と同じ質問数で比べる。

使い方:
  python scripts/swiss.py --disease achondroplasia ra prostate_cancer scz cystinuria
出力: outputs/<疾患>_set100_swiss.csv（遺伝子ごと）と outputs/<疾患>_set100_swiss_long.csv（ラウンド × グループ × 位置）
"""
import argparse, glob, itertools, json, os, random, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from yesno_question_variants import ROOT, load_genes
from rank_variants import Engine, VARIANTS, GROUP_SIZE, prefix_text
from rank_bt import luce
from tournament import elo

VARIANT = "M2r"


def fit_luce(long, syms):
    """long（ラウンド × グループ × 位置の選択確率）から、遺伝子と「該当なし」の強さ log π を推定する。"""
    items = list(syms) + ["NONE"]; ix = {s: i for i, s in enumerate(items)}
    groups, pairs = [], []
    for _, d in long.groupby(["round", "group"]):
        d = d.sort_values("position")
        mem = [ix[s] for s in d["symbol"]] + [ix["NONE"]]; p = np.r_[d["p"].values, d["p_none"].iloc[0]]
        groups.append((mem, p))
        for a, b in itertools.combinations(range(len(mem)), 2):
            if p[a] + p[b] > 0: pairs.append((mem[a], mem[b], p[a] / (p[a] + p[b])))
    return luce(groups, len(items)), elo(len(items), pairs)


def random_groups(idx, rng):
    """rank_variants.py と同じ手順（シャッフル → 先頭から5つずつ → 端数は他から補充）。"""
    order = idx[:]; rng.shuffle(order); out = []
    for start in range(0, len(order), GROUP_SIZE):
        chunk = order[start:start + GROUP_SIZE]
        if len(chunk) < GROUP_SIZE: chunk = chunk + rng.sample([i for i in idx if i not in chunk], GROUP_SIZE - len(chunk))
        out.append(chunk)
    return out


def swiss_groups(idx, strength, rng, jitter):
    """強さの順に並べて先頭から5つずつ組む。強さに乱数（標準偏差 jitter × 強さの標準偏差）を足して毎回少しずらす。
    グループ内の並び（＝番号）はシャッフルして、番号の癖が強さと結び付かないようにする。"""
    s = np.array([strength[i] for i in idx]); s = s + np.array([rng.gauss(0, 1) for _ in idx]) * jitter * (s.std() or 1)
    order = [idx[k] for k in np.argsort(-s)]; out = []
    for start in range(0, len(order), GROUP_SIZE):
        chunk = order[start:start + GROUP_SIZE]
        if len(chunk) < GROUP_SIZE: chunk = chunk + order[start - (GROUP_SIZE - len(chunk)):start]      # 端数は直前（強さの近い）遺伝子で補充
        chunk = chunk[:]; rng.shuffle(chunk); out.append(chunk)
    return out


def run_disease(eng, key, reg, rounds, random_rounds, jitter, model_name):
    D = reg[key]; disease, info = D["name"], D["info"][:5]
    genes = load_genes(D["gene_prefix"], None)
    eng.set_prefix(prefix_text(disease, info))
    line = f"Question: {VARIANTS[VARIANT].format(disease=disease)} Answer: "      # 末尾の空白が必要（rank_variants.py と同じ）
    rng, idx, rows, t0 = random.Random(0), list(genes.index), [], time.time()
    syms = genes["symbol"].tolist()
    for r in range(rounds):
        if r < random_rounds:
            groups = random_groups(idx, rng)
        else:
            s_luce, _ = fit_luce(pd.DataFrame(rows), syms)
            strength = {i: s_luce[k] for k, i in enumerate(idx)}
            groups = swiss_groups(idx, strength, rng, jitter)
        for gi, chunk in enumerate(groups):
            p = eng.score([genes.at[i, "gene_label"] for i in chunk], {VARIANT: line})[VARIANT]
            for slot, i in enumerate(chunk):
                rows.append({"round": r, "group": gi, "position": slot + 1, "symbol": genes.at[i, "symbol"], "category": genes.at[i, "category"],
                             "p": round(float(p[slot]), 6), "p_none": round(float(p[-1]), 6), "win": int(np.argmax(p) == slot),
                             "design": "random" if r < random_rounds else "swiss"})
        print(f"  {key} round {r + 1}/{rounds} ({'random' if r < random_rounds else 'swiss'}, {time.time() - t0:.0f}s)", flush=True)
    long = pd.DataFrame(rows); long["model"] = model_name; long["disease"] = disease
    s_luce, s_elo = fit_luce(long, syms)
    out = pd.DataFrame({"symbol": syms, "category": genes["category"].values, "luce": s_luce[:-1], "elo": s_elo[:-1],
                        "mean_p": long.groupby("symbol")["p"].mean().reindex(syms).values})
    out["luce_above_none"] = out["luce"] > s_luce[-1]
    out["luce_rank"] = out["luce"].rank(ascending=False, method="min").astype(int)
    base = os.path.join(ROOT, "outputs", f"{D['gene_prefix']}_set100_swiss")
    out.to_csv(base + ".csv", index=False); long.to_csv(base + "_long.csv", index=False)
    print(f"{key}: {rounds} rounds ({random_rounds} random + {rounds - random_rounds} swiss) in {time.time() - t0:.0f}s -> {base}.csv", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", nargs="+", default=["achondroplasia", "ra", "prostate_cancer", "scz", "cystinuria"])
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--random-rounds", type=int, default=2)
    ap.add_argument("--jitter", type=float, default=0.3)
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    reg = json.load(open(os.path.join(ROOT, "data", "diseases.json"), encoding="utf-8"))
    model = a.model or sorted(glob.glob(os.path.expanduser("~/llm/models/**/*txgemma*.gguf"), recursive=True))[0]
    print("model:", model, flush=True)
    eng = Engine(model)
    for key in a.disease:
        run_disease(eng, key, reg, a.rounds, a.random_rounds, a.jitter, os.path.basename(model))


if __name__ == "__main__":
    main()
