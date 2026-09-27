"""方向の推定 ステップ1：LLM に「阻害すると改善するか」「活性化すると改善するか」を別々に聞き、既知37遺伝子で当たり具合を見る。

- 質問は config/pipeline07.yaml の stage7.questions（言い回し V1〜V3 × 阻害・活性化）。M3s と同じ書き方で、1問ずつ独立に聞く。
- 両順（Yes or No / No or Yes）× 機能情報なし・あり（UniProt の3行。病名を含む行は除く）で聞き、対数オッズを両順で平均する。
- 前置きは yesno_question_variants.py と同じ（M3s と同じ前置き）。
- 正解は stage7.truth_file（net_effect：inhibit / activate / neither）。

使い方: python scripts/direction_step1.py
出力: outputs/direction/step1_known37.csv（遺伝子 × 言い回し × 機能情報 の、阻害・活性化それぞれの対数オッズ）
"""
import json, math, os, sys, time
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yesno_question_variants import ROOT, Engine, prefix_text, load_genes, logit


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "pipeline07.yaml"), encoding="utf-8"))
    SHORT = {"rheumatoid arthritis": "ra", "schizophrenia": "scz", "cystinuria": "cystinuria", "prostate cancer": "prostate_cancer",
             "achondroplasia": "achondroplasia"}                      # 病名 → 検証用ファイル（data/genes/<短い名前>_set100.tsv）と正解表の disease 列
    D = {SHORT[d["name"]]: d for d in yaml.safe_load(open(os.path.join(ROOT, cfg["diseases_file"]), encoding="utf-8"))["diseases"] if d["name"] in SHORT}
    truth = pd.read_csv(os.path.join(ROOT, cfg["stage7"]["truth_file"]), sep="\t")
    cache = json.load(open(os.path.join(ROOT, cfg["paths"]["function_cache"])))       # 旧形式の機能情報（06 と同じ）
    Q = cfg["stage7"]["questions"]
    import glob
    model = sorted(glob.glob(os.path.expanduser("~/llm/models/**/*txgemma*.gguf"), recursive=True))[0]
    eng = Engine(model)
    rows, t0 = [], time.time()
    for key, d in D.items():
        tk = truth[truth["disease"] == key]
        if tk.empty: continue
        genes = load_genes(key if key != "prostate_cancer" else "prostate_cancer", None).set_index("symbol")
        terms = [d["name"].lower()]
        for order in ("yes_first", "no_first"):
            eng.set_prefix(prefix_text(d["name"], d["info"][:5], order))
            for s in tk["symbol"]:
                acc = genes.at[s, "uniprot_ids"].split("|")[0]
                ftxt = "".join(l + "\n" for l in cache.get(acc, {}).get("text", "").splitlines() if l and not any(x in l.lower() for x in terms))
                for fn, extra in (("no", ""), ("yes", ftxt)):
                    lines = {f"{v}_{dirn}": f"D{dirn[0].upper()}. {Q[v][dirn].format(gene=s, disease=d['name'])} Answer:" for v in Q for dirn in ("inhibit", "activate")}
                    p = eng.score(genes.at[s, "gene_label"], lines, extra)
                    for k, v in p.items():
                        rows.append({"disease": key, "symbol": s, "order": order, "function": fn, "variant": k.split("_")[0], "direction": k.split("_")[1], "lo": logit(v)})
        print(f"  {key}: {len(tk)} genes ({time.time() - t0:.0f}s)", flush=True)
    long = pd.DataFrame(rows)
    w = long.groupby(["disease", "symbol", "function", "variant", "direction"])["lo"].mean().unstack("direction").reset_index()
    w = w.merge(truth[["disease", "symbol", "net_effect", "action_class"]], on=["disease", "symbol"])
    w["diff"] = w["inhibit"] - w["activate"]                                # 正なら阻害寄り、負なら活性化寄り
    out = os.path.join(ROOT, "outputs", "direction"); os.makedirs(out, exist_ok=True)
    w.to_csv(os.path.join(out, "step1_known37.csv"), index=False); long.to_csv(os.path.join(out, "step1_known37_long.csv"), index=False)
    print(f"done in {time.time() - t0:.0f}s -> {out}/step1_known37.csv")


if __name__ == "__main__":
    main()
