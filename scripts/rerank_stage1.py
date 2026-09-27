"""段階1の代わり：リランカー（cross-encoder）で全遺伝子を採点し、既知の標的の順位を埋め込み（bge-m3）と比べる。

- 文書＝遺伝子の説明（段階0と同じ gene_text：タンパク質名＋機能・GO・Reactome。疾患に触れる記述は除去済み）
- 質問＝疾患の説明（病名＋症状の箇条書き）。bge は「そのまま」と「治療標的を探す言い方」の2通り、Qwen3 は指示文付き。
- モデル：BAAI/bge-reranker-v2-m3（点数＝ロジット）、Qwen/Qwen3-Reranker-0.6B（点数＝log P(yes) − log P(no)）

使い方: python scripts/rerank_stage1.py --disease ra [--models bge,qwen06] [--instr base|m3s]
出力: outputs/rerank/<疾患>_rerank.csv（遺伝子 × 設定の点数と順位）
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
QWEN_INSTR = {                                                          # Qwen3-Reranker の指示文（--instr で選ぶ）
    "base": ("Given a disease description, judge whether inhibiting or activating the gene/protein described in the document "
             "could plausibly treat the disease or improve its symptoms"),
    "m3s": ("Given a disease description, judge whether the gene/protein described in the document meets at least one of these criteria:\n"
            "(a) Inhibiting or activating it could plausibly treat the disease or improve at least one of its symptoms.\n"
            "(b) Even outside the causal pathway, modulating it could counteract the abnormal process of the disease through a parallel or opposing pathway.\n"
            "(c) Its own substrate, ligand, pathway, cell type or circuit precisely matches the mechanism of the disease "
            "(a similar but distinct role, even in the same gene family, does not count)."),
}


def load(dkey):
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "pipeline07.yaml"), encoding="utf-8"))
    D = {d["key"]: d for d in yaml.safe_load(open(os.path.join(ROOT, cfg["diseases_file"]), encoding="utf-8"))["diseases"]}[dkey]
    g = pd.read_csv(os.path.join(ROOT, cfg["paths"]["gene_universe"]), sep="\t", dtype=str).fillna("")
    g["acc"] = g["uniprot_ids"].str.split("|").str[0]
    g["label"] = [f"{s} ({(p or n).split('|')[0]})" if (p or n) else s for s, p, n in zip(g["symbol"], g["protein_name_uniprot"], g["gene_name"])]
    g = g.drop_duplicates("symbol").reset_index(drop=True)
    cache = json.load(open(os.path.join(ROOT, cfg["paths"]["function_cache"])))
    docs = [f"{lab}. {cache.get(a, {}).get('function', '')} GO: {cache.get(a, {}).get('go', '')}. Pathways: {cache.get(a, {}).get('reactome', '')}"
            for lab, a in zip(g["label"], g["acc"])]                   # 段階0の gene_text と同じ
    return D, g, docs


def run_bge(queries, docs, bs=32):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
    m = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3", torch_dtype=torch.float16).to(DEV).eval()
    out = {}
    for name, q in queries.items():
        s, t0 = [], time.time()
        for i in range(0, len(docs), bs):
            x = tok([q] * len(docs[i:i + bs]), docs[i:i + bs], padding=True, truncation="only_second", max_length=512, return_tensors="pt").to(DEV)
            with torch.no_grad(): s += m(**x).logits.view(-1).float().cpu().tolist()
            if i % (bs * 100) == 0: print(f"  bge/{name} {i}/{len(docs)} ({time.time() - t0:.0f}s)", flush=True)
        out[f"bge_{name}"] = s
    return out


def run_qwen(query, docs, repo="Qwen/Qwen3-Reranker-0.6B", bs=16, instr="base"):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(repo, padding_side="left")
    m = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16).to(DEV).eval()
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    pre = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. "
           "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    suf = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    s, t0 = [], time.time()
    for i in range(0, len(docs), bs):
        texts = [f"{pre}<Instruct>: {QWEN_INSTR[instr]}\n<Query>: {query}\n<Document>: {d}{suf}" for d in docs[i:i + bs]]
        x = tok(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(DEV)
        with torch.no_grad(): lg = m(**x, logits_to_keep=1).logits[:, -1, :].float()   # 最後の位置だけ語彙の確率を計算（全位置だと数十GB）
        s += (lg[:, yes] - lg[:, no]).cpu().tolist()                    # log P(yes) − log P(no)
        if i % (bs * 200) == 0: print(f"  {repo.split('/')[-1]} {i}/{len(docs)} ({time.time() - t0:.0f}s)", flush=True)
    name = "qwen" + repo.split("-")[-1].replace(".", "").replace("B", "").lower()
    return {name if instr == "base" else f"{name}_{instr}": s}             # 列名：qwen06（base）、qwen06_m3s など


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--disease", default="ra"); ap.add_argument("--models", default="bge,qwen06"); ap.add_argument("--instr", default="base")
    a = ap.parse_args()
    D, g, docs = load(a.disease)
    plain = f"{D['name']}. " + " ".join(D["info"])                    # 段階1の埋め込みと同じ質問文
    target = f"Which gene or protein could be a drug target to treat {D['name']} or improve its symptoms? " + " ".join(D["info"])
    out = os.path.join(ROOT, "outputs", "rerank"); os.makedirs(out, exist_ok=True)
    p = os.path.join(out, f"{a.disease}_rerank.csv")

    def save(res):                                                     # 方法ごとに保存（途中で止まっても前の方法の結果は残る）
        t = pd.DataFrame({"symbol": g["symbol"], **res})
        for c in res: t[f"rank_{c}"] = t[c].rank(ascending=False, method="first").astype(int)
        if os.path.exists(p):                                          # 既存の結果に列を足す
            old = pd.read_csv(p); t = old.drop(columns=[c for c in t.columns if c in old.columns and c != "symbol"]).merge(t, on="symbol")
        t.to_csv(p, index=False); print("saved:", p, list(res), flush=True)

    if "bge" in a.models: save(run_bge({"plain": plain, "target": target}, docs))
    if "qwen06" in a.models: save(run_qwen(plain, docs, "Qwen/Qwen3-Reranker-0.6B", instr=a.instr))
    if "qwen4" in a.models: save(run_qwen(plain, docs, "Qwen/Qwen3-Reranker-4B", bs=4, instr=a.instr))

if __name__ == "__main__":
    main()
