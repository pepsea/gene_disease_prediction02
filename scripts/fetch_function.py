"""遺伝子の機能情報（UniProt の Function、GO の生物学的プロセス・分子機能、Reactome の経路名）を取得し、
疾患に触れる記述を取り除いてプロンプト用の短い文にする。

疾患との関係を直接書いた情報は入れない（症状の説明文に分子名を書かないのと同じ考え方）：
- UniProt の「Involvement in disease」欄は取得しない。
- Function の文のうち、疾患・病態を表す語（DISEASE_TERMS）を含む文は丸ごと除く。
- GO・Reactome の名前のうち、疾患を表す語や「Defective」「causes」などを含むものは除く。
- 最後に、登録疾患の病名が残っていないかを自動でチェックする（残っていれば例外）。

使い方: python scripts/fetch_function.py          # data/genes/<疾患>_set100.tsv の全遺伝子
出力: data/genes/function_cache.json（accession → {function, go_process, go_function, reactome, text}）
"""
import json, os, re, sys, time, urllib.request
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(ROOT, "data", "genes", "function_cache.json")
DISEASES = ["achondroplasia", "ra", "prostate_cancer", "scz", "cystinuria"]
DISEASE_TERMS = [                                        # 文・名前にこれを含んだら除く（小文字で照合）
    "disease", "disorder", "syndrome", "patient", "individuals with", "deficien", "mutation", "mutant", "defective", "causes ",
    "cancer", "tumor", "tumour", "carcinoma", "oncogen", "malignan", "metasta", "leukemia", "lymphoma",
    "arthritis", "rheumat", "autoimmun", "schizophren", "psychos", "psychiatric", "bipolar", "depress", "dwarf", "achondroplas",
    "dysplasia", "cystinuria", "stone", "urolith", "lithiasis", "infection", "viral", "virus", "susceptib", "risk", "pathogen",
]
PROMPT_BLOCK = "Function: {function}\nGO: {go}\nPathways: {reactome}\n"


def clean_sentences(text, max_sent=2, max_chars=320):
    text = re.sub(r"\(PubMed:[^)]*\)|\{ECO:[^}]*\}", "", text)                 # 文献・証拠コードを除く
    sents = [s.strip() for s in re.split(r"(?<=[.;])\s+", text) if s.strip()]
    keep = [s for s in sents if not any(t in s.lower() for t in DISEASE_TERMS)]
    out = " ".join(keep[:max_sent]).strip()
    return out[:max_chars].rsplit(" ", 1)[0] + "…" if len(out) > max_chars else out


def ok_name(name):
    return not any(t in name.lower() for t in DISEASE_TERMS)


def fetch(accs):
    url = ("https://rest.uniprot.org/uniprotkb/accessions?accessions=" + ",".join(accs) +
           "&fields=accession,cc_function,go_p,go_f,xref_reactome&format=json")
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)["results"]


def parse(entry):
    func = " ".join(t["value"] for c in entry.get("comments", []) if c.get("commentType") == "FUNCTION" for t in c.get("texts", []))
    go_p, go_f, rea = [], [], []
    for x in entry.get("uniProtKBCrossReferences", []):
        props = {p["key"]: p["value"] for p in x.get("properties", [])}
        if x["database"] == "GO":
            term = props.get("GoTerm", "")
            if term.startswith("P:") and ok_name(term): go_p.append(term[2:])
            if term.startswith("F:") and ok_name(term): go_f.append(term[2:])
        elif x["database"] == "Reactome" and ok_name(props.get("PathwayName", "")):
            rea.append(props.get("PathwayName", ""))
    f = clean_sentences(func)
    go = "; ".join(go_f[:2] + go_p[:4])
    rp = "; ".join(sorted(set(rea), key=len)[:3])                              # 短い（＝一般的な）経路名から3つ
    return {"function": f or "not described", "go": go or "not described", "reactome": rp or "not described",
            "text": PROMPT_BLOCK.format(function=f or "not described", go=go or "not described", reactome=rp or "not described")}


def leak_check(cache):
    reg = json.load(open(os.path.join(ROOT, "data", "diseases.json"), encoding="utf-8"))
    names = {v["name"].lower() for v in reg.values()}
    bad = {a: [n for n in names if n in v["text"].lower()] for a, v in cache.items()}
    bad = {a: n for a, n in bad.items() if n}
    if bad: raise SystemExit(f"病名が残っています: {bad}")
    print(f"leak check OK: {len(cache)} entries contain none of {sorted(names)}")


def main():
    accs = []
    for k in DISEASES:
        g = pd.read_csv(os.path.join(ROOT, "data", "genes", f"{k}_set100.tsv"), sep="\t", dtype=str).fillna("")
        accs += [u.split("|")[0] for u in g["uniprot_ids"] if u]
    accs = sorted(set(accs))
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    todo = [a for a in accs if a not in cache]
    print(f"{len(accs)} accessions, {len(todo)} to fetch")
    for i in range(0, len(todo), 50):
        for e in fetch(todo[i:i + 50]):
            cache[e["primaryAccession"]] = parse(e)
        time.sleep(0.5)
    for a in accs:
        cache.setdefault(a, parse({}))                                           # 見つからなかったもの
    json.dump(cache, open(CACHE, "w"), ensure_ascii=False, indent=1)
    leak_check(cache)
    print("saved:", CACHE)


if __name__ == "__main__":
    main()
