"""遺伝子リスト（TSV）を作る。

出力（data/genes/）
  hgnc_protein_coding.tsv      HGNC のタンパク質コード遺伝子 全件（記号・遺伝子名・別名・UniProt ID）。
                               ループの known_symbols（記号の照合）にも使う。
  <disease>_known.tsv          正解遺伝子（承認薬の標的）
  <disease>_candidates.tsv     可能性遺伝子（GWAS・エクソーム・CNV・臨床試験・生物学）
  <disease>_random.tsv         ランダム遺伝子（固定シードで抽出。正解・可能性を除く）
  <disease>_set100.tsv         正解 + 可能性 + ランダム = 100 遺伝子（順序はシャッフル）
  <disease>_set1000.tsv        正解 + 可能性 + ランダム = 1000 遺伝子（順序はシャッフル、set100 を含む）

列
  symbol            HGNC 承認記号（curated の記号が旧記号・別名なら現行記号に直す）
  hgnc_id, entrez_id, ensembl_gene_id
  gene_name         HGNC の承認名（タンパク質コード遺伝子ではほぼ UniProt の推奨タンパク質名と一致）
  uniprot_ids       HGNC が持つ UniProt アクセッション（複数は | 区切り）
  protein_name_uniprot  UniProt の推奨タンパク質名。--uniprot で REST から取得（未取得なら空欄）
  alias_symbols     HGNC の alias_symbol + prev_symbol（| 区切り）
  disease, category (known / candidate / random), label (known=1, それ以外 0)
  evidence, note, source

入力
  data/raw/hgnc_complete_set.txt   HGNC 完全版（CC0）  取得: 
      https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt
  data/curated/<disease>_known.tsv, <disease>_candidates.tsv   人が点検する表（既定は Claude の知識で作成、未検証）

オプション
  --opentargets   Open Targets GraphQL から known（承認薬の標的）と candidates（遺伝学的関連）を取り直して
                  curated の代わりに使う（この開発環境からは API に届かず未テスト）
  --uniprot       UniProt REST から protein_name_uniprot を埋める（同じく未テスト）
"""
import argparse, csv, json, os, random, sys, urllib.request, urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HGNC = os.path.join(ROOT, "data", "raw", "hgnc_complete_set.txt")
CURATED = os.path.join(ROOT, "data", "curated")
OUT = os.path.join(ROOT, "data", "genes")
DISEASES = {"ra": ("rheumatoid arthritis", "EFO_0000685"), "scz": ("schizophrenia", "MONDO_0005090")}
SEED = 20260921
COLUMNS = ["symbol", "hgnc_id", "entrez_id", "ensembl_gene_id", "gene_name", "uniprot_ids", "protein_name_uniprot",
           "alias_symbols", "disease", "category", "label", "evidence", "note", "source"]


def load_hgnc():
    """HGNC 完全版を読み、(タンパク質コード遺伝子の dict, 記号・別名・旧記号 → 現行記号 の辞書) を返す。

    解決の優先順位: 現行の承認記号 > 別名・旧記号。
    （別名は他の遺伝子の承認記号と衝突することがある。例: "DLG2" は MPP2 の別名でもある。）
    """
    rows = []
    with open(HGNC, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["status"] == "Approved":
                rows.append(row)
    resolve = {row["symbol"]: row["symbol"] for row in rows}          # 1) 承認記号
    for row in rows:                                                     # 2) 別名・旧記号（未登録のときだけ）
        for a in (row["alias_symbol"] or "").split("|") + (row["prev_symbol"] or "").split("|"):
            if a and a not in resolve:
                resolve[a] = row["symbol"]
    genes = {}
    for row in rows:
        if row["locus_group"] != "protein-coding gene":
            continue
        sym = row["symbol"]
        aliases = [a for a in (row["alias_symbol"] or "").split("|") if a] + [p for p in (row["prev_symbol"] or "").split("|") if p]
        genes[sym] = {"symbol": sym, "hgnc_id": row["hgnc_id"], "entrez_id": row["entrez_id"],
                      "ensembl_gene_id": row["ensembl_gene_id"], "gene_name": row["name"],
                      "uniprot_ids": row["uniprot_ids"], "protein_name_uniprot": "", "alias_symbols": "|".join(aliases)}
    return genes, resolve


def read_curated(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def fetch_opentargets(efo_id):
    """Open Targets GraphQL から承認薬の標的（known）と遺伝学的関連（candidates）を取る。未テスト。"""
    url = "https://api.platform.opentargets.org/api/v4/graphql"
    q = """query($efo: String!) {
      disease(efoId: $efo) {
        knownDrugs(size: 2000) { rows { target { approvedSymbol } drug { name } phase status } }
        associatedTargets(page: {index: 0, size: 300}, datasourceIds: ["gwas_credible_sets", "gene_burden", "eva", "orphanet", "genomics_england"]) {
          rows { target { approvedSymbol } score datatypeScores { id score } }
        }
      }
    }"""
    req = urllib.request.Request(url, data=json.dumps({"query": q, "variables": {"efo": efo_id}}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)["data"]["disease"]
    known = {}
    for row in d["knownDrugs"]["rows"]:
        if row["phase"] == 4:
            s = row["target"]["approvedSymbol"]
            known.setdefault(s, {"symbol": s, "drug_examples": [], "approval_note": "Open Targets knownDrugs phase 4", "source": "Open Targets (CC0)"})
            known[s]["drug_examples"].append(row["drug"]["name"])
    for v in known.values():
        v["drug_examples"] = "; ".join(sorted(set(v["drug_examples"])))
    cands = []
    for row in d["associatedTargets"]["rows"]:
        s = row["target"]["approvedSymbol"]
        if s in known: continue
        cands.append({"symbol": s, "evidence": "genetic_association", "note": f"Open Targets genetic score {row['score']:.3f}",
                      "source": "Open Targets (CC0)"})
    return list(known.values()), cands


def fetch_uniprot_names(accessions):
    """UniProt REST から推奨タンパク質名を取る。未テスト。"""
    names = {}
    accs = [a for a in accessions if a]
    for i in range(0, len(accs), 100):
        chunk = accs[i:i + 100]
        query = " OR ".join(f"accession:{a}" for a in chunk)
        url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(
            {"query": query, "fields": "accession,protein_name", "format": "tsv", "size": 500})
        with urllib.request.urlopen(url, timeout=120) as r:
            lines = r.read().decode().splitlines()[1:]
        for line in lines:
            acc, name = line.split("\t")[:2]
            names[acc] = name
    return names


def write_tsv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opentargets", action="store_true")
    ap.add_argument("--uniprot", action="store_true")
    ap.add_argument("--random-size", type=int, default=0, help="random.tsv の件数（既定: set1000 を満たす数）")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    genes, resolve = load_hgnc()
    write_tsv(os.path.join(OUT, "hgnc_protein_coding.tsv"), [dict(g, disease="", category="", label="") for g in genes.values()])
    print(f"hgnc_protein_coding.tsv: {len(genes)} genes")

    for key, (disease, efo) in DISEASES.items():
        if args.opentargets:
            known_rows, cand_rows = fetch_opentargets(efo)
        else:
            known_rows = read_curated(os.path.join(CURATED, f"{key}_known.tsv"))
            cand_rows = read_curated(os.path.join(CURATED, f"{key}_candidates.tsv"))

        def build(rows, category, label):
            out, seen = [], set()
            for r in rows:
                sym = resolve.get(r["symbol"])
                if sym is None or sym not in genes:
                    print(f"  [warn] {key}: {r['symbol']} not a protein-coding HGNC symbol; skipped"); continue
                if sym != r["symbol"]:
                    print(f"  [info] {key}: {r['symbol']} -> {sym}")
                if sym in seen: continue
                seen.add(sym)
                g = dict(genes[sym], disease=disease, category=category, label=label,
                         evidence=r.get("evidence", "approved_drug"), source=r.get("source", ""),
                         note=r.get("note") or f"{r.get('drug_examples', '')} — {r.get('approval_note', '')}".strip(" —"))
                out.append(g)
            return out

        known = build(known_rows, "known", 1)
        cands = [c for c in build(cand_rows, "candidate", 0) if c["symbol"] not in {k["symbol"] for k in known}]
        used = {g["symbol"] for g in known + cands}
        pool = sorted(s for s in genes if s not in used)
        rng = random.Random(SEED)
        n_random = args.random_size or max(1000 - len(known) - len(cands), 0)
        rand = [dict(genes[s], disease=disease, category="random", label=0, evidence="", source=f"HGNC protein-coding, random seed {SEED}", note="")
                for s in rng.sample(pool, n_random)]

        n_rand100 = 100 - len(known) - len(cands)
        if n_rand100 < 0:      # 可能性遺伝子が多すぎる場合は先頭から削って 100 に収める
            set100 = known + cands[:100 - len(known)]
        else:
            set100 = known + cands + rand[:n_rand100]
        set1000 = known + cands + rand
        rng2 = random.Random(SEED + 1); rng2.shuffle(set100)
        rng3 = random.Random(SEED + 2); rng3.shuffle(set1000)

        if args.uniprot:
            accs = sorted({a for g in set1000 for a in g["uniprot_ids"].split("|") if a})
            names = fetch_uniprot_names(accs)
            for g in set1000:
                g["protein_name_uniprot"] = "|".join(names.get(a, "") for a in g["uniprot_ids"].split("|") if a)

        for name, rows in [("known", known), ("candidates", cands), ("random", rand), ("set100", set100), ("set1000", set1000)]:
            write_tsv(os.path.join(OUT, f"{key}_{name}.tsv"), rows)
        print(f"{key}: known {len(known)}, candidates {len(cands)}, random {len(rand)}, set100 {len(set100)}, set1000 {len(set1000)}")

    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as f:
        f.write(__doc__.strip() + "\n")


if __name__ == "__main__":
    main()
