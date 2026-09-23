"""遺伝子セット TSV の整合性（生成物がリポジトリに入っているので、その中身を検査する）。"""
import csv, os
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = os.path.join(ROOT, "data", "genes")


def read(name):
    with open(os.path.join(G, name), encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


@pytest.fixture(scope="module")
def hgnc():
    return {r["symbol"] for r in read("hgnc_protein_coding.tsv")}


@pytest.mark.parametrize("d", ["ra", "scz", "cystinuria", "prostate_cancer", "achondroplasia"])
def test_sets_sizes_and_containment(d, hgnc):
    s100, s1000 = read(f"{d}_set100.tsv"), read(f"{d}_set1000.tsv")
    assert len(s100) == 100 and len(s1000) == 1000
    sym100, sym1000 = {r["symbol"] for r in s100}, {r["symbol"] for r in s1000}
    assert len(sym100) == 100 and len(sym1000) == 1000          # 重複なし
    assert sym100 <= sym1000
    assert sym1000 <= hgnc                                        # 全部 HGNC のタンパク質コード遺伝子
    known = {r["symbol"] for r in read(f"{d}_known.tsv")}
    assert known <= sym100                                        # 正解は 100 セットに全部入る
    cats = {r["category"] for r in s1000}
    assert cats == {"known", "candidate", "random"}
    assert all((r["label"] == "1") == (r["category"] == "known") for r in s1000)


@pytest.mark.parametrize("d", ["ra", "scz", "cystinuria", "prostate_cancer", "achondroplasia"])
def test_random_excludes_known_and_candidates(d):
    rand = {r["symbol"] for r in read(f"{d}_random.tsv")}
    kc = {r["symbol"] for r in read(f"{d}_known.tsv")} | {r["symbol"] for r in read(f"{d}_candidates.tsv")}
    assert not (rand & kc)


def test_columns_present():
    row = read("ra_set100.tsv")[0]
    for c in ("symbol", "hgnc_id", "gene_name", "uniprot_ids", "protein_name_uniprot", "alias_symbols", "category", "label", "evidence", "source"):
        assert c in row


def test_ra_demo_genes_resolve_in_hgnc(hgnc):
    from demo.ra_demo import ANNOT
    assert set(ANNOT) <= hgnc


@pytest.mark.parametrize("d", ["scz", "cystinuria", "prostate_cancer", "achondroplasia"])
def test_slc_decoys_present(d):
    rand = read(f"{d}_random.tsv")
    slc = [r for r in rand if r["note"] == "SLC decoy"]
    assert len(slc) >= 300 and all(r["symbol"].startswith("SLC") for r in slc)
    s100 = read(f"{d}_set100.tsv")
    assert sum(1 for r in s100 if r["note"] == "SLC decoy") >= 10


def test_cystinuria_causal_genes_are_known_and_not_decoys():
    known = {r["symbol"] for r in read("cystinuria_known.tsv")}
    assert known == {"SLC3A1", "SLC7A9"}
    assert not (known & {r["symbol"] for r in read("cystinuria_random.tsv")})


def test_disease_registry_matches_gene_files():
    import json
    reg = json.load(open(os.path.join(ROOT, "data", "diseases.json"), encoding="utf-8"))
    hg = {r["symbol"] for r in read("hgnc_protein_coding.tsv")}
    for key, d in reg.items():
        assert os.path.exists(os.path.join(G, f"{d['gene_prefix']}_set100.tsv")), key
        assert 1 <= len(d["info"]) <= 5
        for b in d["info"]:                       # 分子名・遺伝子記号の漏れが無いこと
            for tok in __import__("re").findall(r"\b[A-Z][A-Z0-9]{1,9}\b", b):
                assert tok not in hg, (key, tok)
