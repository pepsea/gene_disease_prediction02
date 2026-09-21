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


@pytest.mark.parametrize("d", ["ra", "scz"])
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


@pytest.mark.parametrize("d", ["ra", "scz"])
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
