"""End-to-end replay of the Claude-judged RA demo through the loop."""
import pytest
from target_loop.config import ScoringRules
from target_loop.models import Category
from target_loop.pairwise import ranking_string
from demo.ra_demo import run, SEEDS_PROPOSED
from tests.test_scoring import EXPECTED


@pytest.fixture(scope="module")
def result():
    res, backend = run()
    return res, backend


def test_seed_correction_ctla4_to_cd80(result):
    res, _ = result
    assert res.seeds == ["CD80", "MS4A1", "TNF", "IL6R", "JAK1"]
    assert any("CTLA4: corrected to CD80" in n for n in res.seed_notes)


def test_all_demo_totals_reproduced(result):
    res, _ = result
    for g, (q, p, tot) in EXPECTED.items():
        c = res.candidates[g]
        assert c.q_score == pytest.approx(q, abs=1e-9), g
        assert c.p_score == pytest.approx(p, abs=1e-9), g
        assert c.total == pytest.approx(tot, abs=0.006), g      # transcript rounded to 2 decimals


def test_il10_excluded_by_direction(result):
    res, _ = result
    assert res.candidates["IL10"].category == Category.EXCLUDED


def test_reverse_drug_question_reclassifies_to_known(result):
    res, _ = result
    cats = res.by_category()
    assert set(cats[Category.KNOWN]) == {"CD80", "MS4A1", "TNF", "IL6R", "JAK1", "IL6", "TNFSF11"}
    assert set(cats[Category.MOA_NEAR]) == {"TNFRSF1A", "CSF2", "CSF2RA", "BTK", "TYK2", "IL17A", "IRAK4", "IL6ST", "MYD88"}
    assert cats[Category.MAP_NEAR] == ["IL11"]


def test_pairwise_and_stop(result):
    res, _ = result
    assert ranking_string(res.rounds[0].pairwise_ranking) == "TNFRSF1A > BTK ≒ CSF2 > TYK2"
    assert len(res.rounds) == 2
    assert "top-5 unchanged" in res.stop_reason
    assert set(res.rounds[1].new_candidates) == {"CSF2RA", "IL6ST", "MYD88"}


def test_round2_candidates_are_damped(result):
    res, _ = result
    for g in ("CSF2RA", "IL6ST", "MYD88"):
        assert res.candidates[g].round_found == 2
        assert res.candidates[g].total <= 0.7


def test_total_gene_count_is_18(result):
    res, _ = result
    assert len(res.candidates) == 18


def test_no_silent_fallback_in_recorded_backend():
    from target_loop.backends import RecordedBackend
    b = RecordedBackend({"yes": {}, "text": {}})
    with pytest.raises(KeyError):
        b.yes_probability("unrecorded question")


def test_genetics_question_changes_ranking():
    """With Q6 on, TYK2 (strong RA GWAS) should overtake CSF2 (no RA genetics)."""
    res, _ = run(ScoringRules(use_genetics_question=True))
    ranked = [c.symbol for c in res.ranked()]
    assert ranked.index("TYK2") < ranked.index("CSF2")
