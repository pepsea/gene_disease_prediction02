"""The demo numbers from the transcript must be reproduced from the raw answers."""
import math
import pytest
from target_loop import scoring as S
from target_loop.config import ScoringRules, DEFAULT_RULES
from target_loop.models import Candidate, Direction
from demo.ra_demo import CHAIN, ANSWERS

# gene -> (Q, P, total) as printed in the transcript (round-2 genes include x0.7)
EXPECTED = {
    "IL6": (0.95, 1.0, 0.95), "CSF2": (0.95, 1.0, 0.95), "TNFRSF1A": (0.90, 1.0, 0.90),
    "TYK2": (0.90, 1.0, 0.90), "BTK": (0.90, 1.0, 0.90), "IL17A": (0.85, 1.0, 0.85),
    "TNFSF11": (0.95, 0.7, 0.67), "IRAK4": (0.85, 0.7, 0.60), "IL11": (0.60, 0.6, 0.36),
    "CSF2RA": (0.95, 1.0, 0.67), "IL6ST": (0.80, 1.0, 0.56), "MYD88": (0.65, 0.7, 0.32),
}


def answers(g):
    return {f"Q{i+1}": v for i, v in enumerate(ANSWERS[g])}


@pytest.mark.parametrize("gene,q_exp", [(g, v[0]) for g, v in EXPECTED.items()])
def test_question_score_matches_transcript(gene, q_exp):
    assert S.question_score(answers(gene)) == pytest.approx(q_exp, abs=1e-9)


def test_side_effect_question_is_inverted():
    a = {"Q1": 1, "Q2": 1, "Q3": 1, "Q4": 1, "Q5": 1.0}   # certain severe side effects
    assert S.question_score(a) == pytest.approx(0.8)
    a["Q5"] = 0.0
    assert S.question_score(a) == pytest.approx(1.0)


@pytest.mark.parametrize("m,n,v,p_exp", [
    (1.0, 1.0, 1.0, 1.0),      # IL6
    (0.7, 0.5, 0.0, 0.7),      # TNFSF11: M wins over 0.8*0.5 and 0.6*0
    (0.7, 0.5, 0.5, 0.7),      # IRAK4
    (0.0, 0.5, 1.0, 0.6),      # IL11: vector only
    (0.0, 1.0, 0.0, 0.8),      # direct partner off chain
    (0.7, 1.0, 0.0, 0.8),      # ambiguity found in verification: N direct beats adjacent M
])
def test_proximity_max_rule(m, n, v, p_exp):
    assert S.proximity_score(m, n, v) == pytest.approx(p_exp)


def test_round_damping():
    assert S.total_score(0.95, 1.0, 1) == pytest.approx(0.95)
    assert S.total_score(0.95, 1.0, 2) == pytest.approx(0.665)
    assert S.total_score(0.95, 1.0, 3) == pytest.approx(0.4655)


def test_opposite_direction_is_excluded_regardless_of_score():
    c = Candidate("IL10", stage="S4", direction=Direction.BETTER, network_hops=2, vector_bin="mid",
                  answers=answers("IL10"))
    S.score_candidate(c, CHAIN, ["S4"])
    assert c.category.value == "excluded"
    assert c.total is None


def test_moa_adjacency_uses_nearest_seed_stage():
    # TNFSF11 at S6, seeds at S1,S2,S4,S5 -> nearest is S5 -> adjacent 0.7
    m, opp = S.moa_score("S6", Direction.WORSE, ["S1", "S2", "S4", "S5"], CHAIN)
    assert m == 0.7 and not opp
    # S7 would be two away from S5 -> off
    m, _ = S.moa_score("S7", Direction.WORSE, ["S1", "S2", "S4", "S5"], CHAIN)
    assert m == 0.0


def test_probability_range_is_checked():
    with pytest.raises(ValueError):
        S.question_score({"Q1": 1.2, "Q2": 1, "Q3": 1, "Q4": 1, "Q5": 0})


def test_multi_evidence_bonus_off_by_default_and_capped():
    r = ScoringRules(multi_evidence_bonus=0.1)
    assert S.proximity_score(1.0, 1.0, 1.0, r) == 1.0          # capped
    assert S.proximity_score(0.7, 0.0, 0.0, r) == pytest.approx(0.7)   # single evidence: no bonus
    assert S.proximity_score(0.7, 0.5, 0.0, r) == pytest.approx(0.8)   # two evidences
    assert DEFAULT_RULES.multi_evidence_bonus == 0.0
