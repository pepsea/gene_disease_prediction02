from target_loop.pairwise import pairwise_rank, duel, ranking_string
from demo.ra_demo import PAIRWISE


def cmp(a, b):
    return PAIRWISE[(a, b)]


def test_order_dependent_answer_becomes_tie():
    assert duel("CSF2", "BTK", cmp)[2] == "tie"
    assert duel("CSF2", "TYK2", cmp)[2] == "CSF2"


def test_demo_ranking_reproduced():
    ranking, log = pairwise_rank(["CSF2", "BTK", "TNFRSF1A", "TYK2"], cmp)
    assert ranking_string(ranking) == "TNFRSF1A > BTK ≒ CSF2 > TYK2"
    assert dict(ranking) == {"TNFRSF1A": 3.0, "BTK": 1.5, "CSF2": 1.5, "TYK2": 0.0}
    assert len(log) == 6                       # 6 duels = 12 questions


def test_unparsable_answer_is_tie():
    ranking, _ = pairwise_rank(["A", "B"], lambda a, b: "?")
    assert dict(ranking) == {"A": 0.5, "B": 0.5}
