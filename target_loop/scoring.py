"""Fixed-rule scoring (no training).

    Q = mean over questions of p_yes  (side-effect question inverted: 1 - p)
    M = MoA proximity        {1.0 same stage/direction, 0.7 adjacent, 0 off-chain}
    N = network proximity    {1.0 direct, 0.5 two-hop, 0 far}
    V = vector proximity     {1.0 near, 0.5 mid, 0 far}
    P = max(M, 0.8*N, 0.6*V)             (+ optional multi-evidence bonus)
    total = Q * P * damping**(round-1)
    excluded if direction is opposite or a contradiction was found.
"""
from __future__ import annotations
from typing import Dict, Iterable, Optional, Tuple

from .config import ScoringRules, DEFAULT_RULES, QUESTION_BY_ID, DEMO_QIDS
from .models import Candidate, MoAChain, Direction, Category


def question_score(answers: Dict[str, float], qids: Iterable[str] = DEMO_QIDS,
                   rules: ScoringRules = DEFAULT_RULES) -> float:
    """Mean of the (possibly inverted) yes-probabilities over the chosen questions."""
    qids = list(qids)
    if rules.use_genetics_question and "Q6" not in qids:
        qids.append("Q6")
    vals = []
    for qid in qids:
        if qid not in answers:
            raise KeyError(f"missing answer for {qid}")
        p = float(answers[qid])
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"{qid}: probability out of range: {p}")
        vals.append(1.0 - p if QUESTION_BY_ID[qid].invert else p)
    return sum(vals) / len(vals)


def moa_score(cand_stage: Optional[str], cand_dir: Direction, seed_stages: Iterable[str],
              chain: MoAChain, rules: ScoringRules = DEFAULT_RULES) -> Tuple[float, bool]:
    """Return (M, opposite_direction_flag).

    A candidate on the chain whose direction is opposite to the stage direction
    (e.g. an anti-inflammatory cytokine in an inflammatory stage) is flagged; the
    caller excludes it regardless of score.
    """
    st = chain.by_id(cand_stage)
    if st is None:
        return rules.moa_off_chain, False
    if cand_dir != Direction.UNKNOWN and cand_dir != st.direction:
        return 0.0, True
    seed_idx = {chain.by_id(s).index for s in seed_stages if chain.by_id(s) is not None}
    if not seed_idx:
        return rules.moa_off_chain, False
    dist = min(abs(st.index - i) for i in seed_idx)
    if dist == 0:
        return rules.moa_same_stage, False
    if dist == 1:
        return rules.moa_adjacent, False
    return rules.moa_off_chain, False


def network_score(hops: Optional[int], rules: ScoringRules = DEFAULT_RULES) -> float:
    if hops is None:
        return rules.network_far
    if hops <= 1:
        return rules.network_direct
    if hops == 2:
        return rules.network_two_hop
    return rules.network_far


def vector_score(vbin: Optional[str], rules: ScoringRules = DEFAULT_RULES) -> float:
    return {"near": rules.vector_near, "mid": rules.vector_mid, "far": rules.vector_far, None: 0.0}[vbin]


def vector_bin_from_cos(cos: float, rules: ScoringRules = DEFAULT_RULES) -> str:
    if cos >= rules.vector_near_cos:
        return "near"
    if cos >= rules.vector_mid_cos:
        return "mid"
    return "far"


def proximity_score(m: float, n: float, v: float, rules: ScoringRules = DEFAULT_RULES) -> float:
    parts = (m, rules.network_weight * n, rules.vector_weight * v)
    p = max(parts)
    if rules.multi_evidence_bonus > 0:
        n_evidence = sum(1 for x in (m, n, v) if x > 0)
        if n_evidence >= 2:
            p = min(1.0, p + rules.multi_evidence_bonus)
    return p


def total_score(q: float, p: float, round_found: int, rules: ScoringRules = DEFAULT_RULES) -> float:
    damp = rules.round_damping ** max(0, round_found - 1)
    return q * p * damp


def score_candidate(c: Candidate, chain: MoAChain, seed_stages: Iterable[str],
                    rules: ScoringRules = DEFAULT_RULES, qids: Iterable[str] = DEMO_QIDS) -> Candidate:
    """Fill the score fields of a candidate in place and return it."""
    seed_stages = list(seed_stages)
    m, opposite = moa_score(c.stage, c.direction, seed_stages, chain, rules)
    c.m_score = m
    c.n_score = network_score(c.network_hops, rules)
    c.v_score = vector_score(c.vector_bin, rules)
    c.p_score = proximity_score(c.m_score, c.n_score, c.v_score, rules)

    if opposite:
        c.exclusion_reason = "opposite direction (inhibition expected to worsen disease)"
    elif c.contradiction:
        c.exclusion_reason = "contradiction in verification questions"

    if c.exclusion_reason:
        c.q_score = None
        c.total = None
        c.category = Category.EXCLUDED
        return c

    c.q_score = question_score(c.answers, qids, rules)
    c.total = total_score(c.q_score, c.p_score, c.round_found, rules)

    if c.is_seed or c.verified_known:
        c.category = Category.KNOWN
    elif c.m_score > 0:
        c.category = Category.MOA_NEAR
    elif c.v_score >= rules.vector_near:
        c.category = Category.MAP_NEAR
    elif c.q_score >= 0.75:
        c.category = Category.COMBINATION
    else:
        c.category = Category.UNSUPPORTED
    return c
