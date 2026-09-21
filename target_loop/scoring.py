"""固定ルールによる採点（学習なし）。会話記録の「ステップ0」の式そのもの。

中学生向けの説明:
    通知表にたとえると、Q は「質問への答えの平均点」、P は「既知の標的とのご近所度」、
    総合点はその掛け算です。ご近所度が 0（何のつながりもない）なら、答えが良くても 0 点になります。

Fixed-rule scoring (no training).

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


# ---- Q：質問の点 -------------------------------------------------------------
def question_score(answers: Dict[str, float], qids: Iterable[str] = DEMO_QIDS,
                   rules: ScoringRules = DEFAULT_RULES) -> float:
    """Q（質問の点）＝ 各質問の「はい」確率の平均。副作用の質問は 1−p に反転。

    例: Q1〜Q4 が全部 1.0、Q5（副作用）が 0.25 なら (1+1+1+1+0.75)/5 = 0.95。
    確率が 0〜1 の外なら例外にして、単位の取り違え（%など）を早めに見つけます。
    """
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


# ---- M：しくみ（MoA）の近さ ----------------------------------------------------
def moa_score(cand_stage: Optional[str], cand_dir: Direction, seed_stages: Iterable[str],
              chain: MoAChain, rules: ScoringRules = DEFAULT_RULES) -> Tuple[float, bool]:
    """M（しくみの近さ）と「逆向きフラグ」を返す。

    候補が鎖のどの段階にいるかを、種がいる段階と比べます。
      同じ段階（距離0）  → 1.0    隣の段階（距離1） → 0.7    それ以外・鎖に乗らない → 0
    段階の向きと候補の向きが逆（例: 炎症を抑える IL10）なら、点数に関係なく別枠にするための
    フラグを立てます。

    Return (M, opposite_direction_flag).

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


# ---- N：関係の網の近さ -----------------------------------------------------------
def network_score(hops: Optional[int], rules: ScoringRules = DEFAULT_RULES) -> float:
    """N（網の近さ）。hops は「既知の遺伝子」からの段数。1段＝直接の相手 1.0、2段＝0.5、それ以上 0。
    起点を「既知（種＋検証で既知になった遺伝子）」と決めたのは検証レポート §3.2 の判断です。"""
    if hops is None:
        return rules.network_far
    if hops <= 1:
        return rules.network_direct
    if hops == 2:
        return rules.network_two_hop
    return rules.network_far


# ---- V：地図（ベクトル）の近さ ------------------------------------------------------
def vector_score(vbin: Optional[str], rules: ScoringRules = DEFAULT_RULES) -> float:
    """V（地図の近さ）。near=1.0 / mid=0.5 / far=0。埋め込みがあれば vector_bin_from_cos で近・中・遠に分けます。"""
    return {"near": rules.vector_near, "mid": rules.vector_mid, "far": rules.vector_far, None: 0.0}[vbin]


def vector_bin_from_cos(cos: float, rules: ScoringRules = DEFAULT_RULES) -> str:
    """コサイン類似度（説明文どうしの角度の近さ）を near / mid / far に丸める。閾値は rules にある。"""
    if cos >= rules.vector_near_cos:
        return "near"
    if cos >= rules.vector_mid_cos:
        return "mid"
    return "far"


# ---- P：3種類の近さをまとめる --------------------------------------------------------
def proximity_score(m: float, n: float, v: float, rules: ScoringRules = DEFAULT_RULES) -> float:
    """P ＝ max(M, 0.8×N, 0.6×V)。3つのうち一番強い根拠だけを採用する（デモのステップ0のルール）。

    multi_evidence_bonus > 0 のときだけ、「2種類以上で近い」候補に加点します（設計4章の案、既定は0）。
    """
    parts = (m, rules.network_weight * n, rules.vector_weight * v)
    p = max(parts)
    if rules.multi_evidence_bonus > 0:
        n_evidence = sum(1 for x in (m, n, v) if x > 0)
        if n_evidence >= 2:
            p = min(1.0, p + rules.multi_evidence_bonus)
    return p


# ---- 総合点と分類 ---------------------------------------------------------------------
def total_score(q: float, p: float, round_found: int, rules: ScoringRules = DEFAULT_RULES) -> float:
    """総合点 ＝ Q × P × 減衰^(ループ回数−1)。2回目に見つかった候補は 0.7 倍、3回目は 0.49 倍。"""
    damp = rules.round_damping ** max(0, round_found - 1)
    return q * p * damp


def score_candidate(c: Candidate, chain: MoAChain, seed_stages: Iterable[str],
                    rules: ScoringRules = DEFAULT_RULES, qids: Iterable[str] = DEMO_QIDS) -> Candidate:
    """候補カード1枚を採点して分類を書き込む（その場で書き換えて返す）。

    手順: M と逆向きフラグ → N → V → P → （別枠なら終了）→ Q → 総合点 → 分類。
    分類の優先順: 種または検証で既知 → M>0 なら「しくみが近い」→ V が near なら「地図が近い」
                  → Q≥0.75 なら「組み合わせ」→ それ以外は「根拠なし」。
    """
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
