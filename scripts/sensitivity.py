"""固定ルールの定数を1つずつ変えて、順位がどれだけ変わるかを調べる（感度分析）。
結果は docs/sensitivity.md と検証レポート §3 に転記済み。

How robust is the ranking to the 'fixed' rule constants?  And which parts of
the design are decided by the rules alone, independent of the LLM?

Runs the RA replay under perturbed rules and reports:
  1. rank correlation / top-k overlap vs. the baseline rules
  2. tie structure of the 5-level confidence scale
  3. the damping / stop-rule interaction (an analytic bound, then checked on the replay)
  4. effect of the optional genetics question (Q6) and the 2-of-3 evidence bonus
"""
import itertools, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from collections import Counter
from target_loop.config import ScoringRules, DEFAULT_RULES
from target_loop.models import Category
from demo.ra_demo import run, ANSWERS


def ranking(rules):
    res, _ = run(rules, lenient_pairwise=True)
    return [c.symbol for c in res.ranked()], res


def kendall_tau(a, b):
    common = [x for x in a if x in b]
    pos_a = {x: i for i, x in enumerate(common)}
    pos_b = {x: i for i, x in enumerate(b) if x in pos_a}
    conc = disc = 0
    for x, y in itertools.combinations(common, 2):
        s = (pos_a[x] - pos_a[y]) * (pos_b[x] - pos_b[y])
        conc += s > 0
        disc += s < 0
    n = len(common)
    return (conc - disc) / (n * (n - 1) / 2) if n > 1 else 1.0


def main():
    base, base_res = ranking(DEFAULT_RULES)
    L = []
    L.append("# Sensitivity of the fixed rules (RA replay, 13 candidates)\n")
    L.append(f"baseline ranking (non-known, non-excluded): {' > '.join(base)}\n")

    # 1. perturbation grid ------------------------------------------------------
    L.append("## 1. Rule perturbation\n")
    L.append("| parameter | value | Kendall tau vs baseline | top-3 overlap | top-5 overlap | ranking |")
    L.append("|---|---|---|---|---|---|")
    grid = {
        "network_weight": [0.6, 0.7, 0.9, 1.0],
        "vector_weight": [0.4, 0.5, 0.7, 0.8],
        "moa_adjacent": [0.5, 0.6, 0.8, 0.9],
        "round_damping": [0.5, 0.6, 0.8, 0.9, 1.0],
        "multi_evidence_bonus": [0.1, 0.2],
    }
    taus = []
    for param, values in grid.items():
        for v in values:
            r, _ = ranking(replace(DEFAULT_RULES, **{param: v}))
            tau = kendall_tau(base, r)
            taus.append(tau)
            o3 = len(set(base[:3]) & set(r[:3]))
            o5 = len(set(base[:5]) & set(r[:5]))
            L.append(f"| {param} | {v} | {tau:.2f} | {o3}/3 | {o5}/5 | {' > '.join(r)} |")
    L.append(f"\nminimum Kendall tau over the grid: {min(taus):.2f}; mean {sum(taus)/len(taus):.2f}\n")

    # 2. ties ----------------------------------------------------------------
    L.append("## 2. Tie structure of the 5-level confidence scale\n")
    totals = Counter(round(c.total, 3) for c in base_res.ranked())
    ties = {k: v for k, v in totals.items() if v > 1}
    L.append(f"distinct totals among {len(base)} ranked candidates: {len(totals)}; tied groups: {ties}\n")
    # how many distinct Q values are even possible with 5 questions on a 5-level scale?
    possible_q = len({sum(x) / 5 for x in itertools.product([0, .25, .5, .75, 1], repeat=5)})
    L.append(f"With a 5-level scale and 5 questions Q can only take {possible_q} distinct values, "
             f"so ties are structural; pairwise comparison is needed to break them. "
             f"A real yes-probability (TxGemma logits) is continuous and removes this artefact.\n")

    # 3. damping vs stop rule ------------------------------------------------
    L.append("## 3. Damping and the stop rule are confounded\n")
    d = DEFAULT_RULES.round_damping
    r1_min_top5 = sorted((c.total for c in base_res.ranked() if c.round_found == 1), reverse=True)[4]
    L.append(f"A round-r candidate's total is at most Q*P*{d}^(r-1) <= {d}^(r-1). "
             f"In round 2 that is <= {d:.2f}; the 5th-best round-1 total is {r1_min_top5:.2f}. "
             f"So NO round-2 candidate can enter the top-5 while damping={d} and the round-1 top-5 "
             f"are all above {d}: the 'top-5 unchanged' stop fires in round 2 by construction, "
             f"whatever the LLM answers.  The loop therefore never really used round 2.\n")
    L.append("| round_damping | rounds run | stop reason | round-2 genes in top-5 |")
    L.append("|---|---|---|---|")
    for dv in (0.5, 0.7, 0.9, 1.0):
        r, res = ranking(replace(DEFAULT_RULES, round_damping=dv))
        r2 = [g for g in res.rounds[-1].top_set if res.candidates[g].round_found == 2]
        L.append(f"| {dv} | {len(res.rounds)} | {res.stop_reason} | {', '.join(r2) or '─'} |")
    L.append("\nWith damping >= 0.9 the round-2 candidate CSF2RA does enter the top-5 and the loop goes on "
             "to round 3 (which finds nothing new).  So whether the loop 'converges in 2 rounds' is decided by "
             "the damping constant, not by the LLM's answers.  Recommendation: apply damping to the proximity "
             "term only, or compare the top-k *within* each round, so that the stop rule measures the model.\n")

    # 4. genetics question and evidence bonus -----------------------------------
    L.append("## 4. Optional additions proposed after the demo\n")
    L.append("| variant | ranking |")
    L.append("|---|---|")
    L.append(f"| baseline (Q1-Q5) | {' > '.join(base)} |")
    r6, _ = ranking(replace(DEFAULT_RULES, use_genetics_question=True))
    L.append(f"| + Q6 human genetics (Claude-judged values, unverified) | {' > '.join(r6)} |")
    rb, _ = ranking(replace(DEFAULT_RULES, multi_evidence_bonus=0.15))
    L.append(f"| + 2-of-3 evidence bonus 0.15 | {' > '.join(rb)} |")
    r6b, _ = ranking(replace(DEFAULT_RULES, use_genetics_question=True, multi_evidence_bonus=0.15))
    L.append(f"| both | {' > '.join(r6b)} |")
    L.append("\nWith Q6, TYK2 (strong RA GWAS support) moves above CSF2 (no RA genetics; phase-3 failure). "
             "This is the direction the demo's own post-mortem predicted, but the Q6 values here are "
             "Claude's judgement with hindsight, so it is a consistency check of the design, not evidence.\n")

    out = "\n".join(L) + "\n"
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/sensitivity.md", "w", encoding="utf-8") as f:
        f.write(out)
    print(out)


if __name__ == "__main__":
    main()
