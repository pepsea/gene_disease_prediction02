"""Pairwise ("tournament") comparison with left/right swap and Copeland ranking."""
from __future__ import annotations
from itertools import combinations
from typing import Callable, Dict, List, Tuple

Compare = Callable[[str, str], str]   # (left, right) -> winner symbol (as answered by the LLM)


def duel(a: str, b: str, compare: Compare) -> Tuple[str, str, str]:
    """Ask twice with swapped order. Returns (first_answer, second_answer, verdict)
    where verdict is a symbol or "tie" (answers disagree or are unparsable)."""
    r1 = compare(a, b)
    r2 = compare(b, a)
    if r1 == r2 and r1 in (a, b):
        return r1, r2, r1
    return r1, r2, "tie"


def pairwise_rank(symbols: List[str], compare: Compare) -> Tuple[List[Tuple[str, float]], List[dict]]:
    """Copeland score: +1 per win, +0.5 per tie. Returns ranking and the duel log."""
    score: Dict[str, float] = {s: 0.0 for s in symbols}
    log: List[dict] = []
    for a, b in combinations(symbols, 2):
        r1, r2, verdict = duel(a, b, compare)
        if verdict == "tie":
            score[a] += 0.5
            score[b] += 0.5
        else:
            score[verdict] += 1.0
        log.append({"a": a, "b": b, "first": r1, "swapped": r2, "verdict": verdict})
    ranking = sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranking, log


def ranking_string(ranking: List[Tuple[str, float]]) -> str:
    """'A > B ≒ C > D' using the Copeland scores."""
    out = []
    prev = None
    for sym, sc in ranking:
        if prev is None:
            out.append(sym)
        elif sc == prev:
            out.append(f" ≒ {sym}")
        else:
            out.append(f" > {sym}")
        prev = sc
    return "".join(out)
