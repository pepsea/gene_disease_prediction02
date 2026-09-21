"""結果（LoopResult）を、会話記録と同じ形の Markdown 表にする。計算はしない。

Markdown report of a LoopResult (tables the way the demo presented them)."""
from __future__ import annotations
from typing import List
from .loop import LoopResult
from .models import Category
from .pairwise import ranking_string

CAT_JA = {
    Category.KNOWN: "既知（直接の関連が確認された）",
    Category.MOA_NEAR: "しくみ（MoA）が近い候補",
    Category.MAP_NEAR: "地図（ベクトル）が近い候補",
    Category.COMBINATION: "組み合わせの候補",
    Category.EXCLUDED: "別枠（逆向き・矛盾）",
    Category.UNSUPPORTED: "根拠なし",
}


def _f(x, nd=2):
    return "─" if x is None else f"{x:.{nd}f}"


def to_markdown(res: LoopResult, qids: List[str] = ("Q1", "Q2", "Q3", "Q4", "Q5")) -> str:
    """結果を Markdown の表にする。順に: 種と鎖 / 採点表 / 回ごとの対戦記録 / 4分類 / 反論役 / 根拠の鎖。"""
    L: List[str] = []
    L.append(f"# {res.chain.disease}: training-free target loop\n")
    L.append(f"seeds: {', '.join(res.seeds)}  ")
    for n in res.seed_notes:
        L.append(f"- seed note: {n}")
    L.append("")
    L.append("## Mechanism chain\n")
    L.append("| stage | content | seeds |")
    L.append("|---|---|---|")
    for st in res.chain.stages:
        ss = [s for s in res.seeds if res.candidates[s].stage == st.sid]
        L.append(f"| {st.sid} | {st.name} | {', '.join(ss) or '─'} |")
    L.append("")
    L.append("## Scores\n")
    hdr = "| gene | round | found via | stage | dir | M | N | V | " + " | ".join(qids) + " | Q | P | total | category |"
    L.append(hdr)
    L.append("|" + "---|" * (hdr.count("|") - 1))
    allc = sorted((c for c in res.candidates.values() if not c.is_seed), key=lambda c: (-(c.total or -1), c.symbol))
    for c in allc:
        qa = " | ".join(_f(c.answers.get(q)) for q in qids)
        L.append(f"| {c.symbol} | {c.round_found} | {c.how_found} | {c.stage or '─'} | {c.direction.value} | "
                 f"{_f(c.m_score,1)} | {_f(c.n_score,1)} | {_f(c.v_score,1)} | {qa} | {_f(c.q_score)} | {_f(c.p_score)} | "
                 f"**{_f(c.total)}** | {c.category.value}{(' – ' + c.exclusion_reason) if c.exclusion_reason else ''} |")
    L.append("")
    for r in res.rounds:
        L.append(f"## Round {r.round_no}\n")
        L.append(f"sources: {', '.join(r.sources)}  ")
        L.append(f"new candidates: {', '.join(r.new_candidates) or '(none)'}  ")
        if r.pairwise_log:
            L.append("\n| duel | first | swapped | verdict |")
            L.append("|---|---|---|---|")
            for d in r.pairwise_log:
                L.append(f"| {d['a']} vs {d['b']} | {d['first']} | {d['swapped']} | {d['verdict']} |")
            L.append(f"\npairwise ranking: **{ranking_string(r.pairwise_ranking)}**  ")
        L.append(f"top-{res.rules.stop_topk} for stop rule: {', '.join(r.top_set)}\n")
    L.append(f"stop reason: {res.stop_reason}\n")
    L.append("## Categories\n")
    L.append("| category | genes |")
    L.append("|---|---|")
    for cat, genes in res.by_category().items():
        if genes:
            L.append(f"| {CAT_JA[cat]} | {', '.join(genes)} |")
    L.append("")
    L.append("## Critic notes (top 3)\n")
    for c in allc:
        if c.critique:
            L.append(f"- **{c.symbol}**: " + " / ".join(c.critique))
    L.append("")
    L.append("## Evidence chains\n")
    for c in allc:
        if c.category in (Category.MOA_NEAR, Category.MAP_NEAR, Category.COMBINATION):
            L.append(f"- {c.symbol}: {c.how_found}; stage {c.stage or '─'} ({c.direction.value}); "
                     f"verification: {c.verification_note.strip() or 'no approved drug for this indication'}")
    return "\n".join(L) + "\n"
