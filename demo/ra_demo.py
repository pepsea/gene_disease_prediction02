"""関節リウマチ（RA）デモのデータ。会話記録の「非常に小さいグループでデモ」の転記。

中学生向けの説明:
    ここには「AI がこう答えた」という記録だけが入っています（計算はしません）。
    build_record() がこれを「質問文 → 答え」の辞書に変え、RecordedBackend が再生します。
    つまり、このファイルを差し替えれば別の病気・別の記録でも同じ手順を再現できます。

Rheumatoid-arthritis demo data, transcribed from the Claude-judged demo.

IMPORTANT CAVEATS (same as in the original conversation)
* The "LLM" that produced these answers was Claude, answering from memory
  with a 5-level confidence scale (1.0 / 0.75 / 0.5 / 0.25 / 0) instead of a
  real yes-probability.  Vector bins (near/mid/far) were judged, not computed.
* Claude knows the clinical outcome of several of these genes, so the numbers
  carry hindsight.  This record demonstrates the *procedure*; it is not a
  measurement of performance.
* Expansion lists were pruned to the 18 genes of the demo so that the replay
  reproduces the demo exactly.  Round-2 per-question answers were not printed
  in the transcript and were reconstructed to match the reported Q values
  (CSF2RA 0.95, IL6ST 0.80, MYD88 0.65).
"""
from __future__ import annotations
import json
from typing import Dict, List

from target_loop.models import MoAChain, MoAStage, Direction
from target_loop import config as C
from target_loop import loop as L
from target_loop.backends import RecordedBackend

DISEASE = "rheumatoid arthritis"
TISSUE = "synovium (joint)"

CHAIN = MoAChain(DISEASE, TISSUE, [
    MoAStage("S1", "antigen presentation and T-cell activation", 0),
    MoAStage("S2", "B cells and autoantibodies", 1),
    MoAStage("S3", "innate immune signalling", 2),
    MoAStage("S4", "inflammatory cytokines and their receptors", 3),
    MoAStage("S5", "intracellular signalling (JAK-STAT)", 4),
    MoAStage("S6", "synovial proliferation and osteoclast activation", 5),
    MoAStage("S7", "joint destruction (symptom)", 6),
])

# Seeds as first proposed by the LLM; CTLA4 is corrected to CD80 by the
# drug -> target reverse check (abatacept binds CD80/CD86).
SEEDS_PROPOSED = ["CTLA4", "MS4A1", "TNF", "IL6R", "JAK1"]
SEED_DRUG = {"CTLA4": "abatacept", "MS4A1": "rituximab", "TNF": "adalimumab", "IL6R": "tocilizumab",
             "JAK1": "upadacitinib", "CD80": "abatacept"}
DRUG_TARGET = {"abatacept": "CD80, CD86", "rituximab": "MS4A1", "adalimumab": "TNF",
               "tocilizumab": "IL6R", "upadacitinib": "JAK1"}

# gene -> (stage, direction, vector bin)
ANNOT: Dict[str, tuple] = {
    "CD80": ("S1", "WORSE", "near"), "MS4A1": ("S2", "WORSE", "near"), "TNF": ("S4", "WORSE", "near"),
    "IL6R": ("S4", "WORSE", "near"), "JAK1": ("S5", "WORSE", "near"),
    "IL6": ("S4", "WORSE", "near"), "TNFRSF1A": ("S4", "WORSE", "mid"), "TYK2": ("S5", "WORSE", "near"),
    "BTK": ("S2", "WORSE", "mid"), "CSF2": ("S4", "WORSE", "mid"), "IL17A": ("S4", "WORSE", "mid"),
    "IRAK4": ("S3", "WORSE", "mid"), "TNFSF11": ("S6", "WORSE", "far"), "IL11": ("NONE", "WORSE", "near"),
    "IL10": ("S4", "BETTER", "mid"),
    "CSF2RA": ("S4", "WORSE", "mid"), "IL6ST": ("S4", "WORSE", "near"), "MYD88": ("S3", "WORSE", "mid"),
}

# gene -> [Q1, Q2, Q3, Q4, Q5(raw: prob of serious adverse effects), Q6(genetics, optional)]
# Q6 values are Claude's judgement added *after* the demo (flagged, off by default).
ANSWERS: Dict[str, List[float]] = {
    "IL6":      [1.0, 1.0, 1.0, 1.0, 0.25, 0.75],
    "CSF2":     [1.0, 1.0, 1.0, 1.0, 0.25, 0.25],
    "TNFRSF1A": [1.0, 1.0, 1.0, 1.0, 0.50, 0.25],
    "TYK2":     [0.75, 1.0, 1.0, 1.0, 0.25, 1.0],
    "BTK":      [0.75, 1.0, 1.0, 1.0, 0.25, 0.25],
    "IL17A":    [0.75, 0.75, 1.0, 1.0, 0.25, 0.25],
    "TNFSF11":  [1.0, 1.0, 1.0, 1.0, 0.25, 0.25],
    "IRAK4":    [0.75, 1.0, 1.0, 1.0, 0.50, 0.25],
    "IL11":     [0.5, 0.5, 0.5, 1.0, 0.50, 0.0],
    "IL10":     [1.0, 1.0, 0.0, 1.0, 0.50, 0.25],
    "CSF2RA":   [1.0, 1.0, 1.0, 1.0, 0.25, 0.25],
    "IL6ST":    [1.0, 1.0, 1.0, 1.0, 1.00, 0.25],
    "MYD88":    [0.75, 1.0, 1.0, 0.25, 0.75, 0.0],
}

# expansion lists actually used (everything else answers NONE)
PARTNERS = {"TNF": "TNFRSF1A", "IL6R": "IL6", "JAK1": "TYK2", "CSF2": "CSF2RA", "IL6": "IL6ST"}
UPSTREAM = {"TNF": "IRAK4", "IRAK4": "MYD88"}
DOWNSTREAM = {"TNF": "TNFSF11"}
MAP_NEIGHBOURS = {"IL6R": "IL11"}
SAME_STAGE = {"MS4A1": "BTK", "TNF": "CSF2, IL17A, IL10"}

# reverse drug check: approved drug whose target is the gene, for RA
VERIFIED = {"IL6": (1.0, "olokizumab (approved in some countries, e.g. Russia)"),
            "TNFSF11": (1.0, "denosumab (Japan: inhibition of bone-erosion progression in RA)")}

PAIRWISE = {  # (left, right) -> answer as given
    ("CSF2", "BTK"): "CSF2", ("BTK", "CSF2"): "BTK",                 # order-dependent -> tie
    ("CSF2", "TNFRSF1A"): "TNFRSF1A", ("TNFRSF1A", "CSF2"): "TNFRSF1A",
    ("CSF2", "TYK2"): "CSF2", ("TYK2", "CSF2"): "CSF2",
    ("BTK", "TNFRSF1A"): "TNFRSF1A", ("TNFRSF1A", "BTK"): "TNFRSF1A",
    ("BTK", "TYK2"): "BTK", ("TYK2", "BTK"): "BTK",
    ("TNFRSF1A", "TYK2"): "TNFRSF1A", ("TYK2", "TNFRSF1A"): "TNFRSF1A",
}

CRITIC = {
    "TNFRSF1A": "Several TNF-blocking drugs already exist, so demonstrating added value over them is difficult. Receptor blockade may not be superior to ligand neutralisation. TNFR1 signalling also mediates host defence.",
    "CSF2": "Its role may be redundant with other cytokines, so blocking it alone may give a small effect. Phase 3 anti-GM-CSF antibodies have shown insufficient efficacy. Pulmonary alveolar proteinosis is a theoretical risk.",
    "BTK": "BTK is expressed in platelets as well as B cells, so bleeding risk exists. Oral BTK inhibitors have shown mixed efficacy in RA trials. Off-target kinase effects.",
}

ALL_GENES = sorted(set(ANNOT) | set(ANSWERS) | set(SEEDS_PROPOSED) | {"CD86"})


def build_record() -> Dict:
    """上の読みやすい表（ANSWERS, ANNOT, PARTNERS …）から、記録再生用の「質問文 → 答え」辞書を作る。

    質問文は loop.py と同じ雛形（config.py）から作るので、手順側が聞く文とキーが必ず一致します。
    Generate the exact prompt->answer table the loop will ask for."""
    dummy = L.TargetLoop(RecordedBackend({}), CHAIN)   # only used for .fmt
    fmt = dummy.fmt
    yes: Dict[str, float] = {}
    text: Dict[str, str] = {}

    for g in ALL_GENES:
        # yes/no questions, all paraphrases (same value each; order swap handled by backend)
        if g in ANSWERS:
            for q, val in zip(C.QUESTIONS, ANSWERS[g]):
                for t in (q.text,) + tuple(q.paraphrases):
                    yes[fmt(t, gene=g)] = val
        # annotations
        st, d, vb = ANNOT.get(g, ("NONE", "WORSE", "mid"))
        text[fmt(C.STAGE_PROMPT, gene=g)] = st
        text[fmt(C.DIRECTION_PROMPT, gene=g)] = d
        text[fmt(L.VECTOR_JUDGE_PROMPT, gene=g, seeds="CD80, MS4A1, TNF, IL6R, JAK1")] = vb.upper()
        text[fmt(C.DESCRIPTION_PROMPT, gene=g)] = f"{g}: description not recorded"
        # expansions
        text[fmt(C.DIRECT_PARTNERS_PROMPT, gene=g)] = PARTNERS.get(g, "NONE")
        text[fmt(C.UPSTREAM_PROMPT, gene=g)] = UPSTREAM.get(g, "NONE")
        text[fmt(C.DOWNSTREAM_PROMPT, gene=g)] = DOWNSTREAM.get(g, "NONE")
        text[fmt(L.MAP_NEIGHBOR_PROMPT, gene=g)] = MAP_NEIGHBOURS.get(g, "NONE")
        stage = CHAIN.by_id(st)
        if stage is not None:
            text[fmt(C.SAME_STAGE_PROMPT, gene=g, stage=stage.name, seeds=g)] = SAME_STAGE.get(g, "NONE")
            yes[fmt(L.REVERSE_STAGE_QUESTION, gene=g, stage=stage.name)] = 1.0
        # verification
        p, note = VERIFIED.get(g, (0.0, "NONE"))
        yes[fmt(C.REVERSE_DRUG_QUESTION, gene=g)] = p
        text[fmt(C.SEED_DRUG_CHECK, gene=g)] = SEED_DRUG.get(g, note)
        text[fmt(C.CRITIC_PROMPT, gene=g)] = CRITIC.get(g, "No specific critique recorded.")
    for drug, tgt in DRUG_TARGET.items():
        text[fmt(L.DRUG_TARGET_PROMPT, drug=drug)] = tgt
    for (a, b), ans in PAIRWISE.items():
        text[fmt(C.PAIRWISE_PROMPT, a=a, b=b)] = ans
    return {"yes": yes, "text": text, "meta": {"disease": DISEASE, "judge": "Claude (5-level confidence, hindsight-contaminated)"}}


class LenientPairwiseBackend(RecordedBackend):
    """For sensitivity runs: a pairwise duel that was not asked in the original
    demo answers '?' (counted as a tie) instead of raising.  Everything else
    still raises on an unrecorded prompt.  `unrecorded_duels` counts them."""
    def __init__(self, record):
        super().__init__(record)
        self.unrecorded_duels = 0

    def generate(self, prompt, max_tokens=128):
        if prompt not in self.text and prompt.startswith("Which gene is the more plausible drug target"):
            self.unrecorded_duels += 1
            return "?"
        return super().generate(prompt, max_tokens)


def run(rules=None, lenient_pairwise=False, **kw):
    """RA デモを記録再生で1回まわす。rules を差し替えれば感度分析に使える。"""
    from target_loop.config import DEFAULT_RULES
    cls = LenientPairwiseBackend if lenient_pairwise else RecordedBackend
    backend = cls(build_record())
    loop = L.TargetLoop(backend, CHAIN, rules or DEFAULT_RULES, **kw)
    return loop.run(SEEDS_PROPOSED), backend


if __name__ == "__main__":
    rec = build_record()
    with open("demo/ra_recorded.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print(f"wrote demo/ra_recorded.json: {len(rec['yes'])} yes/no answers, {len(rec['text'])} text answers")
