from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Direction(str, Enum):
    WORSE = "worse"      # gene activity worsens disease -> inhibition helps
    BETTER = "better"    # gene activity protects -> inhibition harms
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MoAStage:
    sid: str            # e.g. "S4"
    name: str           # e.g. "inflammatory cytokines and receptors"
    index: int          # position in the chain (0-based)
    direction: Direction = Direction.WORSE   # direction in which the stage drives disease


@dataclass
class MoAChain:
    disease: str
    tissue: str
    stages: List[MoAStage]

    def by_id(self, sid: Optional[str]) -> Optional[MoAStage]:
        if sid is None:
            return None
        for s in self.stages:
            if s.sid == sid:
                return s
        return None

    def as_text(self) -> str:
        return "\n".join(f"{s.sid}: {s.name}" for s in self.stages)


class Category(str, Enum):
    KNOWN = "known"                     # direct association verified (approved drug etc.)
    MOA_NEAR = "moa_near"               # same/adjacent stage, same direction
    MAP_NEAR = "map_near"               # near in the knowledge map only (siblings)
    COMBINATION = "combination"         # high question score but no proximity evidence
    EXCLUDED = "excluded"               # opposite direction or contradiction found
    UNSUPPORTED = "unsupported"         # nothing speaks for it


@dataclass
class Candidate:
    symbol: str
    round_found: int = 1                   # 0 = seed
    how_found: str = ""                    # evidence chain in words
    parent: Optional[str] = None           # gene it was expanded from
    stage: Optional[str] = None            # MoA stage id or None (off chain)
    direction: Direction = Direction.UNKNOWN
    network_hops: Optional[int] = None     # hops from a *known* gene (1 = direct)
    vector_bin: Optional[str] = None       # "near" | "mid" | "far"
    vector_cos: Optional[float] = None
    answers: Dict[str, float] = field(default_factory=dict)   # qid -> p_yes (already averaged over paraphrases)
    verified_known: Optional[bool] = None  # reverse-question: approved drug exists?
    verification_note: str = ""
    contradiction: bool = False
    critique: List[str] = field(default_factory=list)
    is_seed: bool = False
    # filled by scoring
    q_score: Optional[float] = None
    m_score: Optional[float] = None
    n_score: Optional[float] = None
    v_score: Optional[float] = None
    p_score: Optional[float] = None
    total: Optional[float] = None
    category: Optional[Category] = None
    exclusion_reason: str = ""
