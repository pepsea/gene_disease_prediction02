"""データの入れ物（候補遺伝子カード、しくみの鎖、向き、分類）。

計算はしません。loop.py が候補を見つけて書き込み、scoring.py が点数を書き込み、
report.py が表にします。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Direction(str, Enum):
    """遺伝子の働きの「向き」。
    WORSE  = その遺伝子が働くと病気が悪くなる → 止めると良くなる（標的として望ましい向き）
    BETTER = その遺伝子は病気を抑えている → 止めると悪化する（別枠にする向き）
    """
    WORSE = "worse"      # gene activity worsens disease -> inhibition helps
    BETTER = "better"    # gene activity protects -> inhibition harms
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MoAStage:
    """作用のしくみの鎖の1段（例: S4「炎症性サイトカインとその受容体」）。index は鎖の中の位置。"""
    sid: str            # e.g. "S4"
    name: str           # e.g. "inflammatory cytokines and receptors"
    index: int          # position in the chain (0-based)
    direction: Direction = Direction.WORSE   # direction in which the stage drives disease


@dataclass
class MoAChain:
    """病気1つ分の「しくみの鎖」（引き金 → … → 症状）。M（しくみの近さ）はこの鎖の上の距離で測る。"""
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
    """最終的な4分類（＋別枠・根拠なし）。会話記録の「ステップ6」に対応。"""
    KNOWN = "known"                     # direct association verified (approved drug etc.)
    MOA_NEAR = "moa_near"               # same/adjacent stage, same direction
    MAP_NEAR = "map_near"               # near in the knowledge map only (siblings)
    COMBINATION = "combination"         # high question score but no proximity evidence
    EXCLUDED = "excluded"               # opposite direction or contradiction found
    UNSUPPORTED = "unsupported"         # nothing speaks for it


@dataclass
class Candidate:
    """候補遺伝子1つ分の記録カード。

    上半分（symbol〜is_seed）は「どう見つかり、AI に何と答えられたか」、
    下半分（q_score〜exclusion_reason）は scoring.score_candidate が書き込む採点結果です。
    how_found が「根拠の鎖」の先頭になります（例: "TNF -> direct partner -> TNFRSF1A"）。
    """
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
