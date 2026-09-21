"""target_loop: training-free drug-target discovery loop driven by LLM knowledge.

Pipeline (fixed rules, no learned weights):
    seed -> expand (MoA / network / vector proximity) -> verify (repeated Q&A)
         -> classify (known / MoA-near / map-near / combination / excluded)

The scoring rules are frozen *before* any answer is read (see config.py).
"""
from .config import ScoringRules, DEFAULT_RULES, QUESTIONS
from .models import Candidate, MoAChain, MoAStage, Direction
from .scoring import question_score, proximity_score, total_score, score_candidate
from .pairwise import pairwise_rank
from .loop import run_loop, LoopResult

__all__ = [
    "ScoringRules", "DEFAULT_RULES", "QUESTIONS",
    "Candidate", "MoAChain", "MoAStage", "Direction",
    "question_score", "proximity_score", "total_score", "score_candidate",
    "pairwise_rank", "run_loop", "LoopResult",
]
