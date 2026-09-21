"""The training-free loop:  seed -> expand -> verify -> classify, repeated.

Every LLM interaction goes through `backend`; the loop itself contains no
learned parameter.  All prompts are built from the templates in config.py so
that a RecordedBackend can replay a run and a LlamaCppBackend can execute it.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import config as C
from .config import ScoringRules, DEFAULT_RULES, QUESTION_BY_ID, DEMO_QIDS
from .models import Candidate, MoAChain, Direction, Category
from .backends import LLMBackend, parse_gene_list, cosine
from .scoring import score_candidate, vector_bin_from_cos
from .pairwise import pairwise_rank, ranking_string

MAP_NEIGHBOR_PROMPT = "List the human genes whose biological function is most similar to {gene} (paralogs, same family, same complex). Answer with official gene symbols separated by commas only."
VECTOR_JUDGE_PROMPT = "How similar is the biological function of {gene} to the validated {disease} targets {seeds}? Answer NEAR, MID or FAR only."
REVERSE_STAGE_QUESTION = "In the pathogenesis of {disease}, is the stage '{stage}' the step at which {gene} acts?"
DRUG_TARGET_PROMPT = "What is the molecular target (human gene symbol) of the drug {drug}? Answer with official gene symbols separated by commas only."

STAGE_RE = re.compile(r"\bS\d+\b")


@dataclass
class RoundResult:
    round_no: int
    sources: List[str]
    new_candidates: List[str]
    pairwise_ranking: List[Tuple[str, float]] = field(default_factory=list)
    pairwise_log: List[dict] = field(default_factory=list)
    top_set: List[str] = field(default_factory=list)


@dataclass
class LoopResult:
    chain: MoAChain
    seeds: List[str]
    seed_notes: List[str]
    candidates: Dict[str, Candidate]
    rounds: List[RoundResult]
    stop_reason: str
    rules: ScoringRules

    def ranked(self, include_known: bool = False) -> List[Candidate]:
        cs = [c for c in self.candidates.values() if c.total is not None and not c.is_seed]
        if not include_known:
            cs = [c for c in cs if c.category != Category.KNOWN]
        return sorted(cs, key=lambda c: (-c.total, c.symbol))

    def by_category(self) -> Dict[Category, List[str]]:
        out: Dict[Category, List[str]] = {k: [] for k in Category}
        for c in sorted(self.candidates.values(), key=lambda c: (-(c.total or 0), c.symbol)):
            out[c.category].append(c.symbol)
        return out


# ----------------------------------------------------------------------------
class TargetLoop:
    def __init__(self, backend: LLMBackend, chain: MoAChain, rules: ScoringRules = DEFAULT_RULES,
                 n_paraphrases: int = 3, swap_order: bool = True, known_symbols: Optional[Set[str]] = None,
                 vector_pool: Optional[Sequence[str]] = None, qids: Sequence[str] = DEMO_QIDS,
                 max_new_per_source: int = 8, log: Optional[List[str]] = None):
        self.b = backend
        self.chain = chain
        self.rules = rules
        self.n_paraphrases = n_paraphrases
        self.swap_order = swap_order
        self.known_symbols = known_symbols
        self.vector_pool = list(vector_pool or [])
        self.qids = list(qids)
        if rules.use_genetics_question and "Q6" not in self.qids:
            self.qids.append("Q6")
        self.max_new_per_source = max_new_per_source
        self.log = log if log is not None else []
        self._desc_vec: Dict[str, Optional[List[float]]] = {}

    # ---- helpers -----------------------------------------------------------
    @property
    def disease(self) -> str:
        return self.chain.disease

    def fmt(self, template: str, **kw) -> str:
        return template.format(disease=self.disease, tissue=self.chain.tissue, chain=self.chain.as_text(), **kw)

    def ask(self, qid: str, gene: str) -> float:
        """Average p_yes over paraphrases x option orders."""
        q = QUESTION_BY_ID[qid]
        templates = (q.text,) + tuple(q.paraphrases)
        templates = templates[: max(1, self.n_paraphrases)]
        orders = ("yes_first", "no_first") if self.swap_order else ("yes_first",)
        vals = []
        for t in templates:
            for o in orders:
                vals.append(self.b.yes_probability(self.fmt(t, gene=gene), o))
        return sum(vals) / len(vals)

    def gen_list(self, template: str, exclude: Iterable[str] = (), **kw) -> List[str]:
        txt = self.b.generate(self.fmt(template, **kw))
        return parse_gene_list(txt, self.known_symbols, exclude=list(exclude))[: self.max_new_per_source]

    def stage_of(self, gene: str) -> Optional[str]:
        txt = self.b.generate(self.fmt(C.STAGE_PROMPT, gene=gene), max_tokens=8).upper()
        m = STAGE_RE.search(txt)
        if m and self.chain.by_id(m.group(0)) is not None:
            return m.group(0)
        return None

    def direction_of(self, gene: str) -> Direction:
        txt = self.b.generate(self.fmt(C.DIRECTION_PROMPT, gene=gene), max_tokens=8).upper()
        if "WORSE" in txt:
            return Direction.WORSE
        if "BETTER" in txt:
            return Direction.BETTER
        return Direction.UNKNOWN

    def description_vector(self, gene: str) -> Optional[List[float]]:
        if gene not in self._desc_vec:
            desc = self.b.generate(self.fmt(C.DESCRIPTION_PROMPT, gene=gene), max_tokens=160)
            self._desc_vec[gene] = self.b.embed(desc)
        return self._desc_vec[gene]

    def vector_bin(self, gene: str, seeds: Sequence[str]) -> Tuple[Optional[str], Optional[float]]:
        v = self.description_vector(gene)
        if v is not None:
            best = -1.0
            for s in seeds:
                sv = self.description_vector(s)
                if sv is not None:
                    best = max(best, cosine(v, sv))
            if best >= -1.0 and best != -1.0:
                return vector_bin_from_cos(best, self.rules), best
        # fall back to a judged bin (what the Claude-judged demo did)
        txt = self.b.generate(self.fmt(VECTOR_JUDGE_PROMPT, gene=gene, seeds=", ".join(seeds)), max_tokens=4).upper()
        for k in ("NEAR", "MID", "FAR"):
            if k in txt:
                return k.lower(), None
        return None, None

    # ---- seeds -------------------------------------------------------------
    def build_seeds(self, seeds: Optional[Sequence[str]] = None, n_votes: int = 5, min_votes: int = 4) -> Tuple[List[str], List[str]]:
        """Seeds from LLM knowledge (majority vote over paraphrased asks),
        then a drug-name check and a reverse target check (drug -> target)."""
        notes: List[str] = []
        if seeds is None:
            votes: Dict[str, int] = {}
            for i in range(n_votes):
                lst = self.gen_list(C.SEED_PROMPT + f" (attempt {i+1})")
                for g in lst:
                    votes[g] = votes.get(g, 0) + 1
            seeds = [g for g, n in votes.items() if n >= min_votes]
            notes.append(f"seed vote: {votes}")
        final: List[str] = []
        for g in seeds:
            drug = self.b.generate(self.fmt(C.SEED_DRUG_CHECK, gene=g), max_tokens=16).strip()
            if not drug or drug.upper().startswith("NONE"):
                notes.append(f"{g}: dropped, no approved drug named")
                continue
            targets = self.gen_list(DRUG_TARGET_PROMPT, drug=drug)
            if targets and g not in targets:
                notes.append(f"{g}: corrected to {targets[0]} (drug {drug} binds {', '.join(targets)})")
                g = targets[0]
            if g not in final:
                final.append(g)
        return final, notes

    # ---- one round ---------------------------------------------------------
    def expand(self, sources: Sequence[str], cands: Dict[str, Candidate], round_no: int,
               known: Set[str]) -> List[Candidate]:
        new: List[Candidate] = []
        existing = set(cands) | known
        for src in sources:
            src_c = cands.get(src)
            src_hops = 0 if src in known else (src_c.network_hops if src_c and src_c.network_hops else 2)
            plans = [
                (C.DIRECT_PARTNERS_PROMPT, "direct partner", src_hops + 1, {}),
                (C.UPSTREAM_PROMPT, "upstream regulator", src_hops + 2, {}),
                (C.DOWNSTREAM_PROMPT, "downstream effector", src_hops + 2, {}),
                (MAP_NEIGHBOR_PROMPT, "map neighbour (same family)", src_hops + 2, {}),
            ]
            src_stage = self.chain.by_id(src_c.stage if src_c else None)
            if src_stage is not None:
                plans.append((C.SAME_STAGE_PROMPT, f"same stage {src_stage.sid}", src_hops + 2,
                              {"stage": src_stage.name, "seeds": src}))
            for template, how, hops, kw in plans:
                for g in self.gen_list(template, exclude=existing, gene=src, **kw):
                    if g in existing:
                        continue
                    c = Candidate(symbol=g, round_found=round_no, parent=src,
                                  how_found=f"{src} -> {how} -> {g}", network_hops=hops)
                    cands[g] = c
                    existing.add(g)
                    new.append(c)
        # vector-pool expansion (only when embeddings are available)
        if self.vector_pool:
            seeds = [s for s in sources if s in known] or list(sources)
            for g in self.vector_pool:
                if g in existing:
                    continue
                vb, cos = self.vector_bin(g, seeds)
                if vb == "near":
                    c = Candidate(symbol=g, round_found=round_no, parent=None,
                                  how_found=f"knowledge map near {seeds} (cos={cos:.2f})", network_hops=None,
                                  vector_bin=vb, vector_cos=cos)
                    cands[g] = c
                    existing.add(g)
                    new.append(c)
        return new

    def annotate(self, c: Candidate, seeds: Sequence[str]) -> None:
        c.stage = self.stage_of(c.symbol)
        c.direction = self.direction_of(c.symbol)
        if c.vector_bin is None:
            c.vector_bin, c.vector_cos = self.vector_bin(c.symbol, seeds)
        for qid in self.qids:
            c.answers[qid] = self.ask(qid, c.symbol)

    def verify(self, c: Candidate) -> None:
        """Reverse question: is there an approved drug with this target? -> known."""
        p = self.b.yes_probability(self.fmt(C.REVERSE_DRUG_QUESTION, gene=c.symbol))
        c.verified_known = p >= 0.5
        if c.verified_known:
            c.verification_note = self.b.generate(self.fmt(C.SEED_DRUG_CHECK, gene=c.symbol), max_tokens=32)
        # consistency check (reverse direction): the stage claimed in the forward
        # question (gene -> stage) must be confirmed when asked backwards
        # (stage -> gene).  A claim that does not survive the reverse question is a
        # contradiction and the candidate is set aside.
        if c.stage is not None:
            st = self.chain.by_id(c.stage)
            p_rev = self.b.yes_probability(self.fmt(REVERSE_STAGE_QUESTION, gene=c.symbol, stage=st.name))
            if p_rev < 0.5:
                c.contradiction = True
                c.verification_note += f" reverse stage check failed (p={p_rev:.2f})"

    def critic(self, c: Candidate) -> None:
        txt = self.b.generate(self.fmt(C.CRITIC_PROMPT, gene=c.symbol), max_tokens=160)
        c.critique = [s.strip(" -*") for s in re.split(r"\n+|(?<=\.)\s+", txt) if s.strip(" -*")][:3]

    def compare(self, a: str, b: str) -> str:
        txt = self.b.generate(self.fmt(C.PAIRWISE_PROMPT, a=a, b=b), max_tokens=8).upper()
        syms = parse_gene_list(txt)
        for s in syms:
            if s in (a, b):
                return s
        return "?"

    # ---- driver ------------------------------------------------------------
    def run(self, seeds: Optional[Sequence[str]] = None) -> LoopResult:
        seeds, seed_notes = self.build_seeds(seeds)
        for n in seed_notes:
            self.log.append(n)
        cands: Dict[str, Candidate] = {}
        for s in seeds:
            c = Candidate(symbol=s, round_found=0, is_seed=True, how_found="seed (validated drug target)",
                          network_hops=0, vector_bin="near", verified_known=True)
            c.stage = self.stage_of(s)
            c.direction = self.direction_of(s)
            cands[s] = c
        seed_stages = [cands[s].stage for s in seeds if cands[s].stage]
        known: Set[str] = set(seeds)

        rounds: List[RoundResult] = []
        sources: List[str] = list(seeds)
        prev_top: Optional[List[str]] = None
        stop_reason = f"max_rounds={self.rules.max_rounds} reached"
        for r in range(1, self.rules.max_rounds + 1):
            new = self.expand(sources, cands, r, known)
            self.log.append(f"round {r}: sources={sources} new={[c.symbol for c in new]}")
            for c in new:
                self.annotate(c, seeds)
                self.verify(c)
            # score everything (seeds included, for the report)
            for c in cands.values():
                if c.is_seed:
                    for qid in self.qids:
                        c.answers.setdefault(qid, 1.0)
                score_candidate(c, self.chain, seed_stages, self.rules, self.qids)
                if c.verified_known and not c.is_seed:
                    known.add(c.symbol)
            ranked = [c for c in cands.values() if c.total is not None and not c.is_seed
                      and c.category not in (Category.KNOWN, Category.EXCLUDED)]
            ranked.sort(key=lambda c: (-c.total, c.symbol))
            top = [c.symbol for c in ranked[: self.rules.pairwise_topk]]
            ranking, plog = pairwise_rank(top, self.compare) if len(top) >= 2 else ([], [])
            for sym, _ in ranking[:3]:
                self.critic(cands[sym])
            top_set = [c.symbol for c in ranked[: self.rules.stop_topk]]
            rounds.append(RoundResult(r, list(sources), [c.symbol for c in new], ranking, plog, top_set))
            self.log.append(f"round {r}: pairwise {ranking_string(ranking)}; top{self.rules.stop_topk}={top_set}")
            if not new:
                stop_reason = f"round {r}: no new candidates"
                break
            if prev_top is not None and set(top_set) == set(prev_top):
                stop_reason = f"round {r}: top-{self.rules.stop_topk} unchanged"
                break
            prev_top = top_set
            # next sources: candidates of this round that passed (not excluded), plus verified known
            sources = [c.symbol for c in new if c.category != Category.EXCLUDED]
            if not sources:
                stop_reason = f"round {r}: nothing passed verification"
                break
        return LoopResult(self.chain, seeds, seed_notes, cands, rounds, stop_reason, self.rules)


def run_loop(backend: LLMBackend, chain: MoAChain, seeds: Optional[Sequence[str]] = None,
             rules: ScoringRules = DEFAULT_RULES, **kw) -> LoopResult:
    return TargetLoop(backend, chain, rules, **kw).run(seeds)
