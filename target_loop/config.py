"""固定ルールと質問文（このファイルは「先に決めて、後で変えない」もの）。

中学生向けの説明:
    このループは「学習」をしません。代わりに、採点のルール（定数）と質問文を
    ここに全部書いておき、結果を見る前に固定します。
    ルールを結果に合わせて何度も直すと、それは人間が学習しているのと同じになるからです。

中身:
    ScoringRules  定数の箱（M/N/V の重み、減衰、停止条件、任意の追加ルール）
    QUESTIONS     候補遺伝子に投げる「はい／いいえ」質問（Q1〜Q5、任意の Q6）
    *_PROMPT      種の検証・候補の拡張・反論役・対戦比較に使う文章生成の雛形

Frozen scoring rules and the question set.

These numbers are the ones written down in "Step 0" of the demo *before* any
gene was scored.  They are deliberately not learned from data; change them only
before an evaluation run, never after looking at results (otherwise a human is
doing the "training" by hand).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class ScoringRules:
    """点数のルール（定数）をひとまとめにした箱。

    frozen=True なので、作った後に値を書き換えられません。
    「結果を見てから定数をいじる」ことを防ぐための仕組みです。
    変えたいときは dataclasses.replace(DEFAULT_RULES, round_damping=0.9) のように
    新しい箱を作り、評価の前に決めて使います。
    """
    # --- MoA (mechanism-of-action chain) proximity M ---
    moa_same_stage: float = 1.0      # same stage, same direction as a seed
    moa_adjacent: float = 0.7        # neighbouring stage, direction not contradicting
    moa_off_chain: float = 0.0       # not on the chain
    # --- network proximity N (hops from a *known* gene) ---
    network_direct: float = 1.0      # direct binding partner / receptor-ligand
    network_two_hop: float = 0.5     # partner of a partner
    network_far: float = 0.0
    network_weight: float = 0.8      # N is multiplied by this before max()
    # --- vector (knowledge-map) proximity V ---
    vector_near: float = 1.0
    vector_mid: float = 0.5
    vector_far: float = 0.0
    vector_weight: float = 0.6
    # cosine thresholds used only when V is computed from real embeddings
    vector_near_cos: float = 0.80
    vector_mid_cos: float = 0.65
    # --- loop control ---
    round_damping: float = 0.7       # total of a round-r candidate is multiplied by damping**(r-1)
    max_rounds: int = 3
    stop_topk: int = 5               # stop when the top-k (non-known) set does not change
    pairwise_topk: int = 4           # how many top non-known candidates enter pairwise comparison
    # --- optional rule from section 4 of the design ("2 of 3 proximities") ---
    # The demo's Step-0 rules used max(M, 0.8N, 0.6V) only; this bonus is OFF by
    # default so that the demo numbers are reproduced exactly.
    multi_evidence_bonus: float = 0.0
    # --- optional question on human genetics (proposed after the demo) ---
    use_genetics_question: bool = False


DEFAULT_RULES = ScoringRules()


@dataclass(frozen=True)
class Question:
    """AI に投げる「はい／いいえ」質問の雛形。

    text と paraphrases（言い換え）に {gene} {disease} {tissue} の穴があり、
    実行時に遺伝子名・病名・組織名が入ります。
    invert=True の質問（副作用）は、答えを 1−p にして「副作用が少ないほど高得点」にします。
    """
    qid: str
    text: str                 # {gene} and {disease} / {tissue} are filled in
    paraphrases: tuple        # alternative wordings (each also a template)
    invert: bool = False      # True -> use (1 - p_yes)  (e.g. side-effect question)


# デモの5つの質問（Q1 組織で働くか / Q2 鎖の段階にいるか / Q3 止めると良くなるか /
# Q4 薬が結合しやすい種類か / Q5 重い副作用があるか）と、任意の Q6（ヒト遺伝学の証拠）。
# The five questions from the demo, plus the optional genetics question (Q6).
QUESTIONS: List[Question] = [
    Question("Q1", "Is the gene {gene} expressed or active in the affected tissue ({tissue}) of {disease}?",
             ("Is {gene} up-regulated or functionally active in {tissue} in patients with {disease}?",
              "In {disease}, does the gene {gene} play an active role in the {tissue}?")),
    Question("Q2", "Is {gene} part of the disease mechanism chain of {disease} (i.e. does it act at one of the stages of the pathogenic cascade)?",
             ("Does {gene} act at a defined step of the pathogenic cascade of {disease}?",
              "Can {gene} be placed on the cause-to-symptom pathway of {disease}?")),
    Question("Q3", "Would inhibiting {gene} be expected to improve the symptoms of {disease}?",
             ("Is blocking {gene} expected to reduce disease activity in {disease}?",
              "Does loss or inhibition of {gene} ameliorate {disease}?")),
    Question("Q4", "Is {gene} a druggable protein class (enzyme, receptor, secreted or cell-surface protein) that a small molecule or antibody could target?",
             ("Does {gene} belong to a protein class that is readily targeted by drugs (enzyme, receptor, ion channel, secreted or membrane protein)?",
              "Is the protein encoded by {gene} accessible to small molecules or antibodies?")),
    Question("Q5", "Would inhibiting {gene} be expected to cause serious adverse effects (e.g. severe infection, cytopenia, organ toxicity)?",
             ("Is systemic inhibition of {gene} likely to cause serious safety problems?",
              "Is {gene} essential for functions whose loss causes severe toxicity?"),
             invert=True),
    Question("Q6", "Do common or rare genetic variants in {gene} alter the risk of {disease} in humans (GWAS or Mendelian evidence)?",
             ("Is {gene} a human-genetics-supported locus for {disease}?",
              "Has human genetic evidence linked {gene} to susceptibility to {disease}?")),
]

QUESTION_BY_ID: Dict[str, Question] = {q.qid: q for q in QUESTIONS}
DEMO_QIDS = ("Q1", "Q2", "Q3", "Q4", "Q5")


# ここから下は「文章で答えさせる」質問の雛形。用途は名前のとおり:
#   REVERSE_DRUG_QUESTION  逆引き検証「この遺伝子を狙う承認薬はあるか」→ あれば「既知」
#   DIRECT_PARTNERS/UPSTREAM/DOWNSTREAM_PROMPT  関係の網を広げる
#   SAME_STAGE_PROMPT      同じ段階の遺伝子を挙げさせる
#   STAGE/DIRECTION_PROMPT 候補が鎖のどの段階か、止めると良くなる向きか
#   DESCRIPTION_PROMPT     遺伝子の説明文（埋め込み＝地図の近さ V に使う）
#   CRITIC_PROMPT          反論役「失敗する理由を3つ」
#   PAIRWISE_PROMPT        対戦比較「どちらが妥当か」
#   SEED_PROMPT/SEED_DRUG_CHECK  種の抽出と、承認薬による確認
# Verification / expansion prompts (free-text generation).
REVERSE_DRUG_QUESTION = "Is there an approved drug whose molecular target is {gene}, approved for {disease} in any country? Answer Yes or No."
DIRECT_PARTNERS_PROMPT = "List the human genes whose protein products directly bind to, are the receptor or ligand of, or are in the same complex as {gene}. Answer with official gene symbols separated by commas only."
UPSTREAM_PROMPT = "List the human genes that act immediately upstream of {gene} in its signalling pathway. Answer with official gene symbols separated by commas only."
DOWNSTREAM_PROMPT = "List the human genes that act immediately downstream of {gene}, i.e. whose expression or activity is increased by {gene}. Answer with official gene symbols separated by commas only."
SAME_STAGE_PROMPT = "In the pathogenesis of {disease}, stage '{stage}' involves these genes: {seeds}. List other human genes that act at the same stage and in the same direction. Answer with official gene symbols separated by commas only."
STAGE_PROMPT = "The mechanism of {disease} can be described as the following chain of stages:\n{chain}\nAt which stage does {gene} act? Answer with the stage id only (e.g. S3), or NONE if it is not on this chain."
DIRECTION_PROMPT = "In {disease}, does the activity of {gene} make the disease WORSE (so inhibiting it would help) or BETTER (so inhibiting it would harm)? Answer with WORSE or BETTER only."
DESCRIPTION_PROMPT = "Describe the biological function of the human gene {gene} in three sentences: its protein class, its pathway, and the tissues and cell types where it is active."
CRITIC_PROMPT = "You are a sceptical reviewer. Give the three strongest reasons why {gene} could FAIL as a drug target for {disease}. Be concise."
PAIRWISE_PROMPT = "Which gene is the more plausible drug target for {disease}: {a} or {b}? Answer with the gene symbol only."
SEED_PROMPT = "List the validated drug targets of {disease}, i.e. genes whose protein is the molecular target of an approved drug for {disease}. Answer with official gene symbols separated by commas only."
SEED_DRUG_CHECK = "Name an approved drug for {disease} whose molecular target is {gene}. If none exists, answer NONE."
