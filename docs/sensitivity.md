# Sensitivity of the fixed rules (RA replay, 13 candidates)

baseline ranking (non-known, non-excluded): CSF2 > BTK > TNFRSF1A > TYK2 > IL17A > CSF2RA > IRAK4 > IL6ST > IL11 > MYD88

## 1. Rule perturbation

| parameter | value | Kendall tau vs baseline | top-3 overlap | top-5 overlap | ranking |
|---|---|---|---|---|---|
| network_weight | 0.6 | 1.00 | 3/3 | 5/5 | CSF2 > BTK > TNFRSF1A > TYK2 > IL17A > CSF2RA > IRAK4 > IL6ST > IL11 > MYD88 |
| network_weight | 0.7 | 1.00 | 3/3 | 5/5 | CSF2 > BTK > TNFRSF1A > TYK2 > IL17A > CSF2RA > IRAK4 > IL6ST > IL11 > MYD88 |
| network_weight | 0.9 | 1.00 | 3/3 | 5/5 | CSF2 > BTK > TNFRSF1A > TYK2 > IL17A > CSF2RA > IRAK4 > IL6ST > IL11 > MYD88 |
| network_weight | 1.0 | 1.00 | 3/3 | 5/5 | CSF2 > BTK > TNFRSF1A > TYK2 > IL17A > CSF2RA > IRAK4 > IL6ST > IL11 > MYD88 |
| vector_weight | 0.4 | 0.96 | 3/3 | 5/5 | CSF2 > BTK > TNFRSF1A > TYK2 > IL17A > CSF2RA > IRAK4 > IL6ST > MYD88 > IL11 |
| vector_weight | 0.5 | 0.96 | 3/3 | 5/5 | CSF2 > BTK > TNFRSF1A > TYK2 > IL17A > CSF2RA > IRAK4 > IL6ST > MYD88 > IL11 |
| vector_weight | 0.7 | 1.00 | 3/3 | 5/5 | CSF2 > BTK > TNFRSF1A > TYK2 > IL17A > CSF2RA > IRAK4 > IL6ST > IL11 > MYD88 |
| vector_weight | 0.8 | 1.00 | 3/3 | 5/5 | CSF2 > BTK > TNFRSF1A > TYK2 > IL17A > CSF2RA > IRAK4 > IL6ST > IL11 > MYD88 |
| moa_adjacent | 0.5 | 0.96 | 3/3 | 5/5 | CSF2 > BTK > TNFRSF1A > TYK2 > IL17A > CSF2RA > IL6ST > IRAK4 > IL11 > MYD88 |
| moa_adjacent | 0.6 | 0.96 | 3/3 | 5/5 | CSF2 > BTK > TNFRSF1A > TYK2 > IL17A > CSF2RA > IL6ST > IRAK4 > IL11 > MYD88 |
| moa_adjacent | 0.8 | 0.91 | 3/3 | 5/5 | CSF2 > BTK > TNFRSF1A > TYK2 > IL17A > IRAK4 > CSF2RA > IL6ST > MYD88 > IL11 |
| moa_adjacent | 0.9 | 0.91 | 3/3 | 5/5 | CSF2 > BTK > TNFRSF1A > TYK2 > IL17A > IRAK4 > CSF2RA > IL6ST > MYD88 > IL11 |
| round_damping | 0.5 | 0.96 | 3/3 | 5/5 | CSF2 > BTK > TNFRSF1A > TYK2 > IL17A > IRAK4 > CSF2RA > IL6ST > IL11 > MYD88 |
| round_damping | 0.6 | 0.96 | 3/3 | 5/5 | CSF2 > BTK > TNFRSF1A > TYK2 > IL17A > IRAK4 > CSF2RA > IL6ST > IL11 > MYD88 |
| round_damping | 0.8 | 0.91 | 3/3 | 5/5 | CSF2 > BTK > TNFRSF1A > TYK2 > IL17A > CSF2RA > IL6ST > IRAK4 > MYD88 > IL11 |
| round_damping | 0.9 | 0.87 | 3/3 | 4/5 | CSF2 > BTK > TNFRSF1A > TYK2 > CSF2RA > IL17A > IL6ST > IRAK4 > MYD88 > IL11 |
| round_damping | 1.0 | 0.73 | 2/3 | 4/5 | CSF2 > CSF2RA > BTK > TNFRSF1A > TYK2 > IL17A > IL6ST > IRAK4 > MYD88 > IL11 |
| multi_evidence_bonus | 0.1 | 0.96 | 3/3 | 5/5 | CSF2 > BTK > TNFRSF1A > TYK2 > IL17A > IRAK4 > CSF2RA > IL6ST > IL11 > MYD88 |
| multi_evidence_bonus | 0.2 | 0.96 | 3/3 | 5/5 | CSF2 > BTK > TNFRSF1A > TYK2 > IL17A > IRAK4 > CSF2RA > IL6ST > IL11 > MYD88 |

minimum Kendall tau over the grid: 0.73; mean 0.95

## 2. Tie structure of the 5-level confidence scale

distinct totals among 10 ranked candidates: 8; tied groups: {0.9: 3}

With a 5-level scale and 5 questions Q can only take 21 distinct values, so ties are structural; pairwise comparison is needed to break them. A real yes-probability (TxGemma logits) is continuous and removes this artefact.

## 3. Damping and the stop rule are confounded

A round-r candidate's total is at most Q*P*0.7^(r-1) <= 0.7^(r-1). In round 2 that is <= 0.70; the 5th-best round-1 total is 0.85. So NO round-2 candidate can enter the top-5 while damping=0.7 and the round-1 top-5 are all above 0.7: the 'top-5 unchanged' stop fires in round 2 by construction, whatever the LLM answers.  The loop therefore never really used round 2.

| round_damping | rounds run | stop reason | round-2 genes in top-5 |
|---|---|---|---|
| 0.5 | 2 | round 2: top-5 unchanged | ─ |
| 0.7 | 2 | round 2: top-5 unchanged | ─ |
| 0.9 | 3 | round 3: no new candidates | CSF2RA |
| 1.0 | 3 | round 3: no new candidates | CSF2RA |

With damping >= 0.9 the round-2 candidate CSF2RA does enter the top-5 and the loop goes on to round 3 (which finds nothing new).  So whether the loop 'converges in 2 rounds' is decided by the damping constant, not by the LLM's answers.  Recommendation: apply damping to the proximity term only, or compare the top-k *within* each round, so that the stop rule measures the model.

## 4. Optional additions proposed after the demo

| variant | ranking |
|---|---|
| baseline (Q1-Q5) | CSF2 > BTK > TNFRSF1A > TYK2 > IL17A > CSF2RA > IRAK4 > IL6ST > IL11 > MYD88 |
| + Q6 human genetics (Claude-judged values, unverified) | TYK2 > CSF2 > BTK > TNFRSF1A > IL17A > CSF2RA > IRAK4 > IL6ST > IL11 > MYD88 |
| + 2-of-3 evidence bonus 0.15 | CSF2 > BTK > TNFRSF1A > TYK2 > IL17A > IRAK4 > CSF2RA > IL6ST > IL11 > MYD88 |
| both | TYK2 > CSF2 > BTK > TNFRSF1A > IL17A > IRAK4 > CSF2RA > IL6ST > IL11 > MYD88 |

With Q6, TYK2 (strong RA GWAS support) moves above CSF2 (no RA genetics; phase-3 failure). This is the direction the demo's own post-mortem predicted, but the Q6 values here are Claude's judgement with hindsight, so it is a consistency check of the design, not evidence.

