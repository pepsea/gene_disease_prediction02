# rheumatoid arthritis: training-free target loop

seeds: CD80, MS4A1, TNF, IL6R, JAK1  
- seed note: CTLA4: corrected to CD80 (drug abatacept binds CD80, CD86)

## Mechanism chain

| stage | content | seeds |
|---|---|---|
| S1 | antigen presentation and T-cell activation | CD80 |
| S2 | B cells and autoantibodies | MS4A1 |
| S3 | innate immune signalling | ─ |
| S4 | inflammatory cytokines and their receptors | TNF, IL6R |
| S5 | intracellular signalling (JAK-STAT) | JAK1 |
| S6 | synovial proliferation and osteoclast activation | ─ |
| S7 | joint destruction (symptom) | ─ |

## Scores

| gene | round | found via | stage | dir | M | N | V | Q1 | Q2 | Q3 | Q4 | Q5 | Q | P | total | category |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CSF2 | 1 | TNF -> same stage S4 -> CSF2 | S4 | worse | 1.0 | 0.5 | 0.5 | 1.00 | 1.00 | 1.00 | 1.00 | 0.25 | 0.95 | 1.00 | **0.95** | moa_near |
| IL6 | 1 | IL6R -> direct partner -> IL6 | S4 | worse | 1.0 | 1.0 | 1.0 | 1.00 | 1.00 | 1.00 | 1.00 | 0.25 | 0.95 | 1.00 | **0.95** | known |
| BTK | 1 | MS4A1 -> same stage S2 -> BTK | S2 | worse | 1.0 | 0.5 | 0.5 | 0.75 | 1.00 | 1.00 | 1.00 | 0.25 | 0.90 | 1.00 | **0.90** | moa_near |
| TNFRSF1A | 1 | TNF -> direct partner -> TNFRSF1A | S4 | worse | 1.0 | 1.0 | 0.5 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 0.90 | 1.00 | **0.90** | moa_near |
| TYK2 | 1 | JAK1 -> direct partner -> TYK2 | S5 | worse | 1.0 | 1.0 | 1.0 | 0.75 | 1.00 | 1.00 | 1.00 | 0.25 | 0.90 | 1.00 | **0.90** | moa_near |
| IL17A | 1 | TNF -> same stage S4 -> IL17A | S4 | worse | 1.0 | 0.5 | 0.5 | 0.75 | 0.75 | 1.00 | 1.00 | 0.25 | 0.85 | 1.00 | **0.85** | moa_near |
| CSF2RA | 2 | CSF2 -> direct partner -> CSF2RA | S4 | worse | 1.0 | 0.0 | 0.5 | 1.00 | 1.00 | 1.00 | 1.00 | 0.25 | 0.95 | 1.00 | **0.66** | moa_near |
| TNFSF11 | 1 | TNF -> downstream effector -> TNFSF11 | S6 | worse | 0.7 | 0.5 | 0.0 | 1.00 | 1.00 | 1.00 | 1.00 | 0.25 | 0.95 | 0.70 | **0.66** | known |
| IRAK4 | 1 | TNF -> upstream regulator -> IRAK4 | S3 | worse | 0.7 | 0.5 | 0.5 | 0.75 | 1.00 | 1.00 | 1.00 | 0.50 | 0.85 | 0.70 | **0.59** | moa_near |
| IL6ST | 2 | IL6 -> direct partner -> IL6ST | S4 | worse | 1.0 | 1.0 | 1.0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.80 | 1.00 | **0.56** | moa_near |
| IL11 | 1 | IL6R -> map neighbour (same family) -> IL11 | ─ | worse | 0.0 | 0.5 | 1.0 | 0.50 | 0.50 | 0.50 | 1.00 | 0.50 | 0.60 | 0.60 | **0.36** | map_near |
| MYD88 | 2 | IRAK4 -> upstream regulator -> MYD88 | S3 | worse | 0.7 | 0.0 | 0.5 | 0.75 | 1.00 | 1.00 | 0.25 | 0.75 | 0.65 | 0.70 | **0.32** | moa_near |
| IL10 | 1 | TNF -> same stage S4 -> IL10 | S4 | better | 0.0 | 0.5 | 0.5 | 1.00 | 1.00 | 0.00 | 1.00 | 0.50 | ─ | 0.40 | **─** | excluded – opposite direction (inhibition expected to worsen disease) |

## Round 1

sources: CD80, MS4A1, TNF, IL6R, JAK1  
new candidates: BTK, TNFRSF1A, IRAK4, TNFSF11, CSF2, IL17A, IL10, IL6, IL11, TYK2  

| duel | first | swapped | verdict |
|---|---|---|---|
| CSF2 vs BTK | CSF2 | BTK | tie |
| CSF2 vs TNFRSF1A | TNFRSF1A | TNFRSF1A | TNFRSF1A |
| CSF2 vs TYK2 | CSF2 | CSF2 | CSF2 |
| BTK vs TNFRSF1A | TNFRSF1A | TNFRSF1A | TNFRSF1A |
| BTK vs TYK2 | BTK | BTK | BTK |
| TNFRSF1A vs TYK2 | TNFRSF1A | TNFRSF1A | TNFRSF1A |

pairwise ranking: **TNFRSF1A > BTK ≒ CSF2 > TYK2**  
top-5 for stop rule: CSF2, BTK, TNFRSF1A, TYK2, IL17A

## Round 2

sources: BTK, TNFRSF1A, IRAK4, TNFSF11, CSF2, IL17A, IL6, IL11, TYK2  
new candidates: MYD88, CSF2RA, IL6ST  

| duel | first | swapped | verdict |
|---|---|---|---|
| CSF2 vs BTK | CSF2 | BTK | tie |
| CSF2 vs TNFRSF1A | TNFRSF1A | TNFRSF1A | TNFRSF1A |
| CSF2 vs TYK2 | CSF2 | CSF2 | CSF2 |
| BTK vs TNFRSF1A | TNFRSF1A | TNFRSF1A | TNFRSF1A |
| BTK vs TYK2 | BTK | BTK | BTK |
| TNFRSF1A vs TYK2 | TNFRSF1A | TNFRSF1A | TNFRSF1A |

pairwise ranking: **TNFRSF1A > BTK ≒ CSF2 > TYK2**  
top-5 for stop rule: CSF2, BTK, TNFRSF1A, TYK2, IL17A

stop reason: round 2: top-5 unchanged

## Categories

| category | genes |
|---|---|
| 既知（直接の関連が確認された） | IL6, CD80, IL6R, JAK1, MS4A1, TNF, TNFSF11 |
| しくみ（MoA）が近い候補 | CSF2, BTK, TNFRSF1A, TYK2, IL17A, CSF2RA, IRAK4, IL6ST, MYD88 |
| 地図（ベクトル）が近い候補 | IL11 |
| 別枠（逆向き・矛盾） | IL10 |

## Critic notes (top 3)

- **CSF2**: Its role may be redundant with other cytokines, so blocking it alone may give a small effect. / Phase 3 anti-GM-CSF antibodies have shown insufficient efficacy. / Pulmonary alveolar proteinosis is a theoretical risk.
- **BTK**: BTK is expressed in platelets as well as B cells, so bleeding risk exists. / Oral BTK inhibitors have shown mixed efficacy in RA trials. / Off-target kinase effects.
- **TNFRSF1A**: Several TNF-blocking drugs already exist, so demonstrating added value over them is difficult. / Receptor blockade may not be superior to ligand neutralisation. / TNFR1 signalling also mediates host defence.

## Evidence chains

- CSF2: TNF -> same stage S4 -> CSF2; stage S4 (worse); verification: no approved drug for this indication
- BTK: MS4A1 -> same stage S2 -> BTK; stage S2 (worse); verification: no approved drug for this indication
- TNFRSF1A: TNF -> direct partner -> TNFRSF1A; stage S4 (worse); verification: no approved drug for this indication
- TYK2: JAK1 -> direct partner -> TYK2; stage S5 (worse); verification: no approved drug for this indication
- IL17A: TNF -> same stage S4 -> IL17A; stage S4 (worse); verification: no approved drug for this indication
- CSF2RA: CSF2 -> direct partner -> CSF2RA; stage S4 (worse); verification: no approved drug for this indication
- IRAK4: TNF -> upstream regulator -> IRAK4; stage S3 (worse); verification: no approved drug for this indication
- IL6ST: IL6 -> direct partner -> IL6ST; stage S4 (worse); verification: no approved drug for this indication
- IL11: IL6R -> map neighbour (same family) -> IL11; stage ─ (worse); verification: no approved drug for this indication
- MYD88: IRAK4 -> upstream regulator -> MYD88; stage S3 (worse); verification: no approved drug for this indication
