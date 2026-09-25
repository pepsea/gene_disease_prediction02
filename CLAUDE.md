# CLAUDE.md

## 応答言語
- ユーザーへの回答・説明は常に日本語で書く（コード中のコメントも既存に合わせて日本語）。
- LLM に投げるプロンプト（質問文・前置き）は英語のまま。

## 検証の進め方
- 質問パターンの検証は `scripts/yesno_question_variants.py`（GGUF: `~/llm/models/txgemma-9b-chat-Q6_K.gguf`）で行い、`--qids` で追加した質問だけ聞いて `outputs/<疾患>_set100_qvariants.csv` に足す。
- 取りこぼした既知遺伝子の救済（レスキュー）の効果は `scripts/rescue_report.py` で、既知遺伝子の順位と AUC、ダミーの Yes 率、C1（知名度の対照）との相関を見て判断する。
