# プログラムの読み方（ファイル別・関数別の説明）

コードを開かなくても「どこで何をしているか」が分かるように、モジュールごとに説明します。
各ファイルの先頭にも同じ趣旨の日本語説明（docstring）を入れてあります。

## 全体の流れ

```
config.py（ルールと質問文）
    │
    ▼
loop.py（手順書）── 質問 ──▶ backends.py（AI役：記録の再生 or GGUFモデル）
    │                              ▲
    │ 候補カード                    │ 答え
    ▼                              │
models.py（入れ物） ──▶ scoring.py（採点） ──▶ pairwise.py（対戦比較） ──▶ report.py（表）
```

## target_loop/config.py — 先に決めて、後で変えないもの

| 名前 | 役割 |
|---|---|
| `ScoringRules` | 定数の箱。M/N/V の重み、隣の段階の点、減衰 0.7、最大回数、停止に使う上位k など。`frozen=True` で書き換え不可 |
| `DEFAULT_RULES` | デモのステップ0で宣言した値そのもの |
| `Question` / `QUESTIONS` | Q1〜Q5（＋任意の Q6）。本文と言い換え2通り、副作用の質問は `invert=True` |
| `*_PROMPT` | 文章で答えさせる雛形。種の検証、拡張（直接の相手・上流・下流・同じ段階）、段階・向き、説明文、反論役、対戦比較 |

## target_loop/models.py — データの入れ物

| 名前 | 役割 |
|---|---|
| `Direction` | WORSE（働くと悪化＝止めると良くなる）/ BETTER（逆）/ UNKNOWN |
| `MoAStage`, `MoAChain` | 病気のしくみの鎖（S1〜S7）。M はこの鎖の上の距離 |
| `Category` | 既知 / しくみが近い / 地図が近い / 組み合わせ / 別枠 / 根拠なし |
| `Candidate` | 候補遺伝子1つ分のカード。見つかった経路、AI の答え、採点結果、分類 |

## target_loop/scoring.py — 固定ルールで採点

| 関数 | 計算 |
|---|---|
| `question_score` | Q ＝ 各質問の「はい」確率の平均（Q5 は 1−p） |
| `moa_score` | M ＝ 同じ段階 1.0 / 隣 0.7 / それ以外 0。向きが逆なら別枠フラグ |
| `network_score` | N ＝ 既知からの段数 1 → 1.0、2 → 0.5、3以上 → 0 |
| `vector_score`, `vector_bin_from_cos` | V ＝ near 1.0 / mid 0.5 / far 0。埋め込みがあればコサインから近・中・遠へ |
| `proximity_score` | P ＝ max(M, 0.8N, 0.6V)。任意で「2種類以上」加点 |
| `total_score` | 総合 ＝ Q × P × 0.7^(回数−1) |
| `score_candidate` | 上を順に呼び、カードに書き込み、分類を付ける |

## target_loop/pairwise.py — 対戦比較

| 関数 | 役割 |
|---|---|
| `duel` | 1対戦を左右入れ替えて2回聞く。答えが違えば引き分け |
| `pairwise_rank` | 総当たり。勝ち+1、引き分け+0.5 で順位付け（Copeland） |
| `ranking_string` | `A > B ≒ C > D` の形にする |

## target_loop/backends.py — AI 役

| 名前 | 役割 |
|---|---|
| `LLMBackend` | 共通の顔：`yes_probability`（確率）、`generate`（文章）、`embed`（ベクトル） |
| `RecordedBackend` | 記録（質問文 → 答え）を再生。無い質問は `KeyError` で止める |
| `GuidanceBackend` | guidance の `select` で答えを Yes/No に限り、`_trace_nodes` の `TokenOutput.top_k`（制約前の上位 TOP_K）から確率を読む。`option_logprobs` はお使いの関数そのもの。`echo=True, top_k=TOP_K` が必要 |
| `LlamaCppBackend` | GGUF を llama.cpp で動かし、logits から Yes/No の確率を直読みする。`logits_all` は使わない |
| `parse_gene_list` | 文章から大文字の遺伝子記号だけを取り出す。`known_symbols` で照合可 |

## target_loop/loop.py — 手順書

| メソッド | 会話記録の対応 | 役割 |
|---|---|---|
| `build_seeds` | ステップ1 | 種を挙げさせ、承認薬 → 結合相手の逆引きで検証・修正（CTLA4 → CD80） |
| `expand` | ステップ2 | 種ごとに直接の相手・上流・下流・地図の隣人・同じ段階を挙げさせる |
| `annotate` | ステップ3 | 段階・向き・地図の近さ・Q1〜Q5（言い換え3 × 順序2 ＝ 6回平均） |
| `verify` | ステップ3 | 承認薬の有無で「既知」へ再分類。段階の主張を逆から聞き直す |
| `compare`, `critic` | ステップ4 | 対戦比較と反論役 |
| `run` | ステップ5 | 上を最大3回。減衰と停止判定（上位k 不変・新候補なし） |

## target_loop/report.py

`to_markdown(result)` が、種と鎖 → 採点表 → 回ごとの対戦記録 → 4分類 → 反論役 → 根拠の鎖 の順に Markdown を作ります。

## demo/ra_demo.py — RA デモの記録

会話記録の18遺伝子の答え（`ANSWERS`）、段階・向き・地図（`ANNOT`）、拡張リスト、検証結果、対戦結果、反論役の文を
読みやすい表として持ち、`build_record()` で「質問文 → 答え」の辞書に変換します。
別の病気で使うには、このファイルと同じ形で記録を作るか、`LlamaCppBackend` で本物のLLMに聞きます。

## scripts/

| ファイル | 役割 |
|---|---|
| `run_txgemma.py` | Mac 用。同じ18遺伝子を GGUF モデルで採点し、Claude の判定との相関を出す |
| `sensitivity.py` | 定数を変えたときの順位の変化（検証レポート §3） |
| `make_tiny_gguf.py` | ランダム重みの極小 GGUF。llama.cpp 経路の動作確認専用 |

## notebooks/01_training_free_loop_demo.ipynb

上の手順書のメソッドを1つずつ呼び、各セルの前に「このセルがすること」を書いたノートです。
`MODEL_PATH` を指定すると、記録の再生ではなく本物のLLMで同じセルが動きます。
