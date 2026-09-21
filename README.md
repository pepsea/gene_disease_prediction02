# target_loop — LLMの知識だけで薬の標的候補を広げる「学習なしループ」

会話記録（2026-09-21）の最後で説明した仕組み、

> **種 → 広げる → 確かめる → 分ける** を数回くり返す。重みは学習せず、先に決めたルールで固定する。

を、実際に動くコードにしたものです。関節リウマチ（RA）の小さなデモ（既知5個＋候補13個）を
そのまま再現し、ルールの検証を行いました。検証結果は `docs/verification_report.md` にあります。

## 中学生向けの説明

- **種**：「この病気で確実に効く薬の相手（遺伝子）」を、AIに思い出させて数個選びます。
- **広げる**：種の「知り合い」を3つの方法で探します。
  1. しくみ（MoA）が同じ段階にいる（同じ仕事をしている）
  2. 直接くっつく相手・上流・下流（知り合いの知り合い）
  3. 説明文が似ている（住所が近い）
- **確かめる**：見つかった遺伝子に5つの短い質問をして「はい」の確率を集めます。
  逆方向からも聞き直して、答えが食い違わないかを見ます。
- **分ける**：既知／しくみが近い／地図が近い／逆効果（別枠）に分けて、根拠の鎖を付けて出します。

点数のルール（学習しない、先に固定）：

```
Q = 5つの質問の「はい」確率の平均（副作用の質問は 1−p）
P = max( M, 0.8×N, 0.6×V )        M:しくみ N:関係の網 V:地図
総合 = Q × P × 0.7^(ループ回数−1)
逆向き・矛盾 → 点数に関係なく別枠
```

## Jupyter Notebook で1ステップずつ実証する

```bash
pip install -e ".[dev,notebook]"
jupyter notebook notebooks/01_training_free_loop_demo.ipynb
```

種の検証 → 拡張 → 質問と回答 → 対戦比較 → 2回目 → 分類 の順に、各セルで中間結果（表・グラフ）を確認できます。
ノートブックは自己完結型で、`target_loop` の .py を参照しません。ルール・質問文・記録・採点・対戦・ループの処理がすべてセル内に書かれ、`def` は `option_logprobs`（guidance で選択肢の log 確率を読む）・AI 役の2つ・記号の取り出しの1つ、計4つだけです。
`MODEL_PATH` に GGUF を指定すると、同じ手順が TxGemma などの本物のLLMで動きます。

## 使い方

```bash
pip install -e .            # numpy, pandas のみ
python -m pytest -q         # 39件のテスト（デモ数値の再現を含む）
python demo/run_demo.py     # Claudeが判定したRAデモを再生 → outputs/ra_demo_report.md
python scripts/sensitivity.py   # ルール定数を変えると順位がどう変わるか → outputs/sensitivity.md
```

手元のMacで TxGemma-9B（GGUF）を使って同じ18遺伝子を採点し、Claudeの判定と比べる：

```bash
pip install llama-cpp-python guidance              # macOS では Metal 対応版が入ります
python scripts/run_txgemma.py --model ~/models/txgemma-9b-chat-Q6_K.gguf --mode compare            # 既定: guidance 方式
python scripts/run_txgemma.py --model ~/models/txgemma-9b-chat-Q6_K.gguf --mode compare --engine llama_cpp   # logits 直読み
python scripts/run_txgemma.py --model ~/models/txgemma-9b-chat-Q6_K.gguf --mode loop   # 拡張もLLMに任せる
```

外部への通信は一切ありません（秘密を守る設計）。

## ファイル構成

| パス | 役割 |
|---|---|
| `target_loop/config.py` | 固定ルール（`ScoringRules`）と質問文（Q1〜Q5、任意のQ6） |
| `target_loop/scoring.py` | Q・M・N・V・P・総合点の計算と分類 |
| `target_loop/pairwise.py` | 左右入れ替え2回の対戦比較、Copeland順位 |
| `target_loop/loop.py` | 種の検証（薬→標的の逆引き）、3種類の拡張、検証、ループ制御 |
| `target_loop/backends.py` | `RecordedBackend`（記録の再生）、`GuidanceBackend`（guidance の `select` ＋ top_k トレースで確率を読む。お使いの `option_logprobs` 方式）、`LlamaCppBackend`（logits 直読み） |
| `demo/ra_demo.py` | RAデモの全データ（Claude判定・5段階確信度）と記録ファイル生成 |
| `scripts/run_txgemma.py` | Mac用：TxGemmaで同じ18遺伝子を採点・比較 |
| `scripts/sensitivity.py` | ルール定数の感度分析 |
| `scripts/make_tiny_gguf.py` | ランダム重みの極小GGUF（llama.cpp経路の動作確認用） |
| `docs/verification_report.md` | 検証結果と、設計上見つかった問題点 |
| `docs/code_guide.md` | プログラムの読み方（ファイル別・関数別の説明） |

## ライセンスと商用利用

- このリポジトリのコード：依存は numpy / pandas（BSD）、pytest（MIT）、llama-cpp-python（MIT）。すべて商用利用可。
- TxGemma は Google の Health AI Developer Foundations 利用規約に従います。製品化の前に規約の確認が必要です。
