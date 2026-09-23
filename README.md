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

## Jupyter Notebook

| ノートブック | 内容 |
|---|---|
| `notebooks/01_training_free_loop_demo.ipynb` | 学習なしループ（種→広げる→確かめる→分ける）を1ステップずつ実行。RA の18遺伝子デモを再現 |
| `notebooks/03_fast_six_questions.ipynb` | **高速版**。6種類の質問（A1 既存薬の標的 / A2 ヒト遺伝学 / B1 病態経路 / B2 罹患組織・細胞 / B3 摂動で症状改善 / B4 既知標的との関係）を、前置きの KV キャッシュ保存＋1回の処理で採点。順序の癖は先頭 N 遺伝子で推定して補正。回答の尤度（PMI）も出す。GGUF 直接駆動が本命 |
| `notebooks/02_target_validity_yes_no.ipynb` | **検証用**。どの病気でも使える共通プロンプトで、遺伝子リストの各遺伝子を「活性化または抑制で改善する可能性があるか」Yes/No 判定し、スコア（確率読み／0〜9 評点／サンプリング）を集めて正解・可能性・ランダムで評価。較正チェックと対照疾患補正付き |

どちらも .py を参照しない自己完結型です。**モデルは `~/llm/models` の GGUF（guidance + llama.cpp）と起動中の Ollama から選べます。** 設定セルで `BACKEND`（auto / gguf / ollama / mock）と `MODEL_SELECT`（auto / 一覧の番号 / 名前の一部）を指定します。auto なら txgemma → medgemma → gemma の順で優先します。

### 1ステップずつ実証する

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
python scripts/run_txgemma.py --list-models        # ~/llm/models にある GGUF を表示
python scripts/run_txgemma.py --mode compare       # ~/llm/models の txgemma*.gguf を自動選択（既定: guidance 方式）
python scripts/run_txgemma.py --mode compare --engine llama_cpp                  # logits 直読み
python scripts/run_txgemma.py --model ~/llm/models/<別のモデル>.gguf --mode compare
python scripts/run_txgemma.py --engine ollama --model medgemma:4b --mode compare      # Ollama（--list-models に Ollama のモデルも出ます）
python scripts/run_txgemma.py --model ~/models/txgemma-9b-chat-Q6_K.gguf --mode loop   # 拡張もLLMに任せる
```

外部への通信は一切ありません（秘密を守る設計）。

## 遺伝子リスト（data/genes/）

| ファイル | 内容 |
|---|---|
| `hgnc_protein_coding.tsv` | HGNC のタンパク質コード遺伝子 19,297 件。記号・遺伝子名・別名・UniProt ID。AI の答えの記号照合にも使う |
| `ra_known.tsv` / `scz_known.tsv` | 正解遺伝子（承認薬の標的） |
| `ra_candidates.tsv` / `scz_candidates.tsv` | 可能性遺伝子（GWAS・エクソーム・CNV・臨床試験・生物学。`evidence` 列に種類） |
| `ra_random.tsv` / `scz_random.tsv` | ランダム遺伝子（固定シード 20260921） |
| `ra_set100.tsv` / `scz_set100.tsv` | 正解 + 可能性 + ランダム = 100（順序シャッフル） |
| `ra_set1000.tsv` / `scz_set1000.tsv` | 同 1000（set100 を含む） |

出典: HGNC 完全版（CC0）。`protein_name_uniprot` 列は UniProt REST に届く環境で `python scripts/build_gene_sets.py --uniprot` を実行すると埋まります（HGNC の `gene_name` はタンパク質コード遺伝子ではほぼ同じ名前です）。
**正解・可能性の表（`data/curated/`）は現状 Claude の知識で作った未検証のものです。** Open Targets に届く環境では `--opentargets` で置き換えられます（未テスト）。

```bash
python scripts/build_gene_sets.py                  # data/raw/hgnc_complete_set.txt から再生成
python scripts/run_txgemma.py --gene-set data/genes/ra_set100.tsv   # set100 を Q1〜Q5 で採点
```

## ファイル構成

| パス | 役割 |
|---|---|
| `target_loop/config.py` | 固定ルール（`ScoringRules`）と質問文（Q1〜Q5、任意のQ6） |
| `target_loop/scoring.py` | Q・M・N・V・P・総合点の計算と分類 |
| `target_loop/pairwise.py` | 左右入れ替え2回の対戦比較、Copeland順位 |
| `target_loop/loop.py` | 種の検証（薬→標的の逆引き）、3種類の拡張、検証、ループ制御 |
| `target_loop/backends.py` | `RecordedBackend`（記録の再生）、`GuidanceBackend`（guidance の `select` ＋ top_k トレースで確率を読む。お使いの `option_logprobs` 方式）、`OllamaBackend`（起動中の Ollama を HTTP で。logprobs 非対応ならサンプリング代用）、`LlamaCppBackend`（logits 直読み） |
| `demo/ra_demo.py` | RAデモの全データ（Claude判定・5段階確信度）と記録ファイル生成 |
| `scripts/run_txgemma.py` | Mac用：TxGemmaで同じ18遺伝子を採点・比較 |
| `scripts/sensitivity.py` | ルール定数の感度分析 |
| `scripts/make_tiny_gguf.py` | ランダム重みの極小GGUF（llama.cpp経路の動作確認用） |
| `docs/verification_report.md` | 検証結果と、設計上見つかった問題点 |
| `docs/code_guide.md` | プログラムの読み方（ファイル別・関数別の説明） |
| `docs/concept.md` | 標的探索のコンセプト（整理版） |
| `scripts/build_gene_sets.py` | HGNC と curated 表から遺伝子セット TSV を作る |

## ライセンスと商用利用

- このリポジトリのコード：依存は numpy / pandas（BSD）、pytest（MIT）、llama-cpp-python（MIT）。すべて商用利用可。
- TxGemma は Google の Health AI Developer Foundations 利用規約に従います。製品化の前に規約の確認が必要です。
