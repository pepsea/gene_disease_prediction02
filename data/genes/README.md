遺伝子リスト（TSV）を作る。

出力（data/genes/）
  hgnc_protein_coding.tsv      HGNC のタンパク質コード遺伝子 全件（記号・遺伝子名・別名・UniProt ID）。
                               ループの known_symbols（記号の照合）にも使う。
  <disease>_known.tsv          正解遺伝子（承認薬の標的）
  <disease>_candidates.tsv     可能性遺伝子（GWAS・エクソーム・CNV・臨床試験・生物学）
  <disease>_random.tsv         ダミー遺伝子（固定シードで抽出。正解・可能性を除く）。scz / cystinuria / prostate_cancer /
                               achondroplasia は SLC トランスポーター（SLC*, SLCO*）を優先して埋める（note 列 "SLC decoy"）
  <disease>_set100.tsv         正解 + 可能性 + ランダム = 100 遺伝子（順序はシャッフル）
  <disease>_set1000.tsv        正解 + 可能性 + ランダム = 1000 遺伝子（順序はシャッフル、set100 を含む）

列
  symbol            HGNC 承認記号（curated の記号が旧記号・別名なら現行記号に直す）
  hgnc_id, entrez_id, ensembl_gene_id
  gene_name         HGNC の承認名（タンパク質コード遺伝子ではほぼ UniProt の推奨タンパク質名と一致）
  uniprot_ids       HGNC が持つ UniProt アクセッション（複数は | 区切り）
  protein_name_uniprot  UniProt の推奨タンパク質名。--uniprot で REST から取得（未取得なら空欄）
  alias_symbols     HGNC の alias_symbol + prev_symbol（| 区切り）
  disease, category (known / candidate / random), label (known=1, それ以外 0)
  evidence, note, source

入力
  data/raw/hgnc_complete_set.txt   HGNC 完全版（CC0）  取得: 
      https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt
  data/curated/<disease>_known.tsv, <disease>_candidates.tsv   人が点検する表（既定は Claude の知識で作成、未検証）

オプション
  --opentargets   Open Targets GraphQL から known（承認薬の標的）と candidates（遺伝学的関連）を取り直して
                  curated の代わりに使う（この開発環境からは API に届かず未テスト）
  --uniprot       UniProt REST から protein_name_uniprot を埋める（同じく未テスト）
