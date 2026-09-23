このフォルダは git 管理外です。
notebooks/*.ipynb をここにコピーして実行すると、実行結果の保存で git の更新と衝突しません。
  cp ../01_training_free_loop_demo.ipynb ../02_target_validity_yes_no.ipynb .
コピーしたノートは ROOT = os.path.abspath("..") を os.path.abspath("../..") に直してください（リポジトリ直下を指すため）。
