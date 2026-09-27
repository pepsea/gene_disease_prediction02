"""複数の GPU で、疾患を1つずつ並行に流す（Linux サーバー用。1枚に1疾患、空いた GPU に次の疾患を割り当てる）。

- 先にモデル（段階1のリランカーと生成 LLM）を Hugging Face の既定の場所へ1回だけダウンロードしておく（同時に落とさないため）。
- 段階0（機能情報）は全疾患で共通なので、最初に1回だけ流す。
- そのあと各疾患を scripts/genome_pipeline.py で、CUDA_VISIBLE_DEVICES を1枚ずつ変えて起動する。ログは outputs/genome/logs/<疾患>.log。

使い方:
  python scripts/run_gpus.py                                   # config の run_diseases を、見えている GPU すべてで
  python scripts/run_gpus.py --all                             # config/diseases.yaml の全疾患
  python scripts/run_gpus.py --gpus 0 1 --disease schizophrenia cystinuria
  python scripts/run_gpus.py --run-id latest                   # 止まった実行を、各疾患の最新の日付フォルダで続きから
"""
import argparse, os, re, subprocess, sys, time
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", nargs="*", help="病名（config/diseases.yaml の name）。省略すると run_diseases")
    ap.add_argument("--all", action="store_true", help="diseases.yaml の全疾患")
    ap.add_argument("--gpus", nargs="*", help="使う GPU の番号。省略すると見えているすべて")
    ap.add_argument("--run-id", default="", help="latest なら各疾患の最新の日付フォルダで再開")
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "pipeline07.yaml"))
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    names = [d["name"] for d in yaml.safe_load(open(os.path.join(ROOT, cfg["diseases_file"]), encoding="utf-8"))["diseases"]]
    todo = names if a.all else (a.disease or cfg["run_diseases"])
    gpus = a.gpus or [l.split(":")[0].split()[-1] for l in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines() if l.startswith("GPU")]
    print(f"疾患 {len(todo)} 件: {todo}\nGPU: {gpus}", flush=True)

    # --- 1. モデルを先に1回だけダウンロード（既定の保存場所。済んでいればすぐ終わる） ---
    from huggingface_hub import snapshot_download
    use = "server" if cfg["model"]["use"] == "auto" else cfg["model"]["use"]
    rer = cfg["stage1"]["model"]; rer = rer[use] if isinstance(rer, dict) else rer
    for repo in (rer, cfg["model"][use].get("name")):
        if repo and not os.path.isdir(repo):
            print("ダウンロード確認:", repo, "→", snapshot_download(repo), flush=True)

    # --- 2. 段階0（全疾患で共通）を1回だけ ---
    sys.path.insert(0, HERE); import genome_pipeline as GP
    GP.stage0(GP.load_config(a.config))                               # 疾患のフォルダは作らない（段階0は共通の機能情報だけ）
    py = [sys.executable, os.path.join(HERE, "genome_pipeline.py"), "--config", a.config]

    # --- 3. 各疾患を空いた GPU で起動する（1枚に1疾患） ---
    logdir = os.path.join(ROOT, cfg["paths"]["out_dir"], "logs"); os.makedirs(logdir, exist_ok=True)
    queue, running = list(todo), {}                                   # GPU 番号 → (疾患, プロセス)
    while queue or running:
        for g in gpus:
            if g not in running and queue:
                name = queue.pop(0); log = os.path.join(logdir, re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") + ".log")
                cmd = py + ["--disease", name, "--stages", "1", "2", "3", "4", "5", "6"] + (["--run-id", a.run_id] if a.run_id else [])
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": g}
                running[g] = (name, subprocess.Popen(cmd, env=env, stdout=open(log, "a"), stderr=subprocess.STDOUT))
                print(time.strftime("%H:%M:%S"), f"GPU {g}: {name} を開始（ログ {log}）", flush=True)
        for g, (name, p) in list(running.items()):
            if p.poll() is not None:
                print(time.strftime("%H:%M:%S"), f"GPU {g}: {name} が終了（終了コード {p.returncode}）", flush=True)
                del running[g]
        time.sleep(30)
    print("すべて終了", flush=True)


if __name__ == "__main__":
    main()
