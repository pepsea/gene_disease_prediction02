"""【Mac 用】同じ 18 遺伝子の RA デモを、手元の GGUF モデル（TxGemma-9B-Chat など）で採点する。

    --mode compare  13 候補 × 5 問を TxGemma に聞き、Claude の確信度と並べた CSV と相関を出す
    --mode loop     拡張も TxGemma に任せて、ループ全体を回す（18 遺伝子に限られなくなる）

Run the same 18-gene RA demo with a local GGUF model (TxGemma-9B-Chat etc.)
and compare its yes-probabilities with the Claude-judged confidences.

Usage (Mac, Apple Silicon):
    pip install llama-cpp-python guidance   # Metal build is the default on macOS
    python scripts/run_txgemma.py --model ~/models/txgemma-9b-chat-Q6_K.gguf \
        --mode compare            # 18 genes x 6 questions, ~5 min on M-series
    python scripts/run_txgemma.py --model ... --mode loop        # full loop, LLM does expansion too

Output: outputs/txgemma_compare.csv  (gene, qid, claude_conf, model_p_yes, |diff|)
        outputs/txgemma_loop_report.md (mode loop)
Nothing leaves the machine: no network call is made.
"""
import argparse, csv, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from target_loop.backends import LlamaCppBackend, GuidanceBackend
from target_loop.config import QUESTIONS, ScoringRules
from target_loop.loop import TargetLoop
from target_loop.report import to_markdown
from demo.ra_demo import CHAIN, ANSWERS, SEEDS_PROPOSED


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / (sxx * syy) ** 0.5 if sxx and syy else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to a GGUF file")
    ap.add_argument("--mode", choices=["compare", "loop"], default="compare")
    ap.add_argument("--engine", choices=["guidance", "llama_cpp"], default="guidance",
                    help="guidance: select+top_k トレース方式（既定） / llama_cpp: logits 直読み")
    ap.add_argument("--top-k", type=int, default=50, help="guidance で記録する上位トークン数")
    ap.add_argument("--n-ctx", type=int, default=1024)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--no-chat-wrap", action="store_true", help="for base (non-chat) models")
    ap.add_argument("--embedding", action="store_true", help="also compute description embeddings for V")
    ap.add_argument("--genetics", action="store_true", help="include Q6 (human genetics)")
    args = ap.parse_args()

    os.makedirs("outputs", exist_ok=True)
    if args.engine == "guidance":
        b = GuidanceBackend(args.model, top_k=args.top_k, n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers)
    else:
        b = LlamaCppBackend(args.model, n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers,
                            embedding=args.embedding, chat_wrap=not args.no_chat_wrap)
    rules = ScoringRules(use_genetics_question=args.genetics)
    loop = TargetLoop(b, CHAIN, rules)
    t0 = time.time()

    if args.mode == "compare":
        rows, xs, ys = [], [], []
        qids = [q.qid for q in QUESTIONS if args.genetics or q.qid != "Q6"]
        for gene, conf in ANSWERS.items():
            for qid in qids:
                p = loop.ask(qid, gene)                      # 3 paraphrases x 2 orders
                c = conf[int(qid[1]) - 1]
                rows.append((gene, qid, c, round(p, 4), round(abs(c - p), 4)))
                xs.append(c); ys.append(p)
                print(f"{gene:9s} {qid}  claude={c:.2f}  model={p:.3f}")
        with open("outputs/txgemma_compare.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["gene", "qid", "claude_conf", "model_p_yes", "abs_diff"]); w.writerows(rows)
        print(f"\nPearson r (Claude confidence vs model p_yes) = {pearson(xs, ys):.3f}  n={len(xs)}")
        print(f"mean |diff| = {sum(r[4] for r in rows)/len(rows):.3f}")
    else:
        res = loop.run(SEEDS_PROPOSED)
        md = to_markdown(res)
        with open("outputs/txgemma_loop_report.md", "w", encoding="utf-8") as f:
            f.write(md)
        print(md)
    if getattr(b, "missing", None):
        print(f"[warn] {len(b.missing)} option(s) were not in top_k; raise --top-k. first: {b.missing[0]}")
    print(f"elapsed {time.time()-t0:.0f}s")
    b.close()


if __name__ == "__main__":
    main()
