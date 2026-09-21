"""Replay the RA demo through the loop and write a markdown report."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from demo.ra_demo import run
from target_loop.report import to_markdown

if __name__ == "__main__":
    res, backend = run()
    md = to_markdown(res)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/ra_demo_report.md", "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"[backend calls: {len(backend.calls)}]")
