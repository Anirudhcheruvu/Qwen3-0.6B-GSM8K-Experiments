# benchmark.py
import argparse
import json
from collections import Counter
from typing import List

from common import extract_answer, is_correct

def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def eval_file(path: str):
    data = load_jsonl(path)
    scores = []
    for row in data:
        gt = row.get("ground_truth_answer", "")
        out = row.get("model_output", "")
        try:
            ok = is_correct(out, gt)
            scores.append(1 if ok else 0)
        except Exception:
            scores.append(-1)

    c = Counter(scores)
    total = c[1] + c[0] + c[-1]
    acc = (c[1] / total * 100.0) if total else 0.0
    return {
        "path": path,
        "n": total,
        "correct": c[1],
        "wrong": c[0],
        "format_errors": c[-1],
        "accuracy_pct": acc,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="JSONL files produced by inference.py")
    args = ap.parse_args()

    for fp in args.files:
        r = eval_file(fp)
        print(f"\nFile: {r['path']}")
        print(f"Samples: {r['n']}")
        print(f"Correct: {r['correct']} | Wrong: {r['wrong']} | Format errors: {r['format_errors']}")
        print(f"Accuracy: {r['accuracy_pct']:.2f}%")

if __name__ == "__main__":
    main()