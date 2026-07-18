#!/usr/bin/env python3
"""Score parsed candidates against ground truth (results/parsed -> results/metrics).

One metrics file per run: TP/FP/FN and Precision/Recall/F1 for each of the four
dimensions (entities, relationships, PK&FK, attributes), plus the parse report.

Usage:
  python evaluate.py            # score all top-level parsed records
  python evaluate.py --pilot    # score results/parsed/pilot/
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import erdbench as eb

BASE = eb.BASE
CFG = json.load(open(os.path.join(BASE, "config", "experiment.json"), encoding="utf-8"))

_GT_CACHE = {}


def gt_for(dataset):
    if dataset not in _GT_CACHE:
        _GT_CACHE[dataset] = eb.load_gt(dataset)
    return _GT_CACHE[dataset]


def evaluate_all(pilot=False):
    sub = "pilot/" if pilot else ""
    parsed_dir = os.path.join(BASE, CFG["paths"]["results_parsed"], sub)
    out_dir = os.path.join(BASE, CFG["paths"]["results_metrics"], sub)
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for path in sorted(glob.glob(os.path.join(parsed_dir, "*.json"))):
        rec = json.load(open(path, encoding="utf-8"))
        cand = {"entities": rec["candidate"]["entities"],
                "relationships": rec["candidate"]["relationships"],
                "parse": rec["parse"]}
        res = eb.evaluate(cand, gt_for(rec["dataset"]))
        out = {
            "run_id": rec["run_id"],
            "provider": rec["provider"],
            "dataset": rec["dataset"],
            "cond_id": rec["cond_id"],
            "domain": rec["domain"], "dictionary": rec["dictionary"],
            "rules": rec["rules"], "headers": rec["headers"],
            "temperature": rec["temperature"],
            "replicate": rec["replicate"],
            "parse_failure": rec["parse"]["parse_failure"],
            "metrics": {dim: res[dim] for dim in
                        ("entities", "relationships", "keys", "attributes")},
        }
        with open(os.path.join(out_dir, rec["run_id"] + ".json"), "w",
                  encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        n += 1
    print(f"scored {n} records into {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    evaluate_all(args.pilot)


if __name__ == "__main__":
    main()
