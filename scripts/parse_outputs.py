#!/usr/bin/env python3
"""Parse raw responses into candidate-model JSON (results/raw -> results/parsed).

Deterministic DSL parsing (erdbench), no scoring here. Emits one file per run
with the parsed candidate model and a parse report.

Usage:
  python parse_outputs.py            # parse all top-level raw records
  python parse_outputs.py --pilot    # parse results/raw/pilot/ instead
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


def parse_all(pilot=False):
    sub = "pilot/" if pilot else ""
    raw_dir = os.path.join(BASE, CFG["paths"]["results_raw"], sub)
    out_dir = os.path.join(BASE, CFG["paths"]["results_parsed"], sub)
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.json"))):
        rec = json.load(open(path, encoding="utf-8"))
        cand = eb.parse_candidate(rec["response_text"])
        out = {
            "run_id": rec["run_id"],
            "provider": rec["provider"],
            "model_requested": rec["model_requested"],
            "dataset": rec["dataset"],
            "cond_id": rec["cond_id"],
            "domain": rec["domain"], "dictionary": rec["dictionary"],
            "rules": rec["rules"], "headers": rec["headers"],
            "temperature": rec["temperature"],
            "replicate": rec["replicate"],
            "candidate": {"entities": cand["entities"],
                          "relationships": cand["relationships"]},
            "parse": cand["parse"],
        }
        with open(os.path.join(out_dir, rec["run_id"] + ".json"), "w",
                  encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        n += 1
    print(f"parsed {n} records into {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    parse_all(args.pilot)


if __name__ == "__main__":
    main()
