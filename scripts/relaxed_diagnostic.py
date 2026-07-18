#!/usr/bin/env python3
"""Direction- and cardinality-agnostic relationship diagnostic.

Reports, per model/dataset/condition, the endpoint-only relationship recall (a
relationship counts as recovered if the candidate connects the same pair of
matched entities, regardless of direction or cardinality class), next to the
strict recall used in the paper. It separates 'connected the right entities'
from 'named the direction and cardinality right'. Reads the committed raw
outputs; no API calls.

Usage: python relaxed_diagnostic.py
"""
import glob, json, os, statistics, importlib.util

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "erdbench", os.path.join(BASE, "scripts", "erdbench.py"))
eb = importlib.util.module_from_spec(spec); spec.loader.exec_module(eb)


def relaxed_recall(cand, gt):
    match, _, _ = eb.match_entities(cand, gt)
    c2g = {eb.canon(cand["entities"][ci]["name"]): eb.canon(gt["entities"][gi]["name"])
           for ci, gi in match.items()}
    gt_pairs = {frozenset((eb.canon(r["parent"]), eb.canon(r["child"])))
                for r in gt["relationships"]}
    got = set()
    for r in cand["relationships"]:
        p = c2g.get(eb.canon(r["to"])); c = c2g.get(eb.canon(r["frm"]))
        if p and c:
            got.add(frozenset((p, c)))
    return (len(gt_pairs & got) / len(gt_pairs)) if gt_pairs else 0.0


def main():
    for dataset in ("airlines", "manufacturing"):
        gt = eb.load_gt(dataset)
        for model in ("gemini", "mistral"):
            for cond in ("k000__hm", "k001__hm", "k111__hm"):
                strict, relaxed = [], []
                for f in glob.glob(os.path.join(
                        BASE, "results", "raw",
                        f"{dataset}__{model}__{cond}__t00__r*.json")):
                    cand = eb.parse_candidate(json.load(open(f))["response_text"])
                    strict.append(eb.score_relationships(
                        cand, gt, eb.match_entities(cand, gt)[0])["recall"])
                    relaxed.append(relaxed_recall(cand, gt))
                if strict:
                    print(f"{dataset:14} {model:8} {cond:9} "
                          f"strict={statistics.mean(strict):.2f} "
                          f"relaxed={statistics.mean(relaxed):.2f}")


if __name__ == "__main__":
    main()
