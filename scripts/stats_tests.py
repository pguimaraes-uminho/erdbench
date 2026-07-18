#!/usr/bin/env python3
"""Statistical tests reported in Section 4 of the paper.

Recomputes, from results/metrics/*.json, every inferential number quoted in the
manuscript, so that the claim "every number reported was computed automatically"
holds for the test statistics as well:

  1. Model gap (Section 4.1, fourth observation): median paired advantage of
     Gemini over Mistral in F1 across condition-by-dimension-by-temperature
     cells, per dataset, with a Wilcoxon signed-rank test.
  2. Temperature effect on quality (Section 4.4): grand mean F1 at t=0 vs
     t=0.5 per model; median per-cell change over the 192 condition-by-
     dimension cells; Wilcoxon signed-rank t0 vs t0.5 per model; Friedman
     test across the three temperatures per model.
  3. Temperature effect on reproducibility (Section 4.4): mean within-
     condition standard deviation over replicates at t=0 and t=0.5 per model.

Cell definition: one cell is (dataset, cond_id, dimension); its value at a
temperature is the mean F1 over the replicates. No baseline runs are used.

Usage: python3 scripts/stats_tests.py [--metrics-dir results/metrics]
"""
import argparse
import glob
import json
import os
import statistics
from collections import defaultdict

from scipy.stats import wilcoxon, friedmanchisquare

DIMENSIONS = ["entities", "relationships", "keys", "attributes"]
MODELS = ["google", "mistral"]  # provider strings in the run records; google = Gemini 2.5 Flash
MODEL_LABEL = {"google": "gemini", "mistral": "mistral"}
TEMPS = [0.0, 0.2, 0.5]


def load_cells(metrics_dir):
    """cells[(model, dataset, cond_id, dim)][temp] -> list of replicate F1s."""
    cells = defaultdict(lambda: defaultdict(list))
    for path in sorted(glob.glob(os.path.join(metrics_dir, "*.json"))):
        rec = json.load(open(path))
        model = rec["provider"]
        if model not in MODELS:
            continue  # deterministic baseline: no random component, no tests
        for dim in DIMENSIONS:
            key = (model, rec["dataset"], rec["cond_id"], dim)
            cells[key][rec["temperature"]].append(rec["metrics"][dim]["f1"])
    return cells


def mean_f1(cells, key, temp):
    return statistics.mean(cells[key][temp])


def model_gap(cells, dataset):
    """Median Gemini-minus-Mistral delta and Wilcoxon p over all cells of one dataset."""
    deltas = []
    conds = sorted({k[2] for k in cells if k[0] == "google" and k[1] == dataset})
    for cond in conds:
        for dim in DIMENSIONS:
            for t in TEMPS:
                g = mean_f1(cells, ("google", dataset, cond, dim), t)
                m = mean_f1(cells, ("mistral", dataset, cond, dim), t)
                deltas.append(g - m)
    nonzero = [d for d in deltas if d != 0]
    stat, p = wilcoxon(deltas) if nonzero else (float("nan"), 1.0)
    return statistics.median(deltas), p, len(deltas)


def temperature_effect(cells, model):
    """Per-model quality effect of temperature over condition-by-dimension cells."""
    keys = sorted(k for k in cells if k[0] == model)
    v0 = [mean_f1(cells, k, 0.0) for k in keys]
    v2 = [mean_f1(cells, k, 0.2) for k in keys]
    v5 = [mean_f1(cells, k, 0.5) for k in keys]
    deltas = [b - a for a, b in zip(v0, v5)]
    w_stat, w_p = wilcoxon(v0, v5)
    f_stat, f_p = friedmanchisquare(v0, v2, v5)
    return {
        "cells": len(keys),
        "grand_mean_t0": statistics.mean(v0),
        "grand_mean_t05": statistics.mean(v5),
        "median_delta_t0_t05": statistics.median(deltas),
        "wilcoxon_p_t0_vs_t05": w_p,
        "friedman_p_3temps": f_p,
    }


def within_condition_sd(cells, model, temp):
    """Mean SD over replicates, across all condition-by-dimension cells.

    Sample SD (ddof=1), the same convention as the mean±SD entries of the
    result tables."""
    sds = [statistics.stdev(cells[k][temp]) for k in sorted(cells) if k[0] == model]
    return statistics.mean(sds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "metrics"))
    args = ap.parse_args()
    cells = load_cells(args.metrics_dir)

    print("== Model gap (Gemini - Mistral), per dataset ==")
    for dataset in ("airlines", "manufacturing"):
        med, p, n = model_gap(cells, dataset)
        print(f"{dataset}: median delta F1 = {med:+.3f} over {n} cells, Wilcoxon p = {p:.2e}")

    print("\n== Temperature effect on quality, per model ==")
    for model in MODELS:
        r = temperature_effect(cells, model)
        print(f"{MODEL_LABEL[model]}: grand mean F1 {r['grand_mean_t0']:.3f} (t=0) -> "
              f"{r['grand_mean_t05']:.3f} (t=0.5) over {r['cells']} cells; "
              f"median per-cell delta = {r['median_delta_t0_t05']:.3f}; "
              f"Wilcoxon t0 vs t0.5 p = {r['wilcoxon_p_t0_vs_t05']:.3f}; "
              f"Friedman (3 temps) p = {r['friedman_p_3temps']:.3f}")

    print("\n== Temperature effect on reproducibility, per model ==")
    for model in MODELS:
        sd0 = within_condition_sd(cells, model, 0.0)
        sd5 = within_condition_sd(cells, model, 0.5)
        print(f"{MODEL_LABEL[model]}: mean within-condition SD {sd0:.3f} (t=0) -> {sd5:.3f} (t=0.5)")


if __name__ == "__main__":
    main()
