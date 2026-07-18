#!/usr/bin/env python3
"""Aggregate per-run metrics for the design-v2 factorial.

Outputs (results/aggregate/):
  summary.json       -- model -> dataset -> cond_id -> temperature: mean/SD per
                        dimension (the 24-condition cell level).
  main_effects.json  -- marginal mean F1 with each knowledge factor
                        (domain/dictionary/rules) and the headers factor off/on,
                        per model x dataset x dimension, at temperature 0.
  token_usage.json   -- token accounting over all runs.
  stability.json     -- exact-output agreement and F1 dispersion per temperature.

Usage: python aggregate.py
"""

import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import erdbench as eb

BASE = eb.BASE
CFG = json.load(open(os.path.join(BASE, "config", "experiment.json"), encoding="utf-8"))
DIMS = ["entities", "relationships", "keys", "attributes"]
MODELS = [m["key"] for m in CFG["models"]]
PROV_KEY = {m["provider"]: m["key"] for m in CFG["models"]}


def mean_sd(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def load_metrics():
    recs = []
    for path in sorted(glob.glob(os.path.join(BASE, CFG["paths"]["results_metrics"], "*.json"))):
        r = json.load(open(path, encoding="utf-8"))
        r["model"] = PROV_KEY.get(r["provider"], r["provider"])
        recs.append(r)
    return recs


def build_summary(recs):
    cells = {}
    for r in recs:
        key = (r["model"], r["dataset"], r["cond_id"], str(r["temperature"]))
        cells.setdefault(key, []).append(r)
    summary = {}
    for (model, ds, cond, temp), rs in sorted(cells.items()):
        node = summary.setdefault(model, {}).setdefault(ds, {}) \
                      .setdefault(cond, {}).setdefault(temp, {})
        node["n"] = len(rs)
        node["parse_failures"] = sum(1 for r in rs if r.get("parse_failure"))
        for dim in DIMS:
            node[dim] = {}
            for metric in ("precision", "recall", "f1"):
                m, s = mean_sd([r["metrics"][dim][metric] for r in rs])
                node[dim][metric] = {"mean": round(m, 4) if m is not None else None,
                                     "sd": round(s, 4) if s is not None else None}
    return summary


def main_effects(recs):
    """Marginal mean F1 off/on for each factor, per model x dataset x dim, at T=0."""
    t0 = [r for r in recs if r["temperature"] == 0.0]
    out = {}
    factors = ["domain", "dictionary", "rules"]
    for model in MODELS:
        for ds in CFG["datasets"]:
            sub = [r for r in t0 if r["model"] == model and r["dataset"] == ds]
            if not sub:
                continue
            node = out.setdefault(model, {}).setdefault(ds, {})
            for dim in DIMS:
                node[dim] = {}
                # knowledge factors: compare off vs on at meaningful headers
                mean_hdr = [r for r in sub if r["headers"] == "meaningful"]
                for f in factors:
                    off = [r["metrics"][dim]["f1"] for r in mean_hdr if not r[f]]
                    on = [r["metrics"][dim]["f1"] for r in mean_hdr if r[f]]
                    mo, _ = mean_sd(off); mn, _ = mean_sd(on)
                    node[dim][f] = {
                        "off": round(mo, 4) if mo is not None else None,
                        "on": round(mn, 4) if mn is not None else None,
                        "delta": round(mn - mo, 4) if (mo is not None and mn is not None) else None}
                # headers factor: mean F1 per header level (over all knowledge combos)
                node[dim]["headers"] = {}
                for h in ("meaningful", "cryptic", "none"):
                    vals = [r["metrics"][dim]["f1"] for r in sub if r["headers"] == h]
                    m, _ = mean_sd(vals)
                    node[dim]["headers"][h] = round(m, 4) if m is not None else None
    return out


def token_usage():
    """Total tokens across the whole benchmark, from the raw run records."""
    raw_dir = os.path.join(BASE, CFG["paths"]["results_raw"])
    total = {"runs": 0, "input_tokens": 0, "output_tokens": 0}
    by = {}
    for path in glob.glob(os.path.join(raw_dir, "*.json")):
        r = json.load(open(path, encoding="utf-8"))
        model = PROV_KEY.get(r["provider"], r["provider"])
        u = r.get("usage", {})
        it, ot = u.get("input_tokens") or 0, u.get("output_tokens") or 0
        total["runs"] += 1
        total["input_tokens"] += it
        total["output_tokens"] += ot
        k = f"{model}/{r['dataset']}"
        b = by.setdefault(k, {"runs": 0, "input_tokens": 0, "output_tokens": 0})
        b["runs"] += 1; b["input_tokens"] += it; b["output_tokens"] += ot
    total["total_tokens"] = total["input_tokens"] + total["output_tokens"]
    return {"total": total, "by_model_dataset": by}


def main():
    recs = load_metrics()
    if not recs:
        print("no metrics found; run parse_outputs.py then evaluate.py first")
        return
    agg_dir = os.path.join(BASE, CFG["paths"]["results_aggregate"])
    os.makedirs(agg_dir, exist_ok=True)
    summary = build_summary(recs)
    effects = main_effects(recs)
    json.dump(summary, open(os.path.join(agg_dir, "summary.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    json.dump(effects, open(os.path.join(agg_dir, "main_effects.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    tokens = token_usage()
    json.dump(tokens, open(os.path.join(agg_dir, "token_usage.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    n_cells = sum(1 for m in summary.values() for ds in m.values()
                  for c in ds.values() for _ in c.values())
    t = tokens["total"]
    print(f"aggregated {len(recs)} runs; {n_cells} cells -> summary.json + main_effects.json")
    print(f"benchmark footprint: {t['runs']} runs, {t['total_tokens']:,} tokens "
          f"({t['input_tokens']:,} in / {t['output_tokens']:,} out) -> token_usage.json")


def stability_metrics():
    """Exact-output agreement + F1 dispersion per temperature (k111__hm)."""
    import hashlib
    raw_dir = os.path.join(BASE, CFG["paths"]["results_raw"])
    groups = {}
    for path in glob.glob(os.path.join(raw_dir, "*.json")):
        r = json.load(open(path, encoding="utf-8"))
        if r["provider"] == "baseline":
            continue
        key = (PROV_KEY.get(r["provider"], r["provider"]), r["dataset"],
               r["cond_id"], str(r["temperature"]))
        h = hashlib.sha256(r["response_text"].encode("utf-8")).hexdigest()
        groups.setdefault(key, []).append(h)
    out = {}
    for (model, ds, cond, temp), hashes in sorted(groups.items()):
        node = out.setdefault(model, {}).setdefault(ds, {}) \
                  .setdefault(cond, {}).setdefault(temp, {})
        node["n"] = len(hashes)
        node["identical_outputs"] = len(hashes) - len(set(hashes)) + 1 \
            if len(set(hashes)) < len(hashes) else (1 if len(hashes) > 1 else 1)
        node["all_identical"] = len(set(hashes)) == 1
        node["distinct_outputs"] = len(set(hashes))
    return out


def emit_stability():
    agg_dir = os.path.join(BASE, CFG["paths"]["results_aggregate"])
    stab = stability_metrics()
    json.dump(stab, open(os.path.join(agg_dir, "stability.json"), "w",
                         encoding="utf-8"), indent=2, ensure_ascii=False)
    print("stability.json written")


if __name__ == "__main__":
    main()
    emit_stability()
