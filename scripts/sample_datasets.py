#!/usr/bin/env python3
"""Deterministic stratified sampling of the canonical datasets.

The full 1000-row CSVs are too large to embed in
every prompt. This script builds ONE frozen sample per dataset that keeps
the structural evidence an ERD needs:

  * group-preserving  -- the sampling unit is the natural row group (a whole
    flight / a whole order), never an individual row, so 1:N cardinality and
    within-group functional dependencies stay observable;
  * coverage-driven   -- a deterministic greedy set-cover keeps adding whole
    groups until every distinct value of every low-cardinality column
    (distinct <= MAX_CARD) appears at least MIN_OCC times in the sample
    (or as many times as it occurs in the full data, if that is fewer);
  * fully deterministic -- greedy choice with lexicographic tie-break on the
    group key; no RNG. Re-running reproduces the sample byte-identically.

Outputs (committed, frozen before any execution):
  datasets/samples/<dataset>_sample.csv
  datasets/samples/representativeness_report.json
  datasets/samples/CHECKSUMS.txt

Usage:
  python sample_datasets.py            # build samples + report + checksums
  python sample_datasets.py --verify   # rebuild in memory and compare
"""

import argparse
import csv
import hashlib
import io
import json
import os
import sys
from collections import Counter, defaultdict, OrderedDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "datasets"))
SAMPLES_DIR = os.path.join(DATASETS_DIR, "samples")

# Sampling parameters (mirrored in config/experiment.json; frozen before any execution).
MAX_CARD = 50   # a column is a coverage target iff it has <= MAX_CARD distinct values
MIN_OCC = 2     # each distinct value must appear >= MIN_OCC times in the sample

DATASETS = OrderedDict([
    ("airlines", {
        "csv": "airlines_ground_truth_1000.csv",
        "group_keys": ["flight_number", "flight_date"],
    }),
    ("manufacturing", {
        "csv": "manufacturing_ground_truth_1000.csv",
        "group_keys": ["n_order"],
    }),
])


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, [row for row in reader]


def coverage_columns(fieldnames, rows):
    """Columns with <= MAX_CARD distinct values, in file order."""
    cols = []
    for col in fieldnames:
        distinct = len({r[col] for r in rows})
        if distinct <= MAX_CARD:
            cols.append(col)
    return cols


def requirements(rows, cov_cols):
    """Required occurrence count per (column, value): min(MIN_OCC, full_count)."""
    full = Counter()
    for r in rows:
        for col in cov_cols:
            full[(col, r[col])] += 1
    return {kv: min(MIN_OCC, c) for kv, c in full.items()}


def group_rows(rows, group_keys):
    """Ordered map group_key_tuple -> list of row indices (stable file order)."""
    groups = OrderedDict()
    for idx, r in enumerate(rows):
        key = tuple(r[k] for k in group_keys)
        groups.setdefault(key, []).append(idx)
    return groups


def group_contribution(rows, indices, cov_cols):
    c = Counter()
    for idx in indices:
        r = rows[idx]
        for col in cov_cols:
            c[(col, r[col])] += 1
    return c


def greedy_cover(rows, groups, cov_cols, req):
    """Pick whole groups greedily until every requirement is met.

    Marginal gain of a group = sum over its (col,value) contributions of the
    still-unmet portion. Ties broken by the lexicographically smallest group
    key, so the result is deterministic.
    """
    contribs = {key: group_contribution(rows, idxs, cov_cols)
                for key, idxs in groups.items()}
    current = Counter()
    selected = []
    remaining = set(groups.keys())

    def unmet(kv):
        return max(0, req[kv] - current[kv])

    while True:
        deficit = sum(unmet(kv) for kv in req)
        if deficit == 0:
            break
        best_key, best_gain = None, -1
        for key in sorted(remaining):
            c = contribs[key]
            gain = 0
            for kv, cnt in c.items():
                if kv in req:
                    gain += min(cnt, unmet(kv))
            if gain > best_gain:
                best_gain, best_key = gain, key
        if best_gain <= 0:
            # No group can reduce the deficit further (should not happen given
            # requirements are capped at full-data occurrence counts).
            break
        selected.append(best_key)
        remaining.discard(best_key)
        current.update(contribs[best_key])
    return selected


def build_sample(dataset):
    spec = DATASETS[dataset]
    path = os.path.join(DATASETS_DIR, spec["csv"])
    fieldnames, rows = read_rows(path)
    cov_cols = coverage_columns(fieldnames, rows)
    req = requirements(rows, cov_cols)
    groups = group_rows(rows, spec["group_keys"])
    selected = greedy_cover(rows, groups, cov_cols, req)

    selected_set = set(selected)
    kept_indices = [idx for key in groups if key in selected_set
                    for idx in groups[key]]
    kept_indices.sort()  # emit in original file order

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for idx in kept_indices:
        writer.writerow(rows[idx])
    sample_csv = buf.getvalue()

    report = representativeness(dataset, fieldnames, rows, cov_cols,
                               kept_indices, selected, spec["group_keys"])
    return sample_csv, report


def representativeness(dataset, fieldnames, rows, cov_cols, kept_indices,
                       selected, group_keys):
    sample_rows = [rows[i] for i in kept_indices]
    per_column = OrderedDict()
    for col in fieldnames:
        full_vals = Counter(r[col] for r in rows)
        samp_vals = Counter(r[col] for r in sample_rows)
        distinct_full = len(full_vals)
        distinct_cov = len(samp_vals)
        ge2_full = sum(1 for v, c in full_vals.items() if c >= MIN_OCC)
        ge2_cov = sum(1 for v in full_vals
                      if full_vals[v] >= MIN_OCC and samp_vals[v] >= MIN_OCC)
        per_column[col] = {
            "is_coverage_target": col in cov_cols,
            "distinct_full": distinct_full,
            "distinct_in_sample": distinct_cov,
            "distinct_coverage_pct": round(100.0 * distinct_cov / distinct_full, 1),
            "values_ge2_full": ge2_full,
            "values_ge2_in_sample": ge2_cov,
        }
    return {
        "dataset": dataset,
        "params": {"max_cardinality": MAX_CARD, "min_occurrences": MIN_OCC,
                   "group_keys": group_keys},
        "full_rows": len(rows),
        "sample_rows": len(sample_rows),
        "full_groups": len({tuple(r[k] for k in group_keys) for r in rows}),
        "sample_groups": len(selected),
        "coverage_columns": cov_cols,
        "per_column": per_column,
    }


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="rebuild in memory and compare with committed files")
    args = parser.parse_args()

    samples, report = {}, {}
    for dataset in DATASETS:
        sample_csv, rep = build_sample(dataset)
        samples[f"{dataset}_sample.csv"] = sample_csv
        report[dataset] = rep

    report_json = json.dumps(report, indent=2, ensure_ascii=False) + "\n"

    if args.verify:
        ok = True
        for name, text in samples.items():
            path = os.path.join(SAMPLES_DIR, name)
            if not os.path.exists(path):
                print(f"MISSING {name}"); ok = False; continue
            with open(path, encoding="utf-8", newline="") as f:
                on_disk = f.read()
            status = "OK" if on_disk == text else "MISMATCH"
            if status == "MISMATCH": ok = False
            print(f"{status} {name} sha256={sha256(text)}")
        rp = os.path.join(SAMPLES_DIR, "representativeness_report.json")
        if not os.path.exists(rp) or open(rp, encoding="utf-8").read() != report_json:
            print("MISMATCH representativeness_report.json"); ok = False
        else:
            print("OK representativeness_report.json")
        sys.exit(0 if ok else 1)

    os.makedirs(SAMPLES_DIR, exist_ok=True)
    checksum_lines = []
    for name, text in samples.items():
        path = os.path.join(SAMPLES_DIR, name)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        rep = report[name.replace("_sample.csv", "")]
        checksum_lines.append(f"{sha256(text)}  {name}")
        print(f"wrote {name}: {rep['sample_rows']} rows / "
              f"{rep['sample_groups']} of {rep['full_groups']} groups "
              f"(~{len(text) // 4} tokens)")
    with open(os.path.join(SAMPLES_DIR, "representativeness_report.json"),
              "w", encoding="utf-8", newline="") as f:
        f.write(report_json)
    checksum_lines.append(f"{sha256(report_json)}  representativeness_report.json")
    with open(os.path.join(SAMPLES_DIR, "CHECKSUMS.txt"),
              "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(checksum_lines) + "\n")
    print("wrote representativeness_report.json + CHECKSUMS.txt")


if __name__ == "__main__":
    main()
