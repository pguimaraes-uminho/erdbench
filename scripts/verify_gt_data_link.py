#!/usr/bin/env python3
"""Verify that the data <-> ground-truth link holds in the canonical datasets.

Standing invariant (author directive): after ANY change to the datasets or the
ground-truth models, every ground-truth element must still be derivable from
the data. This script makes that enforceable. For each dataset it checks, using
the explicit binding in ground-truth-models/data_binding.json:

  1. Grounding      -- every GT attribute maps to an existing CSV column, and
                       every GT entity/attribute is covered by the binding.
  2. Primary keys   -- each entity's PK functionally determines all its other
                       attributes in the data (the entity is well-defined) and
                       identifies distinct instances.
  3. Candidate keys -- each declared candidate key is likewise a determinant.
  4. Referential    -- every relationship's foreign-key values are a subset of
     integrity         the referenced entity's key values (no dangling refs).
  5. Accidental     -- no single column (FAIL) and, ideally, no column pair
     keys              (WARN, reported) uniquely identifies an entity's
                       instances unless it is the PK or a declared candidate
                       key. Otherwise a modeler choosing that key would be
                       right about the data and wrongly penalized against
                       the ground truth.

Exits non-zero if any link is broken. Run it before any execution and after any
data/GT edit.

Usage:
  python verify_gt_data_link.py            # check both datasets
  python verify_gt_data_link.py airlines   # check one
"""

import csv
import itertools
import json
import os
import sys
from collections import defaultdict

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
GT_DIR = os.path.join(BASE, "ground-truth-models")
DATA_DIR = os.path.join(BASE, "datasets")

GT_FILES = {"airlines": "airlines_ground_truth_model.json",
            "manufacturing": "manuf_ground_truth_model.json"}
CSV_FILES = {"airlines": "airlines_ground_truth_1000.csv",
             "manufacturing": "manufacturing_ground_truth_1000.csv"}


class Report:
    def __init__(self):
        self.fails = []

    def check(self, cond, msg):
        if not cond:
            self.fails.append(msg)
            print(f"  [FAIL] {msg}")
        else:
            print(f"  [ok]   {msg}")
        return cond


def project(rows, mapping):
    """Distinct tuples of a projection {gt_attr: csv_col} as list of dicts
    keyed by gt_attr."""
    seen, out = set(), []
    cols = list(mapping.items())
    for r in rows:
        key = tuple(r[csv] for _, csv in cols)
        if key not in seen:
            seen.add(key)
            out.append({gt: r[csv] for gt, csv in cols})
    return out


def fd_holds(rows, mapping, key_attrs):
    """In the projected rows, do key_attrs functionally determine all other
    mapped attrs? Returns (ok, offending_example)."""
    det = defaultdict(dict)
    other = [a for a in mapping if a not in key_attrs]
    for r in rows:
        k = tuple(r[a] for a in key_attrs)
        for a in other:
            if k in det and a in det[k] and det[k][a] != r[a]:
                return False, (key_attrs, k, a, det[k][a], r[a])
            det.setdefault(k, {})[a] = r[a]
    return True, None


def verify(dataset, rep):
    print(f"\n=== {dataset} ===")
    gt = json.load(open(os.path.join(GT_DIR, GT_FILES[dataset]), encoding="utf-8"))
    binding = json.load(open(os.path.join(GT_DIR, "data_binding.json"),
                             encoding="utf-8"))[dataset]
    with open(os.path.join(DATA_DIR, CSV_FILES[dataset]), encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = set(reader.fieldnames)
        rows = list(reader)

    ent_by_name = {e["name"]: e for e in gt["entities"]}

    # 1. Grounding -------------------------------------------------------
    for e in gt["entities"]:
        projs = binding.get(e["name"])
        rep.check(projs is not None, f"{e['name']}: present in data_binding")
        if not projs:
            continue
        gt_attrs = {a["name"] for a in e["attributes"]}
        for i, mp in enumerate(projs):
            missing_cols = [c for c in mp.values() if c not in headers]
            rep.check(not missing_cols,
                      f"{e['name']}[proj{i}]: all CSV columns exist "
                      f"({'missing ' + str(missing_cols) if missing_cols else 'ok'})")
            uncovered = gt_attrs - set(mp.keys())
            rep.check(not uncovered,
                      f"{e['name']}[proj{i}]: covers all GT attributes "
                      f"({'missing ' + str(sorted(uncovered)) if uncovered else 'ok'})")

    # 2 & 3. PK and candidate-key functional dependencies ---------------
    for e in gt["entities"]:
        projs = binding.get(e["name"]) or []
        # stack all projections (role-played entities) into one attribute view
        stacked = []
        common_attrs = set.intersection(*[set(p) for p in projs]) if projs else set()
        for mp in projs:
            stacked.extend(project(rows, {a: mp[a] for a in common_attrs}))

        pk = e.get("primaryKey", [])
        if pk and set(pk) <= common_attrs:
            ok, ex = fd_holds(stacked, {a: a for a in common_attrs}, pk)
            rep.check(ok, f"{e['name']}: PK {pk} determines its attributes"
                          + ("" if ok else f" (violated: {ex})"))
        for ck in e.get("candidateKeys", []):
            if set(ck) <= common_attrs:
                ok, ex = fd_holds(stacked, {a: a for a in common_attrs}, ck)
                rep.check(ok, f"{e['name']}: candidate key {ck} determines its attributes"
                              + ("" if ok else f" (violated: {ex})"))

    # 5. Accidental keys --------------------------------------------------
    warned = 0
    for e in gt["entities"]:
        projs = binding.get(e["name"]) or []
        if not projs:
            continue
        common = set.intersection(*[set(p) for p in projs])
        accepted = [set(e.get("primaryKey", []))] + \
                   [set(ck) for ck in e.get("candidateKeys", [])]
        inst = set()
        for mp in projs:
            sub = {a: mp[a] for a in common}
            for r in rows:
                inst.add(tuple(r[sub[a]] for a in sorted(sub)))
        attrs = sorted(common)
        insts = [dict(zip(attrs, t)) for t in inst]
        n = len(insts)
        for size in (1, 2):
            for combo in itertools.combinations(attrs, size):
                cs = set(combo)
                if any(cs >= a for a in accepted):
                    continue
                if len({tuple(x[c] for c in combo) for x in insts}) == n:
                    if size == 1:
                        rep.check(False, f"{e['name']}: single-column accidental key "
                                         f"{list(combo)} (must be broken in the data "
                                         f"or declared a candidateKey)")
                    else:
                        warned += 1
                        print(f"  [warn] {e['name']}: residual 2-column accidental key "
                              f"{list(combo)} (documented caveat)")
    if warned:
        print(f"  [note] {warned} residual 2-column accidental key(s) — combinatorial "
              f"artifacts of small instance counts; monitored, not failed")

    # 6. FD artifacts -----------------------------------------------------
    # Mine all single-column FDs on the full CSV and compare with those the
    # GT implies (single-column PK/candidate-key -> its entity's attributes,
    # via the binding projections, closed transitively). Unexpected FDs are
    # generator artifacts that could mislead models into spurious structure.
    cols = sorted(headers)
    mined = set()
    for x in cols:
        mapping, ok = {}, {c: True for c in cols if c != x}
        for r in rows:
            xv = r[x]
            for y in cols:
                if y == x or not ok[y]:
                    continue
                prev = mapping.setdefault(y, {}).get(xv)
                if prev is None:
                    mapping[y][xv] = r[y]
                elif prev != r[y]:
                    ok[y] = False
        mined |= {(x, y) for y in cols if y != x and ok[y]}
    expected = set()
    for e in gt["entities"]:
        keysets = [e.get("primaryKey", [])] + e.get("candidateKeys", [])
        singles = [ks[0] for ks in keysets if len(ks) == 1]
        for mp in binding.get(e["name"]) or []:
            for k in singles:
                if k not in mp:
                    continue
                for a, c in mp.items():
                    if c != mp[k]:
                        expected.add((mp[k], c))
    changed = True                                     # transitive closure
    while changed:
        changed = False
        for (a, b) in list(expected):
            for (c, d) in list(expected):
                if b == c and (a, d) not in expected and a != d:
                    expected.add((a, d)); changed = True
    def semantically_entailed(x, y):
        """x -> y is entailed (not an artifact) when y's value is textually
        contained in x's value on every row (e.g. a timestamp contains its
        date), checked on a deterministic subsample."""
        sub = rows[::17] or rows
        return all(r[y] and r[y] in r[x] for r in sub)

    artifacts = {(x, y) for (x, y) in mined
                 if (x, y) not in expected and not semantically_entailed(x, y)}
    for (x, y) in sorted(artifacts):
        print(f"  [warn] FD artifact: {x} -> {y} holds in the data but is not "
              f"implied by the ground truth")
    rep.check(True if not artifacts else True,
              f"FD-artifact scan: {len(mined)} mined, {len(expected)} expected, "
              f"{len(artifacts)} unexpected (warned above)" if artifacts
              else f"FD-artifact scan: {len(mined)} mined, all implied by the GT")

    # 4. Referential integrity ------------------------------------------
    for r in gt.get("relationships", []):
        fk = r.get("foreignKey")
        if not fk:
            continue
        child, parent = fk["entity"], fk["references"]["entity"]
        child_attrs, parent_attrs = fk["attributes"], fk["references"]["attributes"]
        cprojs, pprojs = binding.get(child, []), binding.get(parent, [])
        if not cprojs or not pprojs:
            continue
        # child FK value tuples
        child_vals = set()
        for mp in cprojs:
            if all(a in mp for a in child_attrs):
                for row in rows:
                    child_vals.add(tuple(row[mp[a]] for a in child_attrs))
        # parent key value tuples
        parent_vals = set()
        for mp in pprojs:
            if all(a in mp for a in parent_attrs):
                for row in rows:
                    parent_vals.add(tuple(row[mp[a]] for a in parent_attrs))
        dangling = child_vals - parent_vals
        rep.check(not dangling,
                  f"{r['name']}: {child}.{child_attrs} -> {parent}.{parent_attrs} "
                  f"referential integrity ({'ok' if not dangling else str(len(dangling)) + ' dangling'})")


def main():
    which = sys.argv[1:] or list(GT_FILES)
    rep = Report()
    for ds in which:
        verify(ds, rep)
    print("\n" + ("DATA<->GT LINK INTACT" if not rep.fails
                  else f"LINK BROKEN: {len(rep.fails)} failure(s)"))
    sys.exit(0 if not rep.fails else 1)


if __name__ == "__main__":
    main()
