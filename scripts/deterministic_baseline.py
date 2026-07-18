#!/usr/bin/env python3
"""Deterministic (no-LLM) ERD baseline.

A thin, classical data-profiling synthesizer in the tradition of database
reverse engineering: everything is computed from the VALUES of the sample the
LLMs see (never from the column-name semantics), so it runs identically under
meaningful, cryptic, and positional headers:

  * broad type inference per column;
  * functional dependencies with LHS size 1 (X -> Y iff every X value maps to
    exactly one Y value in the sample);
  * entity synthesis: each determinant column becomes an entity holding its
    closest dependents (a dependent that is itself a determinant is kept as a
    foreign key, yielding entity-to-entity relationships);
  * a base "record" entity holds the residual columns plus one FK per
    top-level determinant; its PK is the minimal unique column combination
    (LHS size <= 3) over the sample;
  * relationships from value-level containment (the FK column's values are a
    subset of the referenced entity's key values by construction).

Only the entity NAMES use the header tokens (generic key-ish tokens stripped,
e.g. `airline_code` -> Airline) — under cryptic/none headers the names stay
opaque, which is the point: the baseline recovers structure, not semantics.
Its outputs are emitted in the same DSL and scored by the same pipeline as the
LLMs. Known, deliberate limitations (no role unification of e.g. departure vs
arrival airport; no semantic naming without meaningful headers) are findings,
not bugs.

Writes raw-record-compatible JSONs (provider "baseline") for the 2 datasets x
3 header levels at the data-only knowledge condition (k000), replicate 1.

Usage: python deterministic_baseline.py
"""

import itertools
import json
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_prompts as bp

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CFG = json.load(open(os.path.join(BASE, "config", "experiment.json"), encoding="utf-8"))
RAW_DIR = os.path.join(BASE, CFG["paths"]["results_raw"])

GENERIC_TOKENS = {"id", "code", "number", "n", "key", "num"}


def infer_type(values):
    def all_match(pred):
        return all(pred(v) for v in values if v != "")
    if all_match(lambda v: re.fullmatch(r"-?\d+", v)):
        return "integer"
    if all_match(lambda v: re.fullmatch(r"-?\d+\.\d+", v)):
        return "decimal"
    if all_match(lambda v: re.fullmatch(r"\d{4}-\d{2}-\d{2}", v)):
        return "date"
    if all_match(lambda v: re.fullmatch(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?", v)):
        return "datetime"
    if all_match(lambda v: re.fullmatch(r"\d{2}:\d{2}(:\d{2})?", v)):
        return "time"
    return "varchar"


def merge_bijections(cols, rows, fds):
    """Columns that determine each other (X<->Y) describe ONE concept (e.g. a
    code and its name). Keep a single representative as the determinant:
    more dependents first, then the shorter mean value length (codes beat
    descriptions — a value-based, name-agnostic criterion), then lexicographic."""
    mean_len = {c: sum(len(r[c]) for r in rows) / max(1, len(rows)) for c in cols}
    dropped = set()
    for x in sorted(cols):
        if x in dropped:
            continue
        for y in sorted(fds.get(x, ())):
            if y in dropped or x in dropped:
                continue
            if x in fds.get(y, ()):          # bijective pair
                keep, drop = sorted((x, y),
                                    key=lambda c: (-len(fds[c]), mean_len[c], c))
                dropped.add(drop)
                fds[keep] = (fds[keep] | fds[drop]) - {keep, drop} | {drop}
                fds.pop(drop, None)
    for x in list(fds):
        fds[x] -= dropped - set(fds[x])      # keep dropped cols as plain deps
    return dropped


def single_fds(cols, rows):
    """{X: set(Y)} with X -> Y holding on the sample (LHS size 1)."""
    fds = {c: set() for c in cols}
    for x in cols:
        maps = {}
        broken = set()
        for r in rows:
            xv = r[x]
            for y in cols:
                if y == x or y in broken:
                    continue
                prev = maps.setdefault(y, {}).get(xv)
                if prev is None:
                    maps[y][xv] = r[y]
                elif prev != r[y]:
                    broken.add(y)
        fds[x] = {y for y in cols if y != x and y not in broken}
    return fds


def minimal_unique_combo(cols, rows, max_size=3):
    n = len(rows)
    def unique(combo):
        return len({tuple(r[c] for c in combo) for r in rows}) == n
    for size in range(1, max_size + 1):
        for combo in itertools.combinations(cols, size):
            if unique(combo):
                return list(combo)
    return list(cols)


def entity_name(col):
    tokens = [t for t in re.split(r"[_\W]+", col) if t]
    kept = [t for t in tokens if t.lower() not in GENERIC_TOKENS]
    if not kept:
        kept = tokens
    return "".join(t.capitalize() for t in kept) or col


def synthesize(cols, rows):
    """Return DSL text for the profiled sample."""
    types = {c: infer_type([r[c] for r in rows]) for c in cols}
    fds = single_fds(cols, rows)
    demoted = merge_bijections(cols, rows, fds)  # code<->name pairs become one concept
    determinants = {x for x in cols if x not in demoted and fds.get(x)}

    # closest-determinant assignment: Y belongs to X unless some Z (dependent
    # of X) also determines Y (then Z is closer).
    def closest_dependents(x):
        out = []
        for y in fds[x]:
            closer = any(z != y and z in fds[x] and y in fds[z] and x not in fds[z]
                         for z in determinants)
            if not closer:
                out.append(y)
        return out

    entities = {}   # pk_col -> {"attrs": [(col, is_fk)], "name": str}
    for x in sorted(determinants):
        deps = closest_dependents(x)
        attrs = [(x, False)]
        for y in sorted(deps):
            attrs.append((y, y in determinants))   # dependent determinant = FK
        entities[x] = {"name": entity_name(x), "attrs": attrs}

    # drop entities fully absorbed elsewhere as non-FK? keep all determinants.
    assigned = {x for x in entities} | {y for e in entities.values()
                                        for y, fk in e["attrs"] if not fk}
    residual = [c for c in cols if c not in assigned]

    # base record entity: residual + FK per top-level determinant (one not a
    # dependent of another determinant).
    top = [x for x in sorted(determinants)
           if not any(x in fds[z] for z in determinants if z != x)]
    base_attrs = [(c, False) for c in residual] + [(x, True) for x in top]
    base_cols = [c for c, _ in base_attrs]
    base_pk = minimal_unique_combo(base_cols or cols, rows)
    base_name = "Record"

    lines = []
    for x, e in sorted(entities.items(), key=lambda kv: kv[1]["name"]):
        lines.append(f"ENTITY {e['name']}")
        lines.append(f"PK {x}")
        for col, is_fk in e["attrs"]:
            flags = " : PK" if col == x else ""
            if is_fk:
                flags += " : FK"
            lines.append(f"ATTR {col} : {types[col]}{flags}")
    lines.append(f"ENTITY {base_name}")
    lines.append("PK " + ", ".join(base_pk))
    for col, is_fk in base_attrs:
        flags = " : PK" if col in base_pk else ""
        if is_fk:
            flags += " : FK"
        lines.append(f"ATTR {col} : {types[col]}{flags}")

    # relationships: base -> each top determinant entity; entity -> entity for
    # FK attributes inside dimension entities.
    for x in top:
        lines.append(f"REL {base_name} -> {entities[x]['name']} : 0..N -> 1 : FK={x}")
    for x, e in sorted(entities.items(), key=lambda kv: kv[1]["name"]):
        for col, is_fk in e["attrs"]:
            if is_fk and col in entities:
                lines.append(f"REL {e['name']} -> {entities[col]['name']} "
                             f": 0..N -> 1 : FK={col}")
    return "\n".join(lines)


def run():
    os.makedirs(RAW_DIR, exist_ok=True)
    written = 0
    csv_files = {"airlines": "airlines_ground_truth_1000.csv",
                 "manufacturing": "manufacturing_ground_truth_1000.csv"}
    for ds in bp.DATASETS:
        # The deterministic side has no context window: classical profiling
        # reads the FULL file (an inherent advantage over the sampled prompt
        # the LLMs receive; reported as such in the paper).
        sample_path = os.path.join(BASE, "datasets", csv_files[ds])
        text = open(sample_path, encoding="utf-8", newline="").read().rstrip("\n")
        lines = text.split("\n")
        orig_cols = lines[0].split(",")
        data = [dict(zip(orig_cols, ln.split(","))) for ln in lines[1:]]
        for headers in ("meaningful", "cryptic", "none"):
            new_cols = bp.transform_header(orig_cols, headers)
            rows = [{nc: r[oc] for nc, oc in zip(new_cols, orig_cols)} for r in data]
            dsl = synthesize(new_cols, rows)
            cond_id = f"k000__{bp.__dict__.get('HEADER_TAG', None) or ''}"
            htag = {"meaningful": "hm", "cryptic": "hc", "none": "hn"}[headers]
            cond_id = f"k000__{htag}"
            rid = f"{ds}__baseline__{cond_id}__t00__r1"
            record = {
                "run_id": rid,
                "provider": "baseline",
                "model_requested": "deterministic-profiling-v1",
                "model_returned": "deterministic-profiling-v1",
                "dataset": ds, "cond_id": cond_id,
                "domain": False, "dictionary": False, "rules": False,
                "headers": headers,
                "temperature": 0.0, "replicate": 1, "seed": None,
                "top_p": None, "max_tokens": None,
                "prompt_sha256": None,
                "prompt_file": f"datasets/samples/{ds}_sample.csv (header={headers})",
                "response_text": dsl,
                "finish_reason": "deterministic",
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "thoughts_token_count": None,
                "request_payload": json.dumps({"algorithm": "profiling-v1",
                                               "headers": headers}),
                "sdk_versions": {"python": ".".join(map(str, sys.version_info[:3])),
                                 "provider_sdk": "none (local)"},
                "ecologits": None,
                "ecologits_unavailable_reason": "no API call (deterministic)",
                "latency_ms": 0,
                "transport_retries": 0,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            }
            path = os.path.join(RAW_DIR, rid + ".json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
            written += 1
            print(f"wrote {rid} ({len(dsl.splitlines())} DSL lines)")
    print(f"\n{written} baseline records")


if __name__ == "__main__":
    run()
