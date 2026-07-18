#!/usr/bin/env python3
"""Render the design-v2 prompts: composable knowledge blocks x header level.

For each dataset and each of the 24 conditions (scripts/design.py) this composes
the base task/output-contract template with the à-la-carte knowledge blocks
(domain / dictionary / rules, each independently on-off) and a header-transformed
data block (meaningful / cryptic / none), and writes the exact text the model
will see to prompts/rendered/<dataset>__<cond_id>.txt (committed for audit).

Header transforms change ONLY the CSV header row (values are untouched):
  meaningful -- original column names.
  cryptic    -- opaque, deterministic per-column codes (coded enterprise schema).
  none       -- positional c1..cN (no attribute names at all).

Usage:
  python build_prompts.py            # render all conditions + token estimates
  python build_prompts.py --verify   # rebuild in memory, compare to committed
"""

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PROMPTS_DIR = os.path.join(BASE, "prompts")
RENDERED_DIR = os.path.join(PROMPTS_DIR, "rendered")
INPUTS_DIR = os.path.join(BASE, "inputs")
SAMPLES_DIR = os.path.join(BASE, "datasets", "samples")

DATASETS = ["airlines", "manufacturing"]
INPUT_PREFIX = {"airlines": "airlines", "manufacturing": "manuf"}

DOMAIN_BLOCK = ("DOMAIN CONTEXT (provided by a domain expert)\n"
                "This dataset is about {domain}.\n\n")
DICT_BLOCK = ("DATA DICTIONARY (provided by a domain expert)\n"
              "The meaning of each column is:\n{dictionary}\n\n")
RULES_BLOCK = ("BUSINESS RULES AND CONSTRAINTS (provided by a domain expert)\n"
               "The following rules and constraints govern the domain and must be "
               "reflected in the ERD:\n{rules}\n\n")


def _load(path):
    return json.load(open(path, encoding="utf-8"))


def knowledge_values(dataset):
    p = INPUT_PREFIX[dataset]
    return {
        "domain": _load(os.path.join(INPUTS_DIR, f"{p}_domain.json"))["domain"],
        "dictionary": _load(os.path.join(INPUTS_DIR, f"{p}_dictionary.json"))["dictionary"],
        "rules": "\n".join(f"- {r}" for r in
                           _load(os.path.join(INPUTS_DIR, f"{p}_business_rules.json"))["businessRules"]),
    }


def dictionary_block(dictionary, headers):
    """Render the data dictionary with its column keys transformed to match the
    header condition (so it is usable under cryptic/none headers too)."""
    cols = list(dictionary.keys())
    keys = transform_header(cols, headers)
    return "\n".join(f"- {k}: {dictionary[c]}" for k, c in zip(keys, cols))


def transform_header(header_cols, mode):
    if mode == "meaningful":
        return header_cols
    if mode == "none":
        return [f"c{i+1}" for i in range(len(header_cols))]
    if mode == "cryptic":
        return ["C_" + hashlib.sha256(c.encode("utf-8")).hexdigest()[:6].upper()
                for c in header_cols]
    raise ValueError(mode)


def data_block(dataset, headers):
    csv_text = open(os.path.join(SAMPLES_DIR, f"{dataset}_sample.csv"),
                    encoding="utf-8", newline="").read().rstrip("\n")
    lines = csv_text.split("\n")
    header_cols = lines[0].split(",")
    new_header = ",".join(transform_header(header_cols, headers))
    body = "\n".join(lines[1:])
    n_rows = len(lines) - 1
    note = (f"The following is a representative sample of {n_rows} rows "
            f"from a larger dataset of 1000 rows.\n\n")
    return note + new_header + "\n" + body


def render(dataset, cond):
    base = open(os.path.join(PROMPTS_DIR, "initial_prompt.txt"), encoding="utf-8").read()
    head, sep, tail = base.partition("OUTPUT FORMAT")
    kv = knowledge_values(dataset)
    blocks = ""
    if cond["domain"]:
        blocks += DOMAIN_BLOCK.format(domain=kv["domain"])
    if cond["dictionary"]:
        blocks += DICT_BLOCK.format(dictionary=dictionary_block(kv["dictionary"], cond["headers"]))
    if cond["rules"]:
        blocks += RULES_BLOCK.format(rules=kv["rules"])
    text = head + blocks + sep + tail
    text = text.replace("{{DATA_BLOCK}}", data_block(dataset, cond["headers"]))
    if "{{" in text:
        raise ValueError(f"unfilled placeholder in {dataset}/{cond['cond_id']}")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    conds = design.conditions()
    rendered = {}
    for ds in DATASETS:
        for cond in conds:
            rendered[f"{ds}__{cond['cond_id']}.txt"] = render(ds, cond)

    if args.verify:
        ok = True
        for name, text in rendered.items():
            path = os.path.join(RENDERED_DIR, name)
            if not os.path.exists(path) or open(path, encoding="utf-8").read() != text:
                print(f"MISMATCH {name}"); ok = False
            else:
                print(f"OK {name}")
        sys.exit(0 if ok else 1)

    os.makedirs(RENDERED_DIR, exist_ok=True)
    for name, text in sorted(rendered.items()):
        with open(os.path.join(RENDERED_DIR, name), "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {name}: {len(text)} chars (~{len(text)//4} tokens)")
    print(f"\n{len(rendered)} prompts rendered")


if __name__ == "__main__":
    main()
