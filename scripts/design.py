#!/usr/bin/env python3
"""Design v2 enumeration — single source of truth for the experiment conditions.

Full factorial:
  knowledge factors: domain, dictionary, rules  (each on-off)  -> 2^3 = 8 combos
  header levels:      meaningful, cryptic, none                 -> 3 levels
  => 8 x 3 = 24 conditions, fully crossed.
With 2 models x 2 datasets x 3 temperatures x 3 replicates = 864 runs
(432 per model).
"""

HEADER_LEVELS = ["meaningful", "cryptic", "none"]
HEADER_TAG = {"meaningful": "hm", "cryptic": "hc", "none": "hn"}


def conditions():
    conds = []
    for d in (0, 1):
        for m in (0, 1):          # m = data dictionary (metadata)
            for r in (0, 1):
                for h in HEADER_LEVELS:
                    ktag = f"k{d}{m}{r}"
                    conds.append({
                        "domain": bool(d), "dictionary": bool(m), "rules": bool(r),
                        "headers": h, "ktag": ktag, "htag": HEADER_TAG[h],
                        "cond_id": f"{ktag}__{HEADER_TAG[h]}",
                    })
    return conds


if __name__ == "__main__":
    cs = conditions()
    for c in cs:
        print(c["cond_id"], "| domain=%d dictionary=%d rules=%d headers=%s"
              % (c["domain"], c["dictionary"], c["rules"], c["headers"]))
    print(f"\n{len(cs)} conditions")
