#!/usr/bin/env python3
"""Offline validation of the scoring engine (no API calls).

Run this before spending any API budget:
if a ground-truth model is fed back as a candidate, every dimension must score
Precision = Recall = F1 = 1.0. A second test exercises the CSV-column alias
bridge; a third injects known errors and checks the TP/FP/FN accounting.
"""

import sys
import erdbench as eb


def emit_perfect_dsl(gt, name_map=None):
    """Render a GT model as candidate DSL. name_map optionally rewrites
    attribute names to the CSV columns a model would emit (to exercise aliases).
    """
    name_map = name_map or {}
    lines = []
    for e in gt["entities"]:
        lines.append(f"ENTITY {e['name']}")
        pk = [name_map.get(f"{e['name']}.{c}", c) for c in e["pk"]]
        lines.append("PK " + ", ".join(pk))
        for a in e["attrs"]:
            col = name_map.get(f"{e['name']}.{a['name']}", a["name"])
            flags = ""
            if a["is_pk"]:
                flags += " : PK"
            if a["is_fk"]:
                flags += " : FK"
            lines.append(f"ATTR {col} : varchar{flags}")
    for r in gt["relationships"]:
        fk = r["fk_cols"][0] if r["fk_cols"] else ""
        tail = f" : FK={fk}" if fk else ""
        lines.append(f"REL {r['child']} -> {r['parent']} : 1..N -> 1{tail}")
    return "\n".join(lines)


def all_perfect(result):
    for dim in ("entities", "relationships", "keys", "attributes"):
        m = result[dim]
        if not (m["precision"] == m["recall"] == m["f1"] == 1.0):
            return False, dim, m
    return True, None, None


def check(cond, msg):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {msg}")
    return cond


def main():
    ok = True
    for ds in ("airlines", "manufacturing"):
        gt = eb.load_gt(ds)
        print(f"\n=== {ds} ===")

        # Test 1: GT as a perfect candidate, GT's own attribute names.
        cand = eb.parse_candidate(emit_perfect_dsl(gt))
        res = eb.evaluate(cand, gt)
        good, dim, m = all_perfect(res)
        ok &= check(good, f"GT-as-candidate scores 1.0 on all dimensions"
                          + ("" if good else f" (broke on {dim}: {m})"))

        # Test 2: same, but attributes renamed to their CSV columns (aliases).
        aliases = eb.load_aliases(ds)
        name_map = {}
        for key, alist in aliases.get("attributes", {}).items():
            name_map[key] = alist[0]        # use the first bridged column name
        cand2 = eb.parse_candidate(emit_perfect_dsl(gt, name_map))
        res2 = eb.evaluate(cand2, gt)
        good2, dim2, m2 = all_perfect(res2)
        ok &= check(good2, "CSV-column-named candidate still scores 1.0 (alias bridge)"
                           + ("" if good2 else f" (broke on {dim2}: {m2})"))

    # Test 3: injected-error accounting on a tiny hand-built case.
    print("\n=== error accounting ===")
    gt = eb.load_gt("manufacturing")
    n_ent = len(gt["entities"])
    n_rel = len(gt["relationships"])
    # Drop one entity, add one hallucinated entity.
    dsl = emit_perfect_dsl(gt)
    dsl_lines = dsl.splitlines()
    # remove the WorkCenter entity block (ENTITY..next ENTITY) and add a fake one
    out, skip = [], False
    for ln in dsl_lines:
        if ln.startswith("ENTITY WorkCenter"):
            skip = True; continue
        if skip and ln.startswith("ENTITY "):
            skip = False
        if not skip:
            out.append(ln)
    out.append("ENTITY Ghost")
    out.append("PK ghost_id")
    out.append("ATTR ghost_id : integer : PK")
    res = eb.evaluate(eb.parse_candidate("\n".join(out)), gt)
    e = res["entities"]
    ok &= check(e["tp"] == n_ent - 1 and e["fp"] == 1 and e["fn"] == 1,
                f"drop WorkCenter + add Ghost -> entities TP={e['tp']} FP={e['fp']} FN={e['fn']} "
                f"(expected TP={n_ent-1} FP=1 FN=1)")
    # relationships touching WorkCenter (1 of them) become FN
    r = res["relationships"]
    ok &= check(r["fn"] >= 1, f"WorkCenter relationship now missed -> rel FN={r['fn']} (>=1)")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
