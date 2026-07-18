#!/usr/bin/env python3
"""Core library for the LLM-ERD benchmark: canonicalization, DSL parsing,
ground-truth loading, and deterministic scoring.

No LLM and no human is in the scoring loop: given a candidate model and a
ground-truth model, the metrics are a pure function of the inputs.
"""

import json
import os
import re
import unicodedata

# --------------------------------------------------------------------------
# Canonicalization
# --------------------------------------------------------------------------

IRREGULAR_SINGULARS = {
    "people": "person", "children": "child", "men": "man", "women": "woman",
    "data": "datum", "indices": "index", "statuses": "status",
    "aircraft": "aircraft", "series": "series",
}


def _singularize(token):
    """Naive English singularization of one lowercased token."""
    if token in IRREGULAR_SINGULARS:
        return IRREGULAR_SINGULARS[token]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ses"):
        return token[:-2]        # statuses handled above; addresses -> address
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _tokenize(name):
    name = unicodedata.normalize("NFKC", name).strip()
    name = _CAMEL_RE.sub(" ", name)
    parts = re.split(r"[\s_\-./]+", name)
    return [p.lower() for p in parts if p]


def canon(name):
    """Canonical key of an entity/attribute/column name.

    Tokenize on separators and camelCase, lowercase, singularize the final
    token, then join and strip residual non-alphanumerics.
    """
    tokens = _tokenize(name)
    if not tokens:
        return ""
    tokens[-1] = _singularize(tokens[-1])
    joined = "".join(tokens)
    return re.sub(r"[^a-z0-9]", "", joined)


# --------------------------------------------------------------------------
# Cardinality classification
# --------------------------------------------------------------------------

_MANY = {"n", "0..n", "1..n", "*", "0..*", "1..*", "m", "many"}
_ONE = {"1", "0..1", "one"}


def _mult(card):
    c = card.strip().lower().replace(" ", "")
    if c in _MANY:
        return "many"
    if c in _ONE:
        return "one"
    return "one"  # default conservative


def klass_from_pair(parent_mult, child_mult):
    """(#parents per child, #children per parent) -> relationship class."""
    if parent_mult == "one" and child_mult == "many":
        return "1:N"
    if parent_mult == "one" and child_mult == "one":
        return "1:1"
    if parent_mult == "many" and child_mult == "many":
        return "M:N"
    return "N:1"  # inverted; will not match a GT 1:N with the same parent/child


# --------------------------------------------------------------------------
# DSL parsing
# --------------------------------------------------------------------------

_ARROWS = {"→": "->", "⇒": "->", "⟶": "->", "➔": "->"}
_QUOTES = {"‘": "'", "’": "'", "“": '"', "”": '"'}
_KEYWORDS = ("ENTITY", "PK", "ATTR", "REL")

_REL_RE = re.compile(
    r"^REL\s+(.+?)\s*->\s*(.+?)\s*:\s*(\S+)\s*->\s*(\S+)"
    r"(?:\s*:\s*FK\s*=\s*([^:]+?))?\s*$",
    re.IGNORECASE,
)


def _prepass(text):
    """Deterministic normalization before line parsing.

    Note: no document-level line de-duplication happens here. Identical ATTR/PK
    lines legitimately recur across different entities (e.g. `ATTR name`), so
    de-duplication is done semantically in the parser — ATTR names within one
    entity, and REL tuples globally.
    """
    for a, b in {**_ARROWS, **_QUOTES}.items():
        text = text.replace(a, b)
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("```") or line.startswith("~~~"):
            continue
        line = re.sub(r"^[-*+•]\s+", "", line)           # list markers
        line = re.sub(r"^\d+[.)]\s+", "", line)                # numbered lists
        line = line.replace("**", "").replace("__", "")        # emphasis
        # uppercase a leading lowercased keyword
        m = re.match(r"^([A-Za-z]+)\b", line)
        if m and m.group(1).upper() in _KEYWORDS:
            line = line[:m.start(1)] + m.group(1).upper() + line[m.end(1):]
        out.append(line)
    return out


def parse_candidate(text):
    """Parse candidate DSL text into a normalized model dict.

    Returns {entities: [...], relationships: [...], parse: {...}} where each
    entity is {name, pk:[cols], attrs:[{name, is_pk, is_fk}]} and each
    relationship is {frm, to, cf, ct, fk}. Unparseable lines are counted.
    """
    lines = _prepass(text)
    entities, rels = [], []
    by_name = {}
    cur = None
    parsed = rejected = 0
    rejected_lines = []

    for line in lines:
        upper = line.upper()
        if upper.startswith("ENTITY "):
            name = line[len("ENTITY "):].strip().strip('"').strip()
            if not name:
                rejected += 1; rejected_lines.append(line); continue
            if canon(name) in by_name:
                cur = by_name[canon(name)]           # merge duplicate entity block
            else:
                cur = {"name": name, "pk": [], "attrs": [], "_attr_canon": set()}
                entities.append(cur)
                by_name[canon(name)] = cur
            parsed += 1
        elif upper.startswith("PK "):
            cols = [c.strip() for c in line[3:].split(",") if c.strip()]
            if cur is not None and cols:
                cur["pk"] = cols
                parsed += 1
            else:
                rejected += 1; rejected_lines.append(line)
        elif upper.startswith("ATTR "):
            body = line[5:].strip()
            parts = [p.strip() for p in body.split(":")]
            colname = parts[0]
            flags = {p.upper() for p in parts[2:]} if len(parts) > 2 else set()
            if cur is None or not colname:
                rejected += 1; rejected_lines.append(line); continue
            ck = canon(colname)
            if ck not in cur["_attr_canon"]:
                cur["_attr_canon"].add(ck)
                cur["attrs"].append({
                    "name": colname,
                    "is_pk": "PK" in flags,
                    "is_fk": "FK" in flags,
                })
            parsed += 1
        elif upper.startswith("REL"):
            m = _REL_RE.match(line)
            if not m:
                rejected += 1; rejected_lines.append(line); continue
            frm, to, cf, ct, fk = m.groups()
            fk_str = (fk or "").strip()
            rels.append({"frm": frm.strip(), "to": to.strip(),
                         "cf": cf, "ct": ct, "fk": fk_str,
                         "fk_cols": [canon(c) for c in fk_str.split(",") if c.strip()]})
            parsed += 1
        else:
            rejected += 1
            rejected_lines.append(line)

    # Collapse duplicate REL tuples (same endpoints, cardinalities, FK column).
    seen_rel, deduped = set(), []
    for r in rels:
        key = (canon(r["frm"]), canon(r["to"]), _mult(r["cf"]), _mult(r["ct"]), canon(r["fk"]))
        if key not in seen_rel:
            seen_rel.add(key)
            deduped.append(r)
    rels = deduped

    # PK line overrides ATTR :PK flags; if no PK line, derive from flags.
    for e in entities:
        if not e["pk"]:
            e["pk"] = [a["name"] for a in e["attrs"] if a["is_pk"]]
        e.pop("_attr_canon", None)

    # FK columns referenced by REL lines mark the From-entity attribute as FK.
    for r in rels:
        if not r["fk"]:
            continue
        child = by_name.get(canon(r["frm"]))
        if child:
            fk_set = set(r.get("fk_cols") or [canon(r["fk"])])
            for a in child["attrs"]:
                if canon(a["name"]) in fk_set:
                    a["is_fk"] = True

    return {
        "entities": entities,
        "relationships": rels,
        "parse": {
            "lines_total": len(lines),
            "lines_parsed": parsed,
            "lines_rejected": rejected,
            "rejected_lines": rejected_lines,
            "parse_failure": len(entities) == 0,
        },
    }


# --------------------------------------------------------------------------
# Ground-truth loading
# --------------------------------------------------------------------------

def load_ground_truth(path, aliases=None):
    """Load a GT JSON into the same normalized shape used for candidates,
    plus alias sets and candidate keys for scoring."""
    doc = json.load(open(path, encoding="utf-8"))
    aliases = aliases or {}
    ent_aliases = aliases.get("entities", {})
    attr_aliases = aliases.get("attributes", {})

    entities = []
    for e in doc["entities"]:
        pk = list(e.get("primaryKey", []))
        cks = [list(ck) for ck in e.get("candidateKeys", [])]
        attrs = []
        for a in e["attributes"]:
            is_fk = a.get("semanticType") == "ForeignKey"
            key = f"{e['name']}.{a['name']}"
            attrs.append({
                "name": a["name"],
                "is_pk": a["name"] in pk,
                "is_fk": is_fk,
                "alias_canon": {canon(x) for x in attr_aliases.get(key, [])},
            })
        entities.append({
            "name": e["name"],
            "pk": pk,
            "candidate_keys": cks,
            "attrs": attrs,
            "alias_canon": {canon(x) for x in ent_aliases.get(e["name"], [])},
        })

    rels = []
    for r in doc["relationships"]:
        card = r["cardinality"]
        pm = _mult(card["source"])   # source is the parent
        cm = _mult(card["target"])
        rels.append({
            "parent": r["source"],
            "child": r["target"],
            "klass": klass_from_pair(pm, cm),
            "role": r.get("role", ""),
            "fk_cols": [canon(c) for c in r.get("foreignKey", {}).get("attributes", [])],
        })
    return {"entities": entities, "relationships": rels}


# --------------------------------------------------------------------------
# Matching and metrics
# --------------------------------------------------------------------------

def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def _gt_entity_keyset(g):
    return {canon(g["name"])} | g["alias_canon"]


def match_entities(cand, gt):
    """One-to-one bipartite match of candidate to GT entities by canon/alias.

    Deterministic: candidate entities in declaration order claim the first
    unclaimed GT entity whose canonical/alias set contains the candidate canon.
    Returns (match: {cand_idx: gt_idx}, tp, fp_idx, fn_idx).
    """
    match = {}
    used_gt = set()
    for ci, ce in enumerate(cand["entities"]):
        cc = canon(ce["name"])
        for gi, ge in enumerate(gt["entities"]):
            if gi in used_gt:
                continue
            if cc in _gt_entity_keyset(ge):
                match[ci] = gi
                used_gt.add(gi)
                break
    fp = [ci for ci in range(len(cand["entities"])) if ci not in match]
    fn = [gi for gi in range(len(gt["entities"])) if gi not in used_gt]
    return match, fp, fn


def score_entities(cand, gt, match, fp, fn):
    return _prf(len(match), len(fp), len(fn))


def _attr_match(cand_attr_name, gt_attr):
    cc = canon(cand_attr_name)
    return cc == canon(gt_attr["name"]) or cc in gt_attr["alias_canon"]


def score_attributes(cand, gt, match):
    """D4 over non-FK attributes only, entity-scoped."""
    tp = fp = fn = 0
    gt_for_cand = {ci: gt["entities"][gi] for ci, gi in match.items()}
    matched_gt = set(match.values())

    for ci, ce in enumerate(cand["entities"]):
        ge = gt_for_cand.get(ci)
        cand_attrs = [a for a in ce["attrs"] if not a["is_fk"]]
        if ge is None:
            fp += len(cand_attrs)          # attrs of an unmatched entity
            continue
        gt_attrs = [a for a in ge["attrs"] if not a["is_fk"]]
        used = set()
        for a in cand_attrs:
            hit = None
            for gj, ga in enumerate(gt_attrs):
                if gj in used:
                    continue
                if _attr_match(a["name"], ga):
                    hit = gj; break
            if hit is not None:
                used.add(hit); tp += 1
            else:
                fp += 1
        fn += len(gt_attrs) - len(used)

    for gi, ge in enumerate(gt["entities"]):
        if gi not in matched_gt:
            fn += len([a for a in ge["attrs"] if not a["is_fk"]])
    return _prf(tp, fp, fn)


def _accepted_names(ge, colname):
    """Canonical names accepted for a GT column of entity ge: its own canon
    plus any attribute aliases (CSV-column bridge and synonyms)."""
    c = canon(colname)
    for a in ge["attrs"]:
        if canon(a["name"]) == c:
            return {c} | a["alias_canon"]
    return {c}


def _key_set_matches(cand_cols, gt_cols, ge):
    """True iff the candidate column set corresponds one-to-one to the GT key
    columns, each GT column matched by its own name or an alias."""
    if len(cand_cols) != len(gt_cols):
        return False
    cc = [canon(x) for x in cand_cols]
    used = set()
    for gcol in gt_cols:
        acc = _accepted_names(ge, gcol)
        hit = next((j for j, x in enumerate(cc) if j not in used and x in acc), None)
        if hit is None:
            return False
        used.add(hit)
    return True


def score_keys(cand, gt, match):
    """D3: PK + FK micro-averaged, plus PK-only and FK-only breakdowns."""
    gt_for_cand = {ci: gt["entities"][gi] for ci, gi in match.items()}
    matched_gt = set(match.values())

    # ---- PK ----
    pk_tp = pk_fp = pk_fn = 0
    for ci, ce in enumerate(cand["entities"]):
        ge = gt_for_cand.get(ci)
        cand_pk = ce["pk"]
        if ge is None:
            if cand_pk:
                pk_fp += 1                 # PK on an unmatched entity
            continue
        accepted_sets = [ge["pk"]] + ge["candidate_keys"]
        if cand_pk and any(_key_set_matches(cand_pk, ks, ge) for ks in accepted_sets):
            pk_tp += 1
        else:
            pk_fn += 1
            if cand_pk:
                pk_fp += 1                 # wrong PK: both a miss and a false key
    for gi in range(len(gt["entities"])):
        if gi not in matched_gt:
            pk_fn += 1

    # ---- FK ----
    fk_tp = fk_fp = fk_fn = 0
    for ci, ce in enumerate(cand["entities"]):
        ge = gt_for_cand.get(ci)
        cand_fk = [a for a in ce["attrs"] if a["is_fk"]]
        if ge is None:
            fk_fp += len(cand_fk)
            continue
        gt_fk = [a for a in ge["attrs"] if a["is_fk"]]
        used = set()
        for a in cand_fk:
            hit = None
            for gj, ga in enumerate(gt_fk):
                if gj in used:
                    continue
                if _attr_match(a["name"], ga):
                    hit = gj; break
            if hit is not None:
                used.add(hit); fk_tp += 1
            else:
                fk_fp += 1
        fk_fn += len(gt_fk) - len(used)
    for gi, ge in enumerate(gt["entities"]):
        if gi not in matched_gt:
            fk_fn += len([a for a in ge["attrs"] if a["is_fk"]])

    combined = _prf(pk_tp + fk_tp, pk_fp + fk_fp, pk_fn + fk_fn)
    combined["pk_only"] = _prf(pk_tp, pk_fp, pk_fn)
    combined["fk_only"] = _prf(fk_tp, fk_fp, fk_fn)
    return combined


def _cand_rel_tuple(r, cand, gt, cand_to_gt_canon):
    """Canonical (parent_canon, child_canon, klass, fk_canon) for a candidate
    relationship, with endpoints translated to GT canon via the D1 match.
    Returns None if either endpoint is not matched to a GT entity."""
    parent_c = canon(r["to"])     # To = parent (referenced side)
    child_c = canon(r["frm"])     # From = child (FK holder)
    gp = cand_to_gt_canon.get(parent_c)
    gc = cand_to_gt_canon.get(child_c)
    if gp is None or gc is None:
        return None
    parent_mult = _mult(r["ct"])  # #parents (To) per one child (From)
    child_mult = _mult(r["cf"])   # #children (From) per one parent (To)
    return (gp, gc, klass_from_pair(parent_mult, child_mult), canon(r["fk"]))


def score_relationships(cand, gt, match):
    """D2: endpoints must be D1-matched; match on (parent, child, class),
    disambiguating same-triple GT pairs (roles) by FK column."""
    # candidate canon -> GT canon (primary name) for matched entities
    cand_to_gt = {}
    for ci, gi in match.items():
        cand_to_gt[canon(cand["entities"][ci]["name"])] = canon(gt["entities"][gi]["name"])

    # GT relationships grouped by (parent, child, klass)
    gt_groups = {}
    for r in gt["relationships"]:
        key = (canon(r["parent"]), canon(r["child"]), r["klass"])
        gt_groups.setdefault(key, []).append(r)
    gt_claimed = {id(r): False for grp in gt_groups.values() for r in grp}

    tp = fp = 0
    for r in cand["relationships"]:
        t = _cand_rel_tuple(r, cand, gt, cand_to_gt)
        if t is None:
            fp += 1; continue
        key = (t[0], t[1], t[2])
        grp = gt_groups.get(key)
        if not grp:
            fp += 1; continue
        # prefer an unclaimed GT rel with matching FK column, else any unclaimed
        pick = None
        for r_gt in grp:
            if not gt_claimed[id(r_gt)] and t[3] and t[3] in r_gt["fk_cols"]:
                pick = r_gt; break
        if pick is None:
            for r_gt in grp:
                if not gt_claimed[id(r_gt)]:
                    pick = r_gt; break
        if pick is None:
            fp += 1                       # right triple but all GT rels claimed
        else:
            gt_claimed[id(pick)] = True; tp += 1
    fn = sum(1 for v in gt_claimed.values() if not v)
    return _prf(tp, fp, fn)


def evaluate(cand, gt):
    """Full per-run evaluation across the four dimensions."""
    match, fp_e, fn_e = match_entities(cand, gt)
    return {
        "entities": score_entities(cand, gt, match, fp_e, fn_e),
        "relationships": score_relationships(cand, gt, match),
        "keys": score_keys(cand, gt, match),
        "attributes": score_attributes(cand, gt, match),
        "parse": cand.get("parse", {}),
    }


# --------------------------------------------------------------------------
# Paths / helpers
# --------------------------------------------------------------------------

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def load_aliases(dataset):
    path = os.path.join(BASE, "ground-truth-models", "aliases.json")
    if not os.path.exists(path):
        return {}
    return json.load(open(path, encoding="utf-8")).get(dataset, {})


GT_FILES = {
    "airlines": "airlines_ground_truth_model.json",
    "manufacturing": "manuf_ground_truth_model.json",
}


def load_gt(dataset):
    return load_ground_truth(
        os.path.join(BASE, "ground-truth-models", GT_FILES[dataset]),
        load_aliases(dataset),
    )
