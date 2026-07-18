# Deterministic Stratified Sampling of the Input Data

This document specifies how the data block embedded in each prompt is
constructed. It is the authoritative description of the sampling method.
Implementation: [`scripts/sample_datasets.py`](../scripts/sample_datasets.py).
Machine-readable audit of the result: [`datasets/samples/representativeness_report.json`](../datasets/samples/representativeness_report.json).

## 1. Motivation

The canonical datasets have 1000 data rows each. Embedding a full CSV in every
prompt is undesirable for three reasons: (i) token cost across the 144-run
matrix; (ii) provider rate/token limits on the free tier; and (iii) evidence
dilution — a long, highly repetitive table makes the structural signal an ERD
depends on (which columns co-vary, which repeat, which determine which) harder
to read, not easier.

The wrong way to shrink the data would be to take the first *N* rows or a
uniform random sample: either can drop whole categories, break the visibility
of a 1:N relationship, or hide a functional dependency, so that a missing ERD
element would then reflect the *sample*, not the model under test.

We therefore use a **group-preserving, coverage-driven, fully deterministic**
sample: the smallest set of *whole* natural row-groups that still exhibits
every value of every low-cardinality column at least twice.

## 2. Principles

1. **Group-preserving.** The sampling unit is the natural row group, never an
   individual row. For airlines the group is one flight
   (`flight_number`, `flight_date`); for manufacturing it is one order
   (`n_order`). Keeping groups intact is what preserves 1:N cardinality
   evidence (one flight → many seat/crew/status rows; one order → several
   operations) and the within-group functional dependencies that reveal
   entities (e.g., `airline_code → airline_name`).

2. **Coverage-driven.** A value that appears only once in the sample is an
   isolated fact; a value that appears **twice or more**, across *different*
   groups, is evidence of a shared reference — i.e. of a relationship or a
   dependency. The sample is grown until every distinct value of every
   low-cardinality column is present at least `MIN_OCC = 2` times (or as many
   times as it occurs in the full data, if that is fewer than 2). This is what
   makes statements like "an airplane makes many flights" or "a client places
   many orders" *observable in the sample*.

3. **Fully deterministic.** Greedy selection with lexicographic tie-breaking on
   the group key; no random number generator anywhere. Re-running the script on
   the same canonical CSV reproduces the sample byte-for-byte. The samples and
   their SHA-256 checksums are committed under `datasets/samples/`.

4. **Frozen and constant across the experiment.** One sample per dataset is
   built once, before any execution, and every one of the 144 runs for that dataset
   embeds the *same* sample. Sampling is therefore a constant of the protocol,
   not a factor — it cannot confound the model, temperature, or knowledge-level
   comparisons. Any blind spot in a sample affects all conditions equally.

5. **No ground-truth peeking.** The strata are defined purely from observable
   dataset statistics (column cardinalities and repeated values). The group
   keys and the cardinality threshold are declared sampling parameters, fixed
   before any run; the ground-truth model is never consulted during sampling.

## 3. Algorithm

Let a dataset have rows `R` with columns `C`.

**Coverage targets.** A column `c ∈ C` is a *coverage target* iff it has at
most `MAX_CARD = 50` distinct values over `R`. High-cardinality columns
(identifiers, timestamps, free dates) are excluded — requiring each of their
values twice would force the whole dataset and defeat the purpose.

**Requirement function.** For each coverage target `c` and each value `v` it
takes, the sample must contain at least

```
required(c, v) = min(MIN_OCC, full_count(c, v))       with MIN_OCC = 2
```

occurrences of `v` in column `c`. Capping at `full_count` keeps genuinely
unique values (which legitimately occur once) from making the requirement
unsatisfiable.

**Groups.** Rows are partitioned into groups by the group key `G` (flight or
order). Selecting a group contributes *all* of its rows' values at once.

**Greedy set-cover.** Starting from an empty selection, repeatedly add the
group whose *marginal gain* — the sum over its `(column, value)` contributions
of the still-unmet portion of `required` — is largest. Ties are broken by the
lexicographically smallest group key, making the outcome deterministic. Stop
when every requirement is met. The emitted sample contains the rows of the
selected groups, in original file order (preserving the header).

Greedy set-cover does not guarantee the theoretically minimum number of groups,
but it is deterministic, near-optimal in practice, and — most importantly —
auditable: the representativeness report records exactly what was and was not
covered.

## 4. Parameters (frozen before any execution)

| Parameter | Value | Meaning |
|---|---|---|
| `MAX_CARD` | 50 | a column is a coverage target iff it has ≤ 50 distinct values |
| `MIN_OCC` | 2 | each distinct value must appear ≥ 2× in the sample (or `full_count` if fewer) |
| group key (airlines) | `flight_number`, `flight_date` | one flight = one sampling unit |
| group key (manufacturing) | `n_order` | one order = one sampling unit |

These live in `scripts/sample_datasets.py` and are mirrored into
`config/experiment.json` at freeze.

## 5. Result on the canonical datasets

Both samples achieve **100 % distinct-value coverage** of every coverage-target
column and satisfy the ≥2× requirement with **no gaps**.

| Dataset | Sampled | Groups | Coverage cols | ≈ tokens | Distinct coverage | ≥2× gaps |
|---|---|---|---|---|---|---|
| Airlines | 412 / 1000 rows | 33 / 80 flights | 20 | ~18 600 | 100 % | none |
| Manufacturing | 160 / 1000 rows | 40 / 250 orders | 8 | ~3 600 | 100 % | none |

**Coverage-target columns.**
- *Airlines (20):* `flight_number` (20 distinct), `seat_number` (30),
  `airline_code`/`airline_name` (10), `airport_departure`/`departure_name`/
  `airport_arrival`/`arrival_name` (15), `departure_location`/`arrival_location`/
  `departure_city`/`arrival_city` (10), `belt_number` (30), `license_number`
  (25), `airplane_model` (5), `status_code`/`status` (5), `status_hour` (12),
  `crew_license` (40), `crew_role` (4).
- *Manufacturing (8):* `client_id`/`client_name` (40), `n_material`/
  `description_material` (25), `n_operation`/`description_operation` (4),
  `workcenter_id` (8), `quantity` (8).

**Excluded (high-cardinality > 50).** Airlines: `passport`, `customer_name`,
`customer_surname` (250 each), `flight_date`, `status_date` (80). Manufacturing:
`n_order` (250), `date` (125), `timestamp` (500), `order_operation_id` (1000).
These are still present in the sample through the selected groups — they are
simply not *coverage drivers*.

The manufacturing sample size is driven by its most demanding stratum, the 40
clients each required ≥2× (→ ≥ 80 orders' worth of evidence, realized as 40
whole orders of 4 rows). The airlines size is driven jointly by the 30 belts,
25 airplanes, and 40 crew members.

## 6. Reproducing / verifying

```bash
python scripts/sample_datasets.py            # (re)build samples + report + checksums
python scripts/sample_datasets.py --verify   # rebuild in memory, compare to committed files
```

`--verify` exits non-zero on any byte-level mismatch and is part of the pre-execution
freeze gate. The prompt's data-block header states explicitly that the block is
*"a representative sample of N rows from a larger dataset of M rows"*, so a
model does not mistake sample boundaries (row counts) for population facts.

## 7. Limitations

- The sample is identical for every run, so a value it happens to omit is
  omitted for all models and temperatures alike — a controlled limitation, not
  a per-run confound.
- ≥2× is a heuristic threshold for "observable multiplicity", not a proof that
  every relationship is inferable; the ground-truth datasets are additionally
  engineered so that every GT key and cardinality is forced by the full data
  (enforced by `scripts/generate_datasets.py` and verified by `scripts/verify_gt_data_link.py`), and the sample inherits that
  structure at the group level.
- If a future provider's context budget could not fit a sample, the documented
  fallback (relax `MIN_OCC` to 1 for the single highest-cardinality stratum) is
  recorded in the representativeness report before any run —
  it is not currently needed (both samples fit comfortably in both providers'
  windows).
