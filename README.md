# ERDBench

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21434806.svg)](https://doi.org/10.5281/zenodo.21434806)

Replication package for the manuscript *"Evaluating Large Language Models in Conceptual Modeling: an Experimental Design"* (under revision at Data & Knowledge Engineering).

ERDBench evaluates two large language models (Mistral Small, Gemini 2.5 Flash) and a deterministic, no-LLM profiling baseline at generating Entity-Relationship models from denormalized CSV data, under a full factorial design: three expert-knowledge factors (domain description, data dictionary, business rules; 2^3 = 8 combinations) crossed with three levels of attribute-name informativeness (meaningful, cryptic, none), on two datasets (airline, manufacturing). This yields 24 conditions per dataset; with 3 temperatures (0, 0.2, 0.5) and 3 replicates, 432 executions per model, 864 LLM executions in total, plus 6 baseline evaluations: 870 evaluated models. The evaluation is fully automated and deterministic: no human judgment and no LLM participates in the scoring.

## Layout

| Path | Contents |
|---|---|
| `config/experiment.json` | Frozen experiment configuration: design, models, temperatures, seed policy, prompt version |
| `datasets/` | The two 1000-row CSVs produced by the seeded generator, with SHA-256 checksums; `samples/` holds the frozen prompt samples, their checksums, and the representativeness report |
| `ground-truth-models/` | Reference models curated by domain experts; `aliases.json` (frozen synonym and accepted-key lists); `data_binding.json` (column-to-model binding) |
| `inputs/` | The expert knowledge blocks per dataset: domain description, attribute-level data dictionary, business rules |
| `prompts/` | `initial_prompt.txt` (base prompt, v2.1) and `rendered/` (the 48 condition prompts exactly as submitted) |
| `schemas/` | JSON Schema of the reference-model format |
| `scripts/` | The complete pipeline (generation, audit, sampling, prompt rendering, execution, parsing, scoring, aggregation) |
| `results/raw/` | One JSON per execution: the verbatim model response plus full request metadata (resolved model version, SDK versions, seed, prompt SHA-256, token usage including thinking tokens, latency) |
| `results/parsed/` | The parsed model of every execution |
| `results/metrics/` | Per-execution TP/FP/FN and precision/recall/F1 for the four dimensions (entities, relationships, keys, attributes) |
| `results/aggregate/` | `summary.json`, `main_effects.json` (per-factor deltas), `stability.json`, `token_usage.json` |
| `docs/SAMPLING.md` | Specification and rationale of the coverage-driven sampling |

## Reproduce the results (no API keys required)

Everything downstream of the LLM calls is deterministic and recomputable from the committed raw responses. With Python 3.13:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r scripts/requirements.txt

# scoring self-test (evaluator against known cases)
python scripts/selftest_scoring.py

# regenerate both datasets byte for byte and verify the pinned checksums
python scripts/generate_datasets.py
(cd datasets && shasum -a 256 -c CHECKSUMS.txt)

# audit: every reference-model element is derivable from the data
python scripts/verify_gt_data_link.py

# rebuild the frozen samples and the 48 rendered prompts (byte-identical)
python scripts/sample_datasets.py
(cd datasets/samples && shasum -a 256 -c CHECKSUMS.txt)
python scripts/build_prompts.py

# re-parse, re-score, and re-aggregate the committed responses
python scripts/parse_outputs.py
python scripts/evaluate.py
python scripts/aggregate.py

# deterministic baseline, endpoint-only relationship diagnostic, descriptive statistics
python scripts/deterministic_baseline.py
python scripts/relaxed_diagnostic.py
python scripts/stats_tests.py
```

Re-running this sequence reproduces every committed file byte for byte, with one documented exception: the six baseline records in `results/raw/` embed a `generated_at` timestamp, so their bytes change while their content (the emitted model, and therefore everything in `results/parsed/`, `results/metrics/`, and `results/aggregate/`) is identical.

## Re-run the LLM executions (API keys required)

```bash
cp .env.example .env   # fill in your Mistral and Gemini keys
python scripts/run_experiment.py --provider all
```

`--provider mock` performs a dry run without any API call. Provider-side nondeterminism means a re-run produces different responses (the paper quantifies this: even temperature 0 is not deterministic), so a re-run creates a new set of raw records; the committed ones are the ones reported in the paper.

## Provenance and auditability

- Every run record carries the SHA-256 of the exact prompt submitted, the resolved model version string, the SDK versions, the seed, and the token usage (including thinking tokens).
- The manufacturing dataset is a synthetic replica generated from the domain expert's reference model; no real production data is included.

## Mapping to the paper

- `results/aggregate/summary.json` — the per-condition cells behind the factorial and header-informativeness tables of the paper.
- `results/aggregate/main_effects.json` — the per-factor deltas; `stability.json` — the temperature and determinism analysis; `token_usage.json` — the token accounting.
- `results/metrics/` — the per-execution scores behind every reported number.

## License and citation

Released under the MIT License (see `LICENSE`). This repository is archived at Zenodo: [10.5281/zenodo.21434806](https://doi.org/10.5281/zenodo.21434806). If you use ERDBench, please cite the associated paper: `CITATION.cff` carries the reference (GitHub renders it as "Cite this repository"); the full article reference will be added upon publication.
