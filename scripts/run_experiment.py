#!/usr/bin/env python3
"""Execute the experiment matrix: one isolated one-shot API call per cell.

Matrix = models x datasets x knowledge-levels x temperatures x replicates
(config/experiment.json). Each run is a single stateless request; the verbatim
response and full metadata are written to results/raw/<run_id>.json. Idempotent:
a cell whose raw record already exists is skipped unless --fresh.

Providers:
  mock    -- offline stub (no API, no key); validates orchestration end-to-end.
  gemini  -- google-genai; thinking budget forced to 0.
  mistral -- mistralai.

Usage:
  python run_experiment.py --provider mock                 # dry-run whole matrix
  python run_experiment.py --provider gemini --pilot       # 2 pilot cells (T=0, business_rules)
  python run_experiment.py --provider all                  # full run
  python run_experiment.py --provider gemini --only airlines business_rules 0.0 1
"""

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CONFIG = json.load(open(os.path.join(BASE, "config", "experiment.json"), encoding="utf-8"))


def load_env():
    path = os.path.join(BASE, ".env")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def temp_tag(t):
    return f"{int(round(t * 10)):02d}"


def seed_for(temp, replicate):
    base = CONFIG["decoding"]["seed_base"]
    return 42 if temp == 0 else base + replicate


def run_id(dataset, model, cond_id, temp, rep):
    return f"{dataset}__{model}__{cond_id}__t{temp_tag(temp)}__r{rep}"


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def valid_record(path):
    if not os.path.exists(path):
        return False
    try:
        json.load(open(path, encoding="utf-8"))
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------

class MockProvider:
    key = "mock"
    provider = "mock"
    model_id = "mock-0"

    def sdk_version(self):
        return "mock"

    def generate(self, prompt, temperature, top_p, max_tokens, seed):
        # A tiny, syntactically valid model so the parse/score chain has input.
        text = ("ENTITY Placeholder\nPK id\nATTR id : integer : PK\n"
                "ATTR name : varchar\n")
        return {"text": text, "finish_reason": "stop",
                "input_tokens": len(prompt) // 4, "output_tokens": len(text) // 4,
                "thoughts_token_count": None, "model_returned": self.model_id,
                "request_payload": json.dumps({"provider": "mock", "seed": seed,
                                               "temperature": temperature})}


class GeminiProvider:
    key = "gemini"
    provider = "google"

    def __init__(self, cfg):
        from google import genai
        self._genai = genai
        from google.genai import types
        self._types = types
        self.model_id = os.environ.get(cfg["model_env"], cfg["default_model"])
        self.client = genai.Client(api_key=os.environ[cfg["key_env"]])
        self.thinking_budget = CONFIG["decoding"]["gemini_thinking_budget"]

    def sdk_version(self):
        import google.genai as g
        return f"google-genai {getattr(g, '__version__', '?')}"

    def generate(self, prompt, temperature, top_p, max_tokens, seed):
        types = self._types
        gen_cfg = types.GenerateContentConfig(
            temperature=temperature, top_p=top_p, max_output_tokens=max_tokens,
            seed=seed,
            thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
        )
        resp = self.client.models.generate_content(
            model=self.model_id, contents=prompt, config=gen_cfg)
        cand = (resp.candidates or [None])[0]
        finish = str(getattr(cand, "finish_reason", "")) if cand else "EMPTY"
        um = resp.usage_metadata
        return {
            "text": resp.text or "",
            "finish_reason": finish,
            "input_tokens": getattr(um, "prompt_token_count", None),
            "output_tokens": getattr(um, "candidates_token_count", None),
            "thoughts_token_count": getattr(um, "thoughts_token_count", None),
            "model_returned": getattr(resp, "model_version", self.model_id),
            "request_payload": json.dumps({
                "model": self.model_id, "temperature": temperature, "top_p": top_p,
                "max_output_tokens": max_tokens, "seed": seed,
                "thinking_budget": self.thinking_budget}),
        }


class MistralProvider:
    key = "mistral"
    provider = "mistral"

    def __init__(self, cfg):
        try:
            from mistralai import Mistral          # SDK v1 layout
        except ImportError:
            from mistralai.client import Mistral    # SDK v2 layout (>=2.x)
        self.model_id = os.environ.get(cfg["model_env"], cfg["default_model"])
        self.client = Mistral(api_key=os.environ[cfg["key_env"]])

    def sdk_version(self):
        import importlib.metadata as im
        try:
            return f"mistralai {im.version('mistralai')}"
        except Exception:
            return "mistralai ?"

    def generate(self, prompt, temperature, top_p, max_tokens, seed):
        resp = self.client.chat.complete(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature, top_p=top_p, max_tokens=max_tokens,
            random_seed=seed)
        choice = resp.choices[0]
        u = resp.usage
        return {
            "text": choice.message.content or "",
            "finish_reason": str(choice.finish_reason),
            "input_tokens": getattr(u, "prompt_tokens", None),
            "output_tokens": getattr(u, "completion_tokens", None),
            "thoughts_token_count": None,
            "model_returned": getattr(resp, "model", self.model_id),
            "request_payload": json.dumps({
                "model": self.model_id, "temperature": temperature, "top_p": top_p,
                "max_tokens": max_tokens, "random_seed": seed}),
        }


def make_provider(name):
    if name == "mock":
        return MockProvider()
    cfg = next(m for m in CONFIG["models"] if m["key"] == name)
    return GeminiProvider(cfg) if name == "gemini" else MistralProvider(cfg)


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------

def iter_cells(pilot=False, only=None):
    ds_list = CONFIG["datasets"]
    conds = design.conditions()
    temps = CONFIG["temperatures"]
    reps = CONFIG["replicates"]
    if pilot:
        conds = [c for c in conds if c["cond_id"] == "k111__hm"]  # full expert, meaningful
        temps, reps = [0.0], [1]
    for ds in ds_list:
        for cond in conds:
            for temp in temps:
                for rep in reps:
                    if only and not (ds == only[0] and cond["cond_id"] == only[1]
                                     and abs(temp - float(only[2])) < 1e-9
                                     and rep == int(only[3])):
                        continue
                    yield ds, cond, temp, rep


def call_with_retries(provider, prompt, temperature, top_p, max_tokens, seed):
    tr = CONFIG["transport"]
    retries = 0
    while True:
        try:
            return provider.generate(prompt, temperature, top_p, max_tokens, seed), retries
        except Exception as e:
            if retries >= tr["max_retries"]:
                raise
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate limit" in msg.lower():
                # Rate-limited: honor the provider's suggested delay when
                # present, else wait a full quota window (per-minute quotas).
                m = re.search(r"retry in (\d+(?:\.\d+)?)s", msg, re.IGNORECASE) or \
                    re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+)", msg)
                wait = min(90, float(m.group(1)) + 2) if m else 62
            else:
                wait = tr["backoff_base_seconds"] ** retries
            print(f"    transport error ({e.__class__.__name__}); "
                  f"retry {retries + 1}/{tr['max_retries']} in {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
            retries += 1


def run(provider_names, pilot, only, fresh):
    load_env()
    dec = CONFIG["decoding"]
    raw_dir = os.path.join(BASE, CONFIG["paths"]["results_raw"])
    rendered_dir = os.path.join(BASE, CONFIG["paths"]["prompts_rendered"])
    prefix = "pilot/" if pilot else ""

    providers = [make_provider(n) for n in provider_names]
    total = done = skipped = failed = 0
    last_mistral = 0.0
    last_gemini = 0.0
    # Gemini paid tier: 1M input tokens/min. Pace calls so estimated input
    # stays under ~850k/min (chars/3.5 is a conservative token estimate).
    GEMINI_TOKENS_PER_SEC = 850_000 / 60.0

    for prov in providers:
        for ds, cond, temp, rep in iter_cells(pilot, only):
            total += 1
            rid = run_id(ds, prov.key, cond["cond_id"], temp, rep)
            out_path = os.path.join(raw_dir, prefix + rid + ".json")
            if not fresh and valid_record(out_path):
                skipped += 1
                continue

            prompt = open(os.path.join(rendered_dir, f"{ds}__{cond['cond_id']}.txt"),
                          encoding="utf-8").read()
            seed = seed_for(temp, rep)

            if prov.provider == "mistral":
                gap = CONFIG["transport"]["mistral_min_interval_seconds"] - (time.time() - last_mistral)
                if gap > 0:
                    time.sleep(gap)
            elif prov.provider == "google":
                est_tokens = len(prompt) / 3.5
                min_interval = est_tokens / GEMINI_TOKENS_PER_SEC
                gap = min_interval - (time.time() - last_gemini)
                if gap > 0:
                    time.sleep(gap)

            t0 = time.time()
            try:
                result, retries = call_with_retries(
                    prov, prompt, temp, dec["top_p"], dec["max_tokens"], seed)
            except Exception as e:
                failed += 1
                print(f"  FAIL {rid}: {e}", file=sys.stderr)
                continue
            finally:
                if prov.provider == "mistral":
                    last_mistral = time.time()
                elif prov.provider == "google":
                    last_gemini = time.time()
            latency_ms = int((time.time() - t0) * 1000)

            record = {
                "run_id": rid,
                "provider": prov.provider,
                "model_requested": prov.model_id,
                "model_returned": result["model_returned"],
                "dataset": ds,
                "cond_id": cond["cond_id"],
                "domain": cond["domain"], "dictionary": cond["dictionary"],
                "rules": cond["rules"], "headers": cond["headers"],
                "temperature": temp, "replicate": rep, "seed": seed,
                "top_p": dec["top_p"], "max_tokens": dec["max_tokens"],
                "prompt_sha256": sha256(prompt),
                "prompt_file": f"{CONFIG['paths']['prompts_rendered']}/{ds}__{cond['cond_id']}.txt",
                "response_text": result["text"],
                "finish_reason": result["finish_reason"],
                "usage": {"input_tokens": result["input_tokens"],
                          "output_tokens": result["output_tokens"]},
                "thoughts_token_count": result["thoughts_token_count"],
                "request_payload": result["request_payload"],
                "sdk_versions": {"python": platform.python_version(),
                                 "provider_sdk": prov.sdk_version()},
                "ecologits": None,
                "ecologits_unavailable_reason": "not yet instrumented (pilot TODO)",
                "latency_ms": latency_ms,
                "transport_retries": retries,
            }
            atomic_write_json(out_path, record)
            done += 1
            print(f"  ok {rid}: finish={result['finish_reason']} "
                  f"out_tok={result['output_tokens']} thoughts={result['thoughts_token_count']} "
                  f"{latency_ms}ms")

    print(f"\n{done} run, {skipped} skipped, {failed} failed, {total} total cells")
    return failed == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", default="mock",
                    choices=["mock", "gemini", "mistral", "all"])
    ap.add_argument("--pilot", action="store_true",
                    help="only condition k111__hm (full expert, meaningful) @ T=0, rep 1")
    ap.add_argument("--only", nargs=4, metavar=("DATASET", "COND_ID", "TEMP", "REP"),
                    help="run a single cell, e.g. airlines k101__hc 0.0 1")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore existing raw records and re-query")
    args = ap.parse_args()

    names = ["gemini", "mistral"] if args.provider == "all" else [args.provider]
    ok = run(names, args.pilot, args.only, args.fresh)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
