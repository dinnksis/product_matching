"""Run resumable, label-free Qwen mapping of attribute-name batches."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import time
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCHES = ROOT / "data" / "qwen_attribute_ontology_v1" / "ontology_batches.jsonl"
DEFAULT_ENTRIES = ROOT / "data" / "qwen_attribute_ontology_v1" / "ontology_entries.parquet"
DEFAULT_PROMPT = ROOT / "prompts" / "qwen_attribute_ontology_v1.md"
DEFAULT_SCHEMA = ROOT / "schemas" / "qwen_attribute_ontology_v1.schema.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "qwen_attribute_ontology_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map raw attribute names in compact batches.")
    parser.add_argument("--batches", type=Path, default=DEFAULT_BATCHES)
    parser.add_argument("--entries", type=Path, default=DEFAULT_ENTRIES)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--api-base", default="http://localhost:8194/v1")
    parser.add_argument("--model", default="qwen3.5-397b-a17b-fp8")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means all batches")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def latest_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return result
    for row in read_jsonl(path):
        if row.get("batch_id"):
            result[str(row["batch_id"])] = row
    return result


def extract_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response does not contain a JSON object")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("response root must be an object")
    return parsed


def mapping_semantic_errors(response: dict[str, Any], batch: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if response.get("batch_id") != batch["batch_id"]:
        errors.append("batch_id mismatch")
    expected = [row["entry_id"] for row in batch["attributes"]]
    actual = [row.get("entry_id") for row in response.get("mappings", [])]
    if Counter(actual) != Counter(expected):
        errors.append("mapping entry_ids are missing, duplicated or added")
    for index, mapping in enumerate(response.get("mappings", [])):
        role, anchor = mapping.get("role"), mapping.get("anchor_type")
        if (role == "identity") != (anchor is not None):
            errors.append(f"mappings[{index}] role/anchor_type mismatch")
    return errors


def normalize_mapping_response(
    response: dict[str, Any], batch: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Repair only an unambiguous single opaque entry_id typo."""
    normalized = copy.deepcopy(response)
    mappings = normalized.get("mappings", [])
    expected = {row["entry_id"] for row in batch["attributes"]}
    actual = [row.get("entry_id") for row in mappings]
    missing = expected - set(actual)
    extra = set(actual) - expected
    transformations: list[str] = []
    if len(missing) == 1 and len(extra) == 1 and len(actual) == len(set(actual)):
        missing_id, extra_id = next(iter(missing)), next(iter(extra))
        for mapping in mappings:
            if mapping.get("entry_id") == extra_id:
                mapping["entry_id"] = missing_id
                transformations.append(f"single_entry_id_typo:{extra_id}->{missing_id}")
                break
    for mapping in mappings:
        concept = str(mapping.get("canonical_concept", ""))
        if concept and concept[0].isdigit():
            if concept.startswith(("2g_", "3g_", "4g_", "5g_")):
                repaired = "mobile_" + concept
            elif concept.startswith("80_plus_"):
                repaired = "power_supply_" + concept
            else:
                repaired = "numeric_" + concept
            mapping["canonical_concept"] = repaired
            transformations.append(f"leading_digit_concept:{concept}->{repaired}")
        role, anchor = mapping.get("role"), mapping.get("anchor_type")
        if role != "identity" and anchor is not None:
            mapping["anchor_type"] = None
            transformations.append(
                f"non_identity_anchor_removed:{mapping.get('entry_id')}:{anchor}"
            )
    return normalized, transformations


def locally_valid_response(
    row: dict[str, Any], batch: dict[str, Any], validator: Any
) -> tuple[dict[str, Any] | None, list[str]]:
    parsed = row.get("parsed_response")
    if not isinstance(parsed, dict):
        return None, []
    normalized, transformations = normalize_mapping_response(parsed, batch)
    if any(validator.iter_errors(normalized)):
        return None, transformations
    if mapping_semantic_errors(normalized, batch):
        return None, transformations
    return normalized, transformations


class Client:
    def __init__(self, args: argparse.Namespace, system_prompt: str, validator: Any) -> None:
        self.url = args.api_base.rstrip("/") + "/chat/completions"
        self.model = args.model
        self.timeout = args.timeout
        self.retries = args.retries
        self.max_tokens = args.max_tokens
        self.system_prompt = system_prompt
        self.validator = validator

    def ask(self, batch: dict[str, Any]) -> dict[str, Any]:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(batch, ensure_ascii=False)},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request_hash = hashlib.sha256(encoded).hexdigest()
        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                request = urllib.request.Request(
                    self.url,
                    data=encoded,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(request, timeout=self.timeout) as response:
                    api_response = json.load(response)
                choice = api_response["choices"][0]
                raw = str(choice["message"]["content"])
                metadata = {
                    "batch_id": batch["batch_id"],
                    "status": "ok",
                    "attempt": attempt,
                    "request_hash": request_hash,
                    "latency_seconds": time.perf_counter() - started,
                    "usage": api_response.get("usage", {}),
                    "finish_reason": choice.get("finish_reason"),
                    "raw_response": raw,
                    "completed_at": now(),
                }
                parsed = extract_json(raw)
                parsed, transformations = normalize_mapping_response(parsed, batch)
                schema_errors = sorted(error.message for error in self.validator.iter_errors(parsed))
                if schema_errors:
                    return {**metadata, "status": "invalid", "validation_stage": "schema", "errors": schema_errors, "parsed_response": parsed}
                semantic_errors = mapping_semantic_errors(parsed, batch)
                if semantic_errors:
                    return {**metadata, "status": "invalid", "validation_stage": "semantic", "errors": semantic_errors, "parsed_response": parsed}
                return {
                    **metadata,
                    "parsed_response": parsed,
                    "normalizations": transformations,
                }
            except Exception as error:
                last_error = error
                if attempt < self.retries:
                    time.sleep(min(20, 2 ** (attempt - 1)))
        return {
            "batch_id": batch["batch_id"],
            "status": "error",
            "attempt": self.retries,
            "request_hash": request_hash,
            "latency_seconds": time.perf_counter() - started,
            "error": f"{type(last_error).__name__}: {last_error}",
            "completed_at": now(),
        }


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.retries < 1 or args.max_batches < 0:
        raise ValueError("workers/retries must be positive and max-batches non-negative")
    batches_path, entries_path = args.batches.resolve(), args.entries.resolve()
    prompt_path, schema_path = args.prompt.resolve(), args.schema.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    batches = read_jsonl(batches_path)
    if args.max_batches:
        batches = batches[: args.max_batches]
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    prompt_sha, schema_sha = sha256_file(prompt_path), sha256_file(schema_path)
    system_prompt = prompt_path.read_text(encoding="utf-8") + "\n\nJSON Schema:\n" + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))

    preview = output_dir / "request_batches.jsonl"
    with preview.open("w", encoding="utf-8") as stream:
        for batch in batches:
            stream.write(json.dumps(batch, ensure_ascii=False) + "\n")
    if args.dry_run:
        print(json.dumps({"dry_run": True, "batches": len(batches), "attribute_entries": sum(len(x["attributes"]) for x in batches), "labels_in_input": False, "request_preview": str(preview)}, ensure_ascii=False, indent=2))
        return

    raw_path = output_dir / "raw_responses.jsonl"
    latest = latest_jsonl(raw_path)
    batches_by_id = {batch["batch_id"]: batch for batch in batches}
    completed = {
        batch_id
        for batch_id, row in latest.items()
        if batch_id in batches_by_id
        and row.get("prompt_sha256") == prompt_sha
        and row.get("schema_sha256") == schema_sha
        and row.get("model") == args.model
        and locally_valid_response(row, batches_by_id[batch_id], validator)[0] is not None
    }
    jobs = [batch for batch in batches if batch["batch_id"] not in completed]
    client = Client(args, system_prompt, validator)
    counts: Counter[str] = Counter()
    with raw_path.open("a" if raw_path.exists() else "w", encoding="utf-8", buffering=1) as stream:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(client.ask, batch): batch for batch in jobs}
            for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result = future.result()
                result.update({"prompt_sha256": prompt_sha, "schema_sha256": schema_sha, "model": args.model})
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                counts[result["status"]] += 1
                if done % 5 == 0 or done == len(jobs):
                    print(f"Ontology: {done}/{len(jobs)} new; ok={counts['ok']}, invalid={counts['invalid']}, errors={counts['error']}, reused={len(completed)}", flush=True)

    latest = latest_jsonl(raw_path)
    mapping_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    for batch in batches:
        response = latest.get(batch["batch_id"])
        parsed, transformations = (
            locally_valid_response(response, batch, validator) if response else (None, [])
        )
        if parsed is None:
            failed_rows.append({"batch_id": batch["batch_id"], "status": response.get("status") if response else "missing", "error": json.dumps(response.get("errors", response.get("error")) if response else None, ensure_ascii=False)})
            continue
        for mapping in parsed["mappings"]:
            mapping_rows.append(
                {
                    **mapping,
                    "source_status": response.get("status"),
                    "normalizations_json": json.dumps(transformations, ensure_ascii=False),
                }
            )
    entries = pd.read_parquet(entries_path)
    mappings = pd.DataFrame(mapping_rows)
    if len(mappings):
        mappings = entries.merge(mappings, on="entry_id", how="inner", validate="one_to_one")
    mappings.to_parquet(output_dir / "attribute_ontology.parquet", index=False)
    mappings.to_csv(output_dir / "attribute_ontology.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(failed_rows).to_csv(output_dir / "failed_batches.csv", index=False, encoding="utf-8-sig")
    statistics = {
        "requested_batches": len(batches),
        "ok_batches": len(batches) - len(failed_rows),
        "failed_batches": len(failed_rows),
        "mapped_entries": len(mappings),
        "total_entries": sum(len(batch["attributes"]) for batch in batches),
        "labels_used": False,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps({"created_at": now(), "batches": str(batches_path), "entries": str(entries_path), "prompt": str(prompt_path), "prompt_sha256": prompt_sha, "schema": str(schema_path), "schema_sha256": schema_sha, "model": args.model, "workers": args.workers, "statistics": statistics}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(statistics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
