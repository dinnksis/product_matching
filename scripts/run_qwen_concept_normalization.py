"""Run a small resumable, label-free Qwen pass over extracted concept names."""

from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import hashlib
import json
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTIONS = ROOT / "artifacts" / "qwen_semantic_extraction_v1_3_sanitized500" / "sanitized_pairs.jsonl"
DEFAULT_BASE_MAP = ROOT / "artifacts" / "qwen_concept_normalization_v1_500" / "concept_normalization_map.parquet"
DEFAULT_PROMPT = ROOT / "prompts" / "qwen_concept_normalization_v2.md"
DEFAULT_SCHEMA = ROOT / "schemas" / "qwen_concept_normalization_v2.schema.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "qwen_concept_normalization_v2_500"

FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("identity_model", ("brand", "model", "sku", "part", "article", "code", "series", "line", "collection", "compatib")),
    ("power_electronics", ("power", "voltage", "current", "battery", "frequency", "screen", "display", "memory", "ram", "ssd", "cpu", "gpu", "interface", "usb", "wifi")),
    ("food_scent", ("flavor", "scent", "fragrance", "ingredient", "content", "food", "diet", "allergen")),
    ("audience_age", ("age", "gender", "audience", "target", "pet", "animal", "breed")),
    ("material_color", ("material", "color", "finish", "coating", "texture", "pattern", "design")),
    ("size_measurement", ("size", "length", "width", "height", "depth", "diameter", "dimension", "distance", "area", "weight", "volume", "capacity", "load")),
    ("quantity_package", ("count", "quantity", "package", "packaging", "included", "contents", "configuration", "set")),
    ("form_type", ("type", "form", "shape", "style", "class", "category", "kind")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize extracted concepts in compact Qwen batches.")
    parser.add_argument("--extractions", type=Path, default=DEFAULT_EXTRACTIONS)
    parser.add_argument("--base-map", type=Path, default=DEFAULT_BASE_MAP)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--api-base", default="http://localhost:8194/v1")
    parser.add_argument("--model", default="qwen3.5-397b-a17b-fp8")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=45)
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


def load_base_map(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    result = dict(zip(frame["source_concept"].astype(str), frame["target_concept"].astype(str)))
    return frame, result


def family(concept: str) -> str:
    parts = concept.split("_")
    for name, tokens in FAMILIES:
        if any(
            part == token or (token == "compatib" and part.startswith(token))
            for token in tokens
            for part in parts
        ):
            return name
    return "other"


def facts(pair: dict[str, Any]) -> list[dict[str, Any]]:
    return [*pair.get("identity_anchors", []), *pair.get("differences", []), *pair.get("missing_information", [])]


def build_batches(extractions: Path, concept_map: dict[str, str], batch_size: int) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    support: Counter[str] = Counter()
    categories: dict[str, Counter[str]] = defaultdict(Counter)
    relations: dict[str, Counter[str]] = defaultdict(Counter)
    raw_names: dict[str, Counter[str]] = defaultdict(Counter)
    with extractions.open(encoding="utf-8") as stream:
        for line in stream:
            pair = json.loads(line)
            # Deliberately never access pair["human_label"].
            category = str(pair["category"])
            for fact in facts(pair):
                source = str(fact["concept"])
                concept = concept_map.get(source, source)
                support[concept] += 1
                categories[concept][category] += 1
                relations[concept][str(fact.get("relation", "same"))] += 1
                for side in ("evidence_a", "evidence_b"):
                    for evidence in fact.get(side, []):
                        name = evidence.get("raw_attribute_name")
                        if name:
                            raw_names[concept][str(name)] += 1

    entries: list[dict[str, Any]] = []
    for index, concept in enumerate(sorted(support), 1):
        entries.append({
            "entry_id": f"concept_{index:04d}",
            "source_concept": concept,
            "support": support[concept],
            "categories": [key for key, _ in categories[concept].most_common(5)],
            "relations": [key for key, _ in relations[concept].most_common()],
            "raw_attribute_names": [key for key, _ in raw_names[concept].most_common(5)],
            "family": family(concept),
        })
    entries_frame = pd.DataFrame(entries)
    batches: list[dict[str, Any]] = []
    batch_index = 0
    for family_name, group in entries_frame.groupby("family", sort=True):
        ordered = group.sort_values(["source_concept", "support"], ascending=[True, False]).to_dict("records")
        for start in range(0, len(ordered), batch_size):
            batch_index += 1
            chunk = ordered[start : start + batch_size]
            batches.append({"batch_id": f"concept_batch_{batch_index:03d}", "family": family_name, "concepts": chunk})
    return batches, entries_frame


def extract_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response does not contain a JSON object")
    result = json.loads(text[start : end + 1])
    if not isinstance(result, dict):
        raise ValueError("response root must be an object")
    return result


def semantic_errors(response: dict[str, Any], batch: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if response.get("batch_id") != batch["batch_id"]:
        errors.append("batch_id mismatch")
    expected = {row["entry_id"]: row["source_concept"] for row in batch["concepts"]}
    targets = set(expected.values())
    mappings = response.get("mappings", [])
    actual_ids = [row.get("entry_id") for row in mappings]
    if Counter(actual_ids) != Counter(expected.keys()):
        errors.append("entry_ids missing, duplicated or added")
    for row in mappings:
        entry_id = row.get("entry_id")
        if entry_id not in expected:
            continue
        source, target, decision = expected[entry_id], row.get("target_concept"), row.get("decision")
        if row.get("source_concept") != source:
            errors.append(f"{entry_id}: source_concept mismatch")
        if target not in targets:
            errors.append(f"{entry_id}: target is not in the same batch")
        if decision == "KEEP" and target != source:
            errors.append(f"{entry_id}: KEEP must target itself")
        # A self-targeted MERGE is an unambiguous local KEEP and is normalized
        # during post-processing instead of invalidating the whole batch.
    return errors


class Client:
    def __init__(self, args: argparse.Namespace, prompt: str, validator: Any) -> None:
        self.url = args.api_base.rstrip("/") + "/chat/completions"
        self.model = args.model
        self.timeout = args.timeout
        self.retries = args.retries
        self.max_tokens = args.max_tokens
        self.prompt = prompt
        self.validator = validator

    def ask(self, batch: dict[str, Any]) -> dict[str, Any]:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": json.dumps(batch, ensure_ascii=False)},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                request = urllib.request.Request(self.url, data=encoded, headers={"Content-Type": "application/json"}, method="POST")
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(request, timeout=self.timeout) as response:
                    api_response = json.load(response)
                choice = api_response["choices"][0]
                raw = str(choice["message"]["content"])
                parsed = extract_json(raw)
                schema_errors = sorted(error.message for error in self.validator.iter_errors(parsed))
                errors = schema_errors or semantic_errors(parsed, batch)
                return {
                    "batch_id": batch["batch_id"],
                    "status": "invalid" if errors else "ok",
                    "attempt": attempt,
                    "latency_seconds": time.perf_counter() - started,
                    "usage": api_response.get("usage", {}),
                    "finish_reason": choice.get("finish_reason"),
                    "raw_response": raw,
                    "parsed_response": parsed,
                    "errors": errors,
                    "completed_at": now(),
                }
            except Exception as error:
                last_error = error
                if attempt < self.retries:
                    time.sleep(min(20, 2 ** (attempt - 1)))
        return {"batch_id": batch["batch_id"], "status": "error", "attempt": self.retries, "error": f"{type(last_error).__name__}: {last_error}", "latency_seconds": time.perf_counter() - started, "completed_at": now()}


def latest_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row["batch_id"]): row for row in read_jsonl(path) if row.get("batch_id")}


def lexical_guard(source: str, target: str) -> tuple[bool, str]:
    source_tokens, target_tokens = set(source.split("_")), set(target.split("_"))
    union = source_tokens | target_tokens
    jaccard = len(source_tokens & target_tokens) / len(union) if union else 0.0
    ratio = difflib.SequenceMatcher(None, source, target).ratio()
    allowed = jaccard >= 0.5 or ratio >= 0.85
    return allowed, f"token_jaccard={jaccard:.3f};sequence_ratio={ratio:.3f}"


def resolve_targets(mapping: dict[str, str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for source in mapping:
        current, seen = source, {source}
        while mapping.get(current, current) != current:
            current = mapping[current]
            if current in seen:
                current = source
                break
            seen.add(current)
        resolved[source] = current
    return resolved


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.batch_size < 10 or args.batch_size > 60 or args.retries < 1:
        raise ValueError("workers/retries must be positive; batch-size must be 10..60")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_frame, base_map = load_base_map(args.base_map.resolve())
    batches, entries = build_batches(args.extractions.resolve(), base_map, args.batch_size)
    entries.to_parquet(output_dir / "concept_entries.parquet", index=False)
    entries.to_csv(output_dir / "concept_entries.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "request_batches.jsonl").open("w", encoding="utf-8") as stream:
        for batch in batches:
            stream.write(json.dumps(batch, ensure_ascii=False) + "\n")
    if args.dry_run:
        print(json.dumps({"dry_run": True, "concepts": len(entries), "batches": len(batches), "labels_in_input": False}, ensure_ascii=False, indent=2))
        return

    schema = json.loads(args.schema.resolve().read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    prompt_sha, schema_sha = sha256_file(args.prompt.resolve()), sha256_file(args.schema.resolve())
    prompt = args.prompt.resolve().read_text(encoding="utf-8") + "\n\nJSON Schema:\n" + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    raw_path = output_dir / "raw_responses.jsonl"
    latest = latest_rows(raw_path)
    batch_by_id = {batch["batch_id"]: batch for batch in batches}
    completed = {
        batch_id for batch_id, row in latest.items()
        if batch_id in batch_by_id
        and row.get("prompt_sha256") == prompt_sha
        and row.get("schema_sha256") == schema_sha
        and str(row.get("model", "")).casefold() == args.model.casefold()
        and not list(validator.iter_errors(row.get("parsed_response", {})))
        and not semantic_errors(row.get("parsed_response", {}), batch_by_id[batch_id])
    }
    jobs = [batch for batch in batches if batch["batch_id"] not in completed]
    client = Client(args, prompt, validator)
    counts: Counter[str] = Counter()
    with raw_path.open("a" if raw_path.exists() else "w", encoding="utf-8", buffering=1) as stream:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(client.ask, batch): batch for batch in jobs}
            for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result = future.result()
                result.update({"prompt_sha256": prompt_sha, "schema_sha256": schema_sha, "model": args.model})
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                counts[result["status"]] += 1
                print(f"Concepts: {done}/{len(jobs)} batches; ok={counts['ok']}, invalid={counts['invalid']}, errors={counts['error']}, reused={len(completed)}", flush=True)

    latest = latest_rows(raw_path)
    proposals: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    qwen_map = {concept: concept for concept in entries["source_concept"].astype(str)}
    for batch in batches:
        row = latest.get(batch["batch_id"])
        parsed = row.get("parsed_response", {}) if row else {}
        if (
            not row
            or list(validator.iter_errors(parsed))
            or semantic_errors(parsed, batch)
        ):
            failed.append({"batch_id": batch["batch_id"], "status": row.get("status") if row else "missing", "errors": json.dumps(row.get("errors", row.get("error")) if row else None, ensure_ascii=False)})
            continue
        for raw_proposal in parsed["mappings"]:
            proposal = dict(raw_proposal)
            source, target = proposal["source_concept"], proposal["target_concept"]
            if proposal["decision"] == "MERGE" and target == source:
                proposal["raw_decision"] = "MERGE"
                proposal["decision"] = "KEEP"
                proposal["local_normalization"] = "self_targeted_merge_to_keep"
            guard, guard_reason = lexical_guard(source, target)
            applied = proposal["decision"] == "MERGE" and proposal["confidence"] == "high" and guard
            if applied:
                qwen_map[source] = target
            proposals.append({**proposal, "batch_id": batch["batch_id"], "lexical_guard": guard, "lexical_guard_reason": guard_reason, "applied": applied})
    qwen_map = resolve_targets(qwen_map)

    final_frame = base_frame.copy()
    final_frame["deterministic_target_concept"] = final_frame["target_concept"].astype(str)
    final_frame["target_concept"] = final_frame["deterministic_target_concept"].map(qwen_map).fillna(final_frame["deterministic_target_concept"])
    final_frame["qwen_changed"] = final_frame["target_concept"] != final_frame["deterministic_target_concept"]
    final_frame.to_parquet(output_dir / "concept_normalization_map.parquet", index=False)
    final_frame.to_csv(output_dir / "concept_normalization_map.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(proposals).to_csv(output_dir / "qwen_proposals.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(failed).to_csv(output_dir / "failed_batches.csv", index=False, encoding="utf-8-sig")
    stats = {
        "concepts": len(entries),
        "batches": len(batches),
        "ok_batches": len(batches) - len(failed),
        "failed_batches": len(failed),
        "qwen_merge_proposals": sum(row.get("decision") == "MERGE" for row in proposals),
        "high_confidence_lexically_safe_merges_applied": sum(row.get("applied", False) for row in proposals),
        "concepts_after": int(final_frame["target_concept"].nunique()),
        "labels_used": False,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
