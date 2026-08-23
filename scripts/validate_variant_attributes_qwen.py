"""Validate variant-defining attribute candidates with an OpenAI-compatible Qwen API."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


SYSTEM_PROMPT = """You are auditing attributes for marketplace product identity.
Decide whether changing only the given attribute normally creates a different
sellable product variant in the stated category. Use the human-labelled examples
as evidence, but distinguish causation from incidental differences. Be conservative:
approve only attributes that can safely create a hard negative when changed to a
different plausible value while brand/model and all other variant fields stay fixed.
Return one JSON object with exactly these keys:
decision (variant_defining|context_dependent|not_defining), safe_to_mutate (boolean),
normalized_attribute (string), rationale (short string), constraints (short string),
replacement_kind (categorical|numeric_same_unit|identifier|unsafe)."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://127.0.0.1:8193/v1")
    parser.add_argument("--model", default="Qwen3.5-397B-A17B-FP8")
    parser.add_argument(
        "--candidates", type=Path,
        default=Path("reports/variant_attributes/attribute_candidates.csv"),
    )
    parser.add_argument(
        "--examples", type=Path,
        default=Path("reports/variant_attributes/candidate_examples.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("reports/variant_attributes/qwen_validation.jsonl"),
    )
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Response does not contain a JSON object")
    result = json.loads(text[start : end + 1])
    required = {
        "decision", "safe_to_mutate", "normalized_attribute",
        "rationale", "constraints", "replacement_kind",
    }
    if set(result) != required:
        raise ValueError(f"Unexpected JSON keys: {sorted(result)}")
    if result["decision"] not in {
        "variant_defining", "context_dependent", "not_defining"
    }:
        raise ValueError("Invalid decision")
    if not isinstance(result["safe_to_mutate"], bool):
        raise ValueError("safe_to_mutate must be boolean")
    return result


def request_qwen(args: argparse.Namespace, record: dict[str, Any]) -> dict[str, Any]:
    prompt = json.dumps(record, ensure_ascii=False, indent=2)
    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 500,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(args.retries):
        try:
            request = urllib.request.Request(
                args.api_base.rstrip("/") + "/chat/completions",
                data=encoded,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                payload = json.load(response)
            content = payload["choices"][0]["message"]["content"]
            return {**record, "qwen": extract_json(content), "raw_response": content}
        # SSH/proxy failures can surface as several http.client exception types,
        # including BadStatusLine, so all request/parse failures are retried.
        except Exception as error:
            last_error = error
            time.sleep(2**attempt)
    raise RuntimeError(f"Qwen request failed: {last_error}")


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.candidates)
    candidates = candidates[candidates["candidate"].astype(bool)].copy()
    if args.limit is not None:
        candidates = candidates.head(args.limit)

    examples: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with args.examples.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            examples[(row["category"], row["attribute"])].append(row)

    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if args.output.exists():
        with args.output.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                completed[(row["category"], row["attribute"])] = row

    records = []
    for row in candidates.to_dict("records"):
        key = (row["category"], row["attribute"])
        if key in completed:
            continue
        records.append(
            {
                "category": row["category"],
                "attribute": row["attribute"],
                "statistics": {
                    "conflicts": int(row["conflicts"]),
                    "negative_rate_given_conflict": float(
                        row["negative_rate_given_conflict"]
                    ),
                    "category_negative_rate": float(row["category_negative_rate"]),
                },
                "human_examples": examples[key],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.output.exists() else "w"
    done = 0
    with args.output.open(mode, encoding="utf-8", buffering=1) as output:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(request_qwen, args, row): row for row in records}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                done += 1
                if done % 10 == 0 or done == len(records):
                    print(f"Validated {done}/{len(records)} new candidates", flush=True)
    print(f"Saved {args.output}; reused={len(completed)}, added={done}")


if __name__ == "__main__":
    main()
