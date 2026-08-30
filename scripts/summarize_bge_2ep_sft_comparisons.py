#!/usr/bin/env python3
"""Paired IID/hard comparison for the BGE-2ep SFT candidate family.

Unlike the generic MiniLM helper, this module never expects or compares OOD
predictions.  The former OOD pairs are part of BGE supervised training, so OOD
is represented only by the campaign's explicit ``-1`` sentinel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import create_bge_2ep_sft_notebooks as builder
import push_bge_2ep_sft_baseline_dataset as baseline_uploader
from src.experiment_significance import (
    SignificanceError,
    compare_prediction_frames,
    holm_adjust,
    read_prediction_artifact,
)


SPLITS = ("iid", "hard")
PRIMARY_SPLIT = "iid"
DIAGNOSTIC_SPLITS = ("hard",)
EXPECTED_ROWS = {"iid": builder.EXPECTED_IID, "hard": builder.EXPECTED_HARD}
PRACTICAL_TIE_MARGIN = 0.002
DEFAULT_ALPHA = 0.05
HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
EXPERIMENT_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")


class BgeComparisonError(ValueError):
    """Raised when the BGE paired-comparison contract is not satisfied."""


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BgeComparisonError(f"Could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise BgeComparisonError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _find_exactly_one(directory: Path, filename: str) -> Path:
    paths = list(directory.rglob(filename))
    if len(paths) != 1:
        raise BgeComparisonError(
            f"Expected exactly one {filename} below {directory}, got {paths}"
        )
    if paths[0].is_symlink() or not paths[0].is_file():
        raise BgeComparisonError(f"Artifact must be a regular file: {paths[0]}")
    return paths[0]


def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or HASH_PATTERN.fullmatch(value) is None:
        raise BgeComparisonError(f"{label} is not an exact lowercase SHA-256")
    return value


def _completion_report(completion: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    report = completion.get("training_report")
    if not isinstance(report, Mapping):
        raise BgeComparisonError(f"{label} completion has no training_report")
    validation = report.get("validation_splits")
    if not isinstance(validation, Mapping) or set(validation) != {"iid", "hard", "ood"}:
        raise BgeComparisonError(f"{label} report has unexpected validation splits")
    if validation.get("ood") != builder.OOD_SENTINEL:
        raise BgeComparisonError(f"{label} report changed the exact OOD=-1 sentinel")
    if report.get("evaluated_validation_splits") != ["iid", "hard"]:
        raise BgeComparisonError(f"{label} report claims OOD evaluation")
    if report.get("ood_evaluation_policy") != "disabled_train_contaminated":
        raise BgeComparisonError(f"{label} report changed the OOD policy")
    return report


def _validate_common_completion(
    completion: Mapping[str, Any],
    *,
    label: str,
) -> tuple[str, str, Mapping[str, Any]]:
    if completion.get("status") != "complete":
        raise BgeComparisonError(f"{label} completion is not complete")
    if completion.get("experiment_group") != "sft":
        raise BgeComparisonError(f"{label} completion is not routed to sft_exps")
    if completion.get("campaign") != builder.CAMPAIGN:
        raise BgeComparisonError(f"{label} completion belongs to another campaign")
    if completion.get("ood_evaluation_policy") != "disabled_train_contaminated":
        raise BgeComparisonError(f"{label} completion changed the OOD policy")
    run_id = str(completion.get("run_id", "")).strip()
    if re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
        raise BgeComparisonError(f"{label} run_id is not a 32-hex UUID")
    experiment = str(completion.get("experiment", "")).strip()
    if EXPERIMENT_PATTERN.fullmatch(experiment) is None:
        raise BgeComparisonError(f"{label} experiment name is invalid")
    for key in (
        "validation_manifest_sha256",
        "initial_checkpoint_manifest_sha256",
        "initial_checkpoint_model_sha256",
        "code_bundle_sha256",
        "frozen_recipe_sha256",
        "campaign_identity_sha256",
        "executable_cells_sha256",
        "loss_hook_sha256",
    ):
        _require_hash(completion.get(key), f"{label}.{key}")
    return run_id, experiment, _completion_report(completion, label=label)


def load_frozen_baseline(directory: Path) -> dict[str, Any]:
    directory = directory.expanduser().resolve(strict=True)
    if list(directory.rglob("ood_validation_predictions.parquet")):
        raise BgeComparisonError("Frozen baseline contains forbidden OOD predictions")
    manifest_path = _find_exactly_one(directory, baseline_uploader.MANIFEST_FILENAME)
    manifest = read_json_object(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("is_private") is not True:
        raise BgeComparisonError("Frozen baseline manifest is not exact/private v1")
    if manifest.get("evaluated_splits") != ["iid", "hard"]:
        raise BgeComparisonError("Frozen baseline manifest changed evaluated splits")
    ood = manifest.get("ood")
    if (
        not isinstance(ood, Mapping)
        or ood.get("evaluated") is not False
        or ood.get("metric_sentinel") != -1.0
        or ood.get("comparison") is not None
        or ood.get("prediction_file") is not None
    ):
        raise BgeComparisonError("Frozen baseline manifest fabricated OOD evidence")
    files = manifest.get("files")
    expected_files = {
        baseline_uploader.COMPLETION_FILENAME,
        *baseline_uploader.PREDICTION_FILENAMES.values(),
    }
    if not isinstance(files, Mapping) or set(files) != expected_files:
        raise BgeComparisonError("Frozen baseline manifest file ledger differs")
    artifact_paths: dict[str, Path] = {}
    for filename in expected_files:
        path = _find_exactly_one(directory, filename)
        declaration = files.get(filename)
        if not isinstance(declaration, Mapping):
            raise BgeComparisonError(f"Invalid baseline declaration for {filename}")
        measured = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        expected = {
            "bytes": declaration.get("bytes"),
            "sha256": declaration.get("sha256"),
        }
        if measured != expected:
            raise BgeComparisonError(f"Frozen baseline file drifted: {filename}")
        artifact_paths[filename] = path

    completion = read_json_object(
        artifact_paths[baseline_uploader.COMPLETION_FILENAME]
    )
    run_id, experiment, report = _validate_common_completion(
        completion, label="baseline"
    )
    if completion.get("role") != "baseline":
        raise BgeComparisonError("Frozen baseline completion role differs")
    if completion.get("baseline_comparison") not in (None, {}):
        raise BgeComparisonError("Frozen baseline contains a stale comparison")
    binding = manifest.get("binding")
    if not isinstance(binding, Mapping):
        raise BgeComparisonError("Frozen baseline manifest has no identity binding")
    exact_binding = {
        "baseline_run_id": run_id,
        "baseline_experiment": experiment,
        "campaign": completion["campaign"],
        "campaign_identity_sha256": completion["campaign_identity_sha256"],
        "source_sha256": completion["code_bundle_sha256"],
        "recipe_sha256": completion["frozen_recipe_sha256"],
        "executable_cells_sha256": completion["executable_cells_sha256"],
        "loss_hook_sha256": completion["loss_hook_sha256"],
        "checkpoint_manifest_sha256": completion[
            "initial_checkpoint_manifest_sha256"
        ],
        "checkpoint_model_sha256": completion["initial_checkpoint_model_sha256"],
        "validation_manifest_sha256": completion["validation_manifest_sha256"],
    }
    if dict(binding) != exact_binding:
        raise BgeComparisonError("Frozen baseline manifest/completion identity differs")
    return {
        "directory": directory,
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "completion": completion,
        "report": report,
        "run_id": run_id,
        "experiment": experiment,
        "binding": exact_binding,
        "predictions": {
            split: artifact_paths[filename]
            for split, filename in baseline_uploader.PREDICTION_FILENAMES.items()
        },
    }


def load_candidate(
    directory: Path,
    *,
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    directory = directory.expanduser().resolve(strict=True)
    if list(directory.rglob("ood_validation_predictions.parquet")):
        raise BgeComparisonError(f"Candidate contains forbidden OOD predictions: {directory}")
    completion_path = _find_exactly_one(directory, "notebook_completed.json")
    completion = read_json_object(completion_path)
    run_id, experiment, report = _validate_common_completion(
        completion, label=f"candidate {directory.name}"
    )
    if completion.get("role") != "candidate":
        raise BgeComparisonError(f"Candidate {experiment} has role {completion.get('role')!r}")
    if run_id == baseline["run_id"]:
        raise BgeComparisonError(f"Candidate {experiment} reuses the baseline run_id")
    if completion.get("baseline_comparison") not in (None, {}):
        raise BgeComparisonError(f"Candidate {experiment} contains a stale comparison")
    binding = baseline["binding"]
    common = {
        "model": baseline["completion"].get("model"),
        "dataset_ref": baseline["completion"].get("dataset_ref"),
        "initial_checkpoint_ref": baseline["completion"].get(
            "initial_checkpoint_ref"
        ),
        "code_bundle_sha256": binding["source_sha256"],
        "loss_hook_sha256": binding["loss_hook_sha256"],
        "initial_checkpoint_manifest_sha256": binding[
            "checkpoint_manifest_sha256"
        ],
        "initial_checkpoint_model_sha256": binding["checkpoint_model_sha256"],
        "validation_manifest_sha256": binding["validation_manifest_sha256"],
    }
    for key, expected in common.items():
        if completion.get(key) != expected:
            raise BgeComparisonError(f"Candidate {experiment} differs at {key}")
    predictions = {
        split: _find_exactly_one(
            directory,
            baseline_uploader.PREDICTION_FILENAMES[split],
        )
        for split in SPLITS
    }
    return {
        "directory": directory,
        "completion_path": completion_path,
        "completion": completion,
        "report": report,
        "run_id": run_id,
        "experiment": experiment,
        "predictions": predictions,
    }


def _metric_from_report(
    report: Mapping[str, Any],
    *,
    split: str,
    label: str,
) -> float:
    validation = report["validation_splits"]
    metrics = validation[split]
    if not isinstance(metrics, Mapping):
        raise BgeComparisonError(f"{label} has no {split} metrics")
    value = metrics.get("macro_average_precision")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise BgeComparisonError(f"{label} has invalid {split} macro AP")
    return float(value)


def _ood_result() -> dict[str, Any]:
    return {
        "evaluated": False,
        "status": "disabled_train_contaminated",
        "reason": "former frozen OOD pairs are part of BGE supervised training",
        "examples": 0,
        "delta_macro_average_precision": None,
        "p_value": None,
        "p_value_holm": None,
        "ci95_low": None,
        "ci95_high": None,
    }


def _practical_relation(delta: float, margin: float) -> str:
    if delta > margin:
        return "improves_beyond_margin"
    if delta < -margin:
        return "degrades_beyond_margin"
    return "practical_tie"


def summarize_candidate_family(
    baseline_dir: Path,
    candidate_dirs: Sequence[Path],
    *,
    planned_experiments: Sequence[str],
    family_name: str,
    tie_break_order: Sequence[str] | None = None,
    practical_tie_margin: float = PRACTICAL_TIE_MARGIN,
    alpha: float = DEFAULT_ALPHA,
    permutations: int = 2_000,
    bootstrap_resamples: int = 2_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare one complete, predeclared candidate family to the BGE baseline."""
    if not family_name.strip():
        raise BgeComparisonError("family_name is required")
    if practical_tie_margin < 0 or not math.isfinite(practical_tie_margin):
        raise BgeComparisonError("practical_tie_margin must be finite and non-negative")
    if not 0 < alpha <= 1:
        raise BgeComparisonError("alpha must be in (0, 1]")
    planned = [str(value).strip() for value in planned_experiments]
    if not planned or any(EXPERIMENT_PATTERN.fullmatch(value) is None for value in planned):
        raise BgeComparisonError("planned_experiments must contain valid experiment names")
    if len(set(planned)) != len(planned):
        raise BgeComparisonError("planned_experiments contains duplicates")

    baseline = load_frozen_baseline(baseline_dir)
    candidates = [load_candidate(path, baseline=baseline) for path in candidate_dirs]
    by_experiment = {candidate["experiment"]: candidate for candidate in candidates}
    if len(by_experiment) != len(candidates):
        raise BgeComparisonError("Candidate experiment names are not unique")
    if set(by_experiment) != set(planned):
        raise BgeComparisonError(
            "Holm family must be complete before selection: "
            f"missing={sorted(set(planned) - set(by_experiment))}, "
            f"unexpected={sorted(set(by_experiment) - set(planned))}"
        )
    if baseline["experiment"] in set(planned):
        raise BgeComparisonError("The baseline must not be listed as a candidate hypothesis")

    comparisons: dict[str, dict[str, Any]] = {}
    for candidate_index, experiment in enumerate(planned):
        candidate = by_experiment[experiment]
        split_results: dict[str, dict[str, Any]] = {}
        for split_index, split in enumerate(SPLITS):
            try:
                result = compare_prediction_frames(
                    read_prediction_artifact(baseline["predictions"][split]),
                    read_prediction_artifact(candidate["predictions"][split]),
                    permutations=permutations,
                    bootstrap_resamples=bootstrap_resamples,
                    seed=seed + candidate_index * len(SPLITS) + split_index,
                )
            except (SignificanceError, ValueError) as error:
                raise BgeComparisonError(
                    f"Could not compare {experiment} on {split}: {error}"
                ) from error
            if result["examples"] != EXPECTED_ROWS[split]:
                raise BgeComparisonError(f"{experiment} has unexpected {split} rows")
            baseline_report_ap = _metric_from_report(
                baseline["report"], split=split, label="baseline"
            )
            candidate_report_ap = _metric_from_report(
                candidate["report"], split=split, label=experiment
            )
            if not math.isclose(
                float(result["baseline_macro_average_precision"]),
                baseline_report_ap,
                rel_tol=0,
                abs_tol=1e-12,
            ):
                raise BgeComparisonError(f"Baseline {split} report/predictions differ")
            if not math.isclose(
                float(result["candidate_macro_average_precision"]),
                candidate_report_ap,
                rel_tol=0,
                abs_tol=1e-12,
            ):
                raise BgeComparisonError(
                    f"Candidate {experiment} {split} report/predictions differ"
                )
            split_results[split] = result
        comparisons[experiment] = {
            "candidate": candidate,
            "splits": split_results,
        }

    # The planned experiments are the family of hypotheses.  Holm is applied
    # across candidates for each split, never across IID/hard/OOD.  IID is the
    # only selection metric; hard remains explicitly diagnostic.
    for split in SPLITS:
        adjusted = holm_adjust(
            {
                experiment: float(comparisons[experiment]["splits"][split]["p_value"])
                for experiment in planned
            }
        )
        for experiment, value in adjusted.items():
            comparisons[experiment]["splits"][split]["p_value_holm"] = value
            comparisons[experiment]["splits"][split]["holm_family"] = family_name
            comparisons[experiment]["splits"][split]["holm_family_size"] = len(planned)

    order = list(tie_break_order or [baseline["experiment"], *planned])
    expected_order = {baseline["experiment"], *planned}
    if len(order) != len(expected_order) or set(order) != expected_order:
        raise BgeComparisonError(
            "tie_break_order must contain baseline and every planned candidate exactly once"
        )
    iid_scores = {
        baseline["experiment"]: _metric_from_report(
            baseline["report"], split="iid", label="baseline"
        ),
        **{
            experiment: float(
                comparisons[experiment]["splits"]["iid"][
                    "candidate_macro_average_precision"
                ]
            )
            for experiment in planned
        },
    }
    maximum = max(iid_scores.values())
    numeric_winner = min(
        (name for name, value in iid_scores.items() if value == maximum),
        key=order.index,
    )
    within_margin = {
        name for name, value in iid_scores.items() if maximum - value <= practical_tie_margin
    }
    practical_winner = min(within_margin, key=order.index)

    candidate_outputs: dict[str, dict[str, Any]] = {}
    for experiment in planned:
        candidate = comparisons[experiment]["candidate"]
        split_results = comparisons[experiment]["splits"]
        iid_delta = float(split_results["iid"]["delta_macro_average_precision"])
        comparison = {
            "schema_version": 1,
            "status": "ready_ood_disabled",
            "baseline_run_id": baseline["run_id"],
            "candidate_run_id": candidate["run_id"],
            "baseline_experiment": baseline["experiment"],
            "candidate_experiment": experiment,
            "baseline_manifest_sha256": baseline["manifest_sha256"],
            "baseline_binding": baseline["binding"],
            "candidate_binding": {
                "campaign": candidate["completion"]["campaign"],
                "campaign_identity_sha256": candidate["completion"][
                    "campaign_identity_sha256"
                ],
                "source_sha256": candidate["completion"]["code_bundle_sha256"],
                "recipe_sha256": candidate["completion"]["frozen_recipe_sha256"],
                "checkpoint_manifest_sha256": candidate["completion"][
                    "initial_checkpoint_manifest_sha256"
                ],
                "checkpoint_model_sha256": candidate["completion"][
                    "initial_checkpoint_model_sha256"
                ],
                "validation_manifest_sha256": candidate["completion"][
                    "validation_manifest_sha256"
                ],
                "loss_hook_sha256": candidate["completion"]["loss_hook_sha256"],
            },
            "method": "paired_component_permutation",
            "confidence_interval_method": "paired_component_bootstrap_percentile",
            "multiple_testing_correction": "holm_within_planned_candidate_family_per_split",
            "holm_family": family_name,
            "holm_family_members": planned,
            "primary_split": PRIMARY_SPLIT,
            "diagnostic_splits": list(DIAGNOSTIC_SPLITS),
            "practical_tie_margin": practical_tie_margin,
            "iid_practical_relation": _practical_relation(
                iid_delta, practical_tie_margin
            ),
            "iid_statistically_significant_after_holm": (
                float(split_results["iid"]["p_value_holm"]) <= alpha
            ),
            "ood_policy": "disabled_train_contaminated_no_paired_comparison",
            "splits": {
                "iid": split_results["iid"],
                "hard": split_results["hard"],
                "ood": _ood_result(),
            },
        }
        augmented = {
            **candidate["completion"],
            "experiment_group": "sft",
            "baseline_comparison": comparison,
        }
        candidate_outputs[experiment] = {
            "comparison": comparison,
            "augmented_completion": augmented,
        }

    summary_candidates = {
        experiment: {
            "run_id": comparisons[experiment]["candidate"]["run_id"],
            "campaign_identity_sha256": comparisons[experiment]["candidate"][
                "completion"
            ]["campaign_identity_sha256"],
            "source_sha256": comparisons[experiment]["candidate"]["completion"][
                "code_bundle_sha256"
            ],
            "recipe_sha256": comparisons[experiment]["candidate"]["completion"][
                "frozen_recipe_sha256"
            ],
            "iid_macro_ap": iid_scores[experiment],
            "iid_delta": comparisons[experiment]["splits"]["iid"][
                "delta_macro_average_precision"
            ],
            "iid_p_value": comparisons[experiment]["splits"]["iid"]["p_value"],
            "iid_p_value_holm": comparisons[experiment]["splits"]["iid"][
                "p_value_holm"
            ],
            "hard_macro_ap": comparisons[experiment]["splits"]["hard"][
                "candidate_macro_average_precision"
            ],
            "hard_delta": comparisons[experiment]["splits"]["hard"][
                "delta_macro_average_precision"
            ],
            "hard_p_value": comparisons[experiment]["splits"]["hard"]["p_value"],
            "hard_p_value_holm": comparisons[experiment]["splits"]["hard"][
                "p_value_holm"
            ],
            "ood_macro_ap": -1.0,
            "ood_compared": False,
            "iid_practical_relation": candidate_outputs[experiment]["comparison"][
                "iid_practical_relation"
            ],
        }
        for experiment in planned
    }
    family_summary = {
        "schema_version": 1,
        "status": "complete",
        "campaign": builder.CAMPAIGN,
        "family_name": family_name,
        "planned_family_complete": True,
        "planned_experiments": planned,
        "primary_split": PRIMARY_SPLIT,
        "diagnostic_splits": list(DIAGNOSTIC_SPLITS),
        "multiple_testing_correction": "holm_within_planned_candidate_family_per_split",
        "alpha": alpha,
        "practical_tie_margin": practical_tie_margin,
        "tie_break_order": order,
        "baseline": {
            "run_id": baseline["run_id"],
            "experiment": baseline["experiment"],
            "manifest_sha256": baseline["manifest_sha256"],
            "campaign_identity_sha256": baseline["binding"][
                "campaign_identity_sha256"
            ],
            "source_sha256": baseline["binding"]["source_sha256"],
            "recipe_sha256": baseline["binding"]["recipe_sha256"],
            "checkpoint_manifest_sha256": baseline["binding"][
                "checkpoint_manifest_sha256"
            ],
            "iid_macro_ap": iid_scores[baseline["experiment"]],
            "hard_macro_ap": _metric_from_report(
                baseline["report"], split="hard", label="baseline"
            ),
            "ood_macro_ap": -1.0,
        },
        "candidates": summary_candidates,
        "selection": {
            "numeric_iid_winner": numeric_winner,
            "numeric_iid_max": maximum,
            "selected_with_practical_tie_break": practical_winner,
            "selected_iid_macro_ap": iid_scores[practical_winner],
            "selected_iid_delta_vs_baseline": (
                iid_scores[practical_winner] - iid_scores[baseline["experiment"]]
            ),
            "hard_used_for_selection": False,
            "ood_used_for_selection": False,
        },
        "candidate_outputs": candidate_outputs,
    }
    return family_summary


def materialize_summary(summary: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    candidate_outputs = summary.get("candidate_outputs")
    if not isinstance(candidate_outputs, Mapping):
        raise BgeComparisonError("Summary has no candidate outputs")
    written: dict[str, dict[str, str]] = {}
    for experiment, raw_output in candidate_outputs.items():
        if not isinstance(raw_output, Mapping):
            raise BgeComparisonError(f"Invalid candidate output for {experiment}")
        candidate_dir = output_dir / str(experiment)
        comparison_path = candidate_dir / "baseline_comparison.json"
        completion_path = candidate_dir / "completion_with_comparison.json"
        write_json(comparison_path, raw_output["comparison"])
        write_json(completion_path, raw_output["augmented_completion"])
        written[str(experiment)] = {
            "comparison": str(comparison_path),
            "completion": str(completion_path),
        }
    compact_summary = dict(summary)
    compact_summary.pop("candidate_outputs", None)
    summary_path = output_dir / "family_summary.json"
    write_json(summary_path, compact_summary)
    return {"summary": str(summary_path), "candidates": written}


def validate_output_isolation(output_dir: Path, input_dirs: Sequence[Path]) -> Path:
    output = output_dir.expanduser().resolve()
    for raw_input in input_dirs:
        source = raw_input.expanduser().resolve(strict=True)
        if output == source or output in source.parents or source in output.parents:
            raise BgeComparisonError(
                f"Comparison output and source must be disjoint: output={output}, source={source}"
            )
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a complete planned BGE candidate family against the slim "
            "BGE baseline on IID/hard only."
        )
    )
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--planned-experiment",
        action="append",
        required=True,
        help="Repeat for every predeclared non-baseline candidate in the Holm family",
    )
    parser.add_argument("--family-name", required=True)
    parser.add_argument("--tie-break-order", action="append", default=[])
    parser.add_argument("--practical-tie-margin", type=float, default=PRACTICAL_TIE_MARGIN)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--permutations", type=int, default=2_000)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = validate_output_isolation(
        args.output_dir,
        [args.baseline_dir, *args.candidate_dir],
    )
    summary = summarize_candidate_family(
        args.baseline_dir,
        args.candidate_dir,
        planned_experiments=args.planned_experiment,
        family_name=args.family_name,
        tie_break_order=args.tie_break_order or None,
        practical_tie_margin=args.practical_tie_margin,
        alpha=args.alpha,
        permutations=args.permutations,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    outputs = materialize_summary(summary, output_dir)
    compact = dict(summary)
    compact.pop("candidate_outputs", None)
    print(json.dumps({"result": compact, "outputs": outputs}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
