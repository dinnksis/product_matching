#!/usr/bin/env python3
"""Generate four comparable 2xT4 architecture-baseline notebooks."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import sys
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import create_minilm_validation_baseline_notebook as baseline_builder
import create_qwen_training_notebook as shared


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "architecture_baselines.json"
OUTPUT_DIR = ROOT / "notebooks" / "architecture_baselines"
VALIDATION_DATA_DIR = ROOT / "prepared" / "validation_splits_v1"
VALIDATION_MANIFEST = VALIDATION_DATA_DIR / "manifest.json"
VALIDATION_DATASET_REF = "alexproger23/product-matching-validation-splits-v1"
RAW_DATASET_REF = "dinakepecheva/e-cup-human-data"
SERIALIZATION_FREQUENCY = (
    ROOT / "prepared" / "serialization_ablation" / "attribute_name_frequency.csv"
)
SERIALIZATION_VARIANT = "S2_VALUES_ONLY"
LOCAL_VALIDATION_FILES = {
    "human/items.parquet": "human/items.parquet",
    "human/train_pairs.parquet": "human/train_pairs.parquet",
    "human/iid_validation_pairs.parquet": "human/iid_validation_pairs.parquet",
    "human/hard_validation_pairs.parquet": "human/hard_validation_pairs.parquet",
    "human/ood_validation_pairs.parquet": "human/ood_validation_pairs.parquet",
}
NOTEBOOK_FILENAMES = {
    "gte": "gte_architecture_baseline_2xt4.ipynb",
    "rumodernbert": "rumodernbert_architecture_baseline_2xt4.ipynb",
    "bge-v2-m3": "bge_v2_m3_architecture_baseline_2xt4.ipynb",
    "minilm-5ep": "minilm_5ep_architecture_baseline_2xt4.ipynb",
}
EXTRA_EMBEDDED_FILES = (
    Path("requirements-architecture-baselines.txt"),
    Path("src/serialization_ablation.py"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def load_validation_dataset() -> dict[str, object]:
    manifest = json.loads(VALIDATION_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("version") != "human_v1":
        raise ValueError("Validation manifest must have version='human_v1'")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Validation manifest has no outputs mapping")
    for relative, local_name in LOCAL_VALIDATION_FILES.items():
        declaration = outputs.get(relative)
        path = VALIDATION_DATA_DIR / local_name
        if not isinstance(declaration, dict) or not path.is_file():
            raise FileNotFoundError(f"Frozen validation input is missing: {relative}")
        if path.stat().st_size != int(declaration.get("bytes", -1)):
            raise ValueError(f"Frozen validation size differs: {relative}")
        if sha256(path) != str(declaration.get("sha256")):
            raise ValueError(f"Frozen validation SHA-256 differs: {relative}")
    return {
        "dataset": VALIDATION_DATASET_REF,
        "manifest_sha256": sha256(VALIDATION_MANIFEST),
        "manifest": manifest,
    }


def load_configuration() -> dict[str, object]:
    configuration = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    protocol = configuration.get("protocol")
    profiles = configuration.get("profiles")
    if not isinstance(protocol, dict) or not isinstance(profiles, dict):
        raise ValueError("Architecture config requires protocol and profiles mappings")
    if set(profiles) != set(NOTEBOOK_FILENAMES):
        raise ValueError("Architecture config profiles differ from notebook profiles")
    if protocol.get("serialization") != SERIALIZATION_VARIANT:
        raise ValueError("Architecture config changed the frozen serialization")
    for name, raw_profile in profiles.items():
        if not isinstance(raw_profile, dict):
            raise ValueError(f"Profile {name!r} must be a JSON object")
        effective_batch = (
            int(raw_profile["batch_size"])
            * 2
            * int(raw_profile["gradient_accumulation"])
        )
        if effective_batch != int(protocol["target_effective_batch"]):
            raise ValueError(
                f"Profile {name!r} effective batch is {effective_batch}, expected "
                f"{protocol['target_effective_batch']}"
            )
    return configuration


def trainer_config(
    protocol: dict[str, object],
    profile: dict[str, object],
) -> dict[str, object]:
    return {
        "model": profile["model"],
        "model_backend": "sequence_classification",
        "trust_remote_code": bool(profile.get("trust_remote_code", False)),
        "epochs": int(protocol["epochs"]),
        "batch_size": int(profile["batch_size"]),
        "eval_batch_size": int(profile["eval_batch_size"]),
        "gradient_accumulation": int(profile["gradient_accumulation"]),
        "learning_rate": float(protocol["learning_rate"]),
        "weight_decay": float(protocol["weight_decay"]),
        "warmup_ratio": float(protocol["warmup_ratio"]),
        "max_length": int(protocol["max_length"]),
        "attention_implementation": profile["attention_implementation"],
        "train_subset": protocol["train_subset"],
        "sampling": protocol["sampling"],
        "loss_weighting": protocol["loss_weighting"],
        "lexical_hard_negative_strength": 0.0,
        "bucket_size_multiplier": 50,
        "dataloader_workers": 2,
        "prefetch_factor": 2,
        "tokenization_batch_size": 512,
        "tokenization_log_every": 50,
        "gradient_checkpointing": bool(profile["gradient_checkpointing"]),
        "symmetric_validation": bool(protocol["symmetric_validation"]),
        "label_smoothing": float(protocol["label_smoothing"]),
        "max_grad_norm": 1.0,
        "log_every": 50,
        "seed": int(protocol["seed"]),
    }


def embedded_sources() -> tuple[dict[str, str], str, str]:
    sources, _ = baseline_builder.embedded_sources()
    for relative in EXTRA_EMBEDDED_FILES:
        sources[relative.as_posix()] = (ROOT / relative).read_text(encoding="utf-8")
    frequency_content = SERIALIZATION_FREQUENCY.read_text(encoding="utf-8")
    sources["serialization/attribute_name_frequency.csv"] = frequency_content
    digest = hashlib.sha256()
    for relative, content in sorted(sources.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content.encode("utf-8"))
        digest.update(b"\0")
    serialization_digest = hashlib.sha256()
    serialization_digest.update(SERIALIZATION_VARIANT.encode("utf-8"))
    serialization_digest.update(b"\0")
    serialization_digest.update(
        sources["src/serialization_ablation.py"].encode("utf-8")
    )
    serialization_digest.update(b"\0")
    serialization_digest.update(frequency_content.encode("utf-8"))
    return sources, digest.hexdigest(), serialization_digest.hexdigest()


def heading_index(notebook: nbf.NotebookNode, heading: str) -> int:
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown" and cell.source.strip().splitlines()[0] == heading:
            return index
    raise ValueError(f"Notebook heading is missing: {heading}")


def code(value: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(value).strip())


def build_notebook(
    dataset: dict[str, object],
    protocol: dict[str, object],
    profile_name: str,
    profile: dict[str, object],
) -> nbf.NotebookNode:
    experiment = str(profile["experiment"])
    config = trainer_config(protocol, profile)
    checkpoint_dataset = str(profile.get("initial_checkpoint_dataset", ""))
    initial_checkpoint = (
        {
            "dataset": checkpoint_dataset,
            "manifest_sha256": profile["initial_checkpoint_manifest_sha256"],
        }
        if checkpoint_dataset
        else None
    )
    notebook = baseline_builder.build_notebook(
        dataset,
        config,
        experiment_name=experiment,
        experiment_title=f"Architecture baseline: {profile['architecture']}",
        experiment_description=(
            "Одна human-эпоха на frozen train и единая S2_VALUES_ONLY "
            "сериализация. IID, hard и OOD считаются после обучения в обоих "
            "порядках пары. Notebook пишет только в architecture_exps."
        ),
        initial_checkpoint=initial_checkpoint,
    )
    manifest = dataset.get("manifest")
    outputs = manifest.get("outputs") if isinstance(manifest, dict) else None
    if not isinstance(outputs, dict):
        raise ValueError("Frozen validation manifest has no output declarations")
    expected_validation_files = {
        relative: {
            "bytes": int(outputs[relative]["bytes"]),
            "sha256": str(outputs[relative]["sha256"]),
        }
        for relative in baseline_builder.REMOTE_FILES
    }
    bootstrap = next(
        cell
        for cell in notebook.cells
        if cell.cell_type == "code" and "EXPECTED_MANIFEST_SHA256" in cell.source
    )
    bootstrap.source = bootstrap.source.replace(
        f"REMOTE_FILES = {baseline_builder.REMOTE_FILES!r}\n",
        (
            f"REMOTE_FILES = {baseline_builder.REMOTE_FILES!r}\n"
            f"EXPECTED_VALIDATION_FILES = {expected_validation_files!r}\n"
        ),
    )
    old_manifest_validation = """manifest_path = exactly_one("validation_splits_manifest.json")
if file_sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
    raise RuntimeError("Attached validation Dataset manifest has changed")
validation_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if validation_manifest["ood_categories"] != ["Одежда", "Бытовая техника"]:
    raise RuntimeError("Unexpected OOD categories in attached Dataset")
attached_files = {
    relative: exactly_one(remote_name)
    for relative, remote_name in REMOTE_FILES.items()
}
for relative, path in attached_files.items():
    expected = validation_manifest["outputs"][relative]
    if path.stat().st_size != expected["bytes"] or file_sha256(path) != expected["sha256"]:
        raise RuntimeError(f"Attached Dataset file differs from manifest: {relative}")
"""
    new_manifest_validation = """manifest_path = exactly_one("validation_splits_manifest.json")
attached_manifest_sha256 = file_sha256(manifest_path)
validation_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if validation_manifest.get("version") != "human_v1":
    raise RuntimeError("Unexpected validation protocol version")
if validation_manifest.get("ood_categories") != ["Одежда", "Бытовая техника"]:
    raise RuntimeError("Unexpected OOD categories in attached Dataset")
attached_outputs = validation_manifest.get("outputs", {})
attached_files = {
    relative: exactly_one(remote_name)
    for relative, remote_name in REMOTE_FILES.items()
}
for relative, path in attached_files.items():
    expected = EXPECTED_VALIDATION_FILES[relative]
    declared = attached_outputs.get(relative)
    if not isinstance(declared, dict) or (
        int(declared.get("bytes", -1)) != expected["bytes"]
        or str(declared.get("sha256", "")) != expected["sha256"]
    ):
        raise RuntimeError(
            f"Attached validation manifest changes frozen file {relative}"
        )
    if (
        path.stat().st_size != expected["bytes"]
        or file_sha256(path) != expected["sha256"]
    ):
        raise RuntimeError(f"Attached Dataset file differs from frozen protocol: {relative}")
"""
    if old_manifest_validation not in bootstrap.source:
        raise RuntimeError("Could not adapt validation manifest checks in notebook")
    bootstrap.source = bootstrap.source.replace(
        old_manifest_validation,
        new_manifest_validation,
    ).replace(
        '"manifest_sha256": EXPECTED_MANIFEST_SHA256,',
        (
            '"protocol_manifest_sha256": EXPECTED_MANIFEST_SHA256,\n'
            '    "attached_manifest_sha256": attached_manifest_sha256,'
        ),
    )
    sources, source_hash, serialization_hash = embedded_sources()
    source_bundle = base64.b64encode(
        gzip.compress(
            json.dumps(
                sources,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
            compresslevel=9,
            mtime=0,
        )
    ).decode("ascii")
    remote_files = repr(baseline_builder.REMOTE_FILES)
    code_heading = heading_index(notebook, "## Код и frozen data")
    notebook.cells[code_heading].source = "## Код, S2_VALUES_ONLY и frozen data"
    notebook.cells[code_heading + 1] = code(
        f"""
        import base64
        import gzip

        EMBEDDED_SOURCE_BUNDLE_B64 = {source_bundle!r}
        EMBEDDED_SOURCES = json.loads(
            gzip.decompress(
                base64.b64decode(EMBEDDED_SOURCE_BUNDLE_B64)
            ).decode("utf-8")
        )
        EMBEDDED_SOURCE_SHA256 = {source_hash!r}
        SERIALIZATION_VARIANT = {SERIALIZATION_VARIANT!r}
        SERIALIZATION_SHA256 = {serialization_hash!r}
        RAW_DATASET_REF = {RAW_DATASET_REF!r}
        REMOTE_FILES = {remote_files}

        PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
        for relative, content in EMBEDDED_SOURCES.items():
            destination = PROJECT_ROOT / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--disable-pip-version-check",
                "--upgrade-strategy",
                "only-if-needed",
                "-r",
                str(PROJECT_ROOT / "requirements-architecture-baselines.txt"),
            ],
            check=True,
        )

        import numpy as np
        import pandas as pd
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from src.serialization_ablation import parse_attributes, serialize_product

        raw_candidates = list(INPUT_ROOT.glob("**/items_human.parquet"))
        if len(raw_candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one items_human.parquet from {{RAW_DATASET_REF}}, "
                f"found {{raw_candidates}}"
            )
        raw_items_path = raw_candidates[0]
        PREPARED_DIR.mkdir(parents=True, exist_ok=True)
        prepared_pair_names = {{
            "human/train_pairs.parquet": "train_pairs.parquet",
            "human/iid_validation_pairs.parquet": "iid_validation_pairs.parquet",
            "human/hard_validation_pairs.parquet": "hard_validation_pairs.parquet",
            "human/ood_validation_pairs.parquet": "ood_validation_pairs.parquet",
        }}
        for relative, destination_name in prepared_pair_names.items():
            destination = PREPARED_DIR / destination_name
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            destination.symlink_to(attached_files[relative])

        serialization_started = time.perf_counter()
        reference_items = pd.read_parquet(
            attached_files["human/items.parquet"],
            columns=["id", "name", "category"],
        )
        raw_items = pd.read_parquet(
            raw_items_path,
            columns=["id", "name", "attributes", "category"],
        )
        if reference_items["id"].duplicated().any() or raw_items["id"].duplicated().any():
            raise ValueError("Human item catalogues contain duplicate IDs")
        reference_ids = set(reference_items["id"].tolist())
        raw_ids = set(raw_items["id"].tolist())
        if reference_ids != raw_ids:
            raise ValueError(
                f"Raw/frozen human item IDs differ: missing={{len(reference_ids - raw_ids)}}, "
                f"extra={{len(raw_ids - reference_ids)}}"
            )
        raw_items = raw_items.set_index("id", verify_integrity=True).loc[
            reference_items["id"].to_numpy()
        ].reset_index()
        if not raw_items["category"].astype(str).reset_index(drop=True).equals(
            reference_items["category"].astype(str).reset_index(drop=True)
        ):
            raise ValueError("Raw/frozen item categories differ")
        if not raw_items["name"].astype(str).reset_index(drop=True).equals(
            reference_items["name"].astype(str).reset_index(drop=True)
        ):
            raise ValueError("Raw/frozen item names differ")

        frequency_path = PROJECT_ROOT / "serialization/attribute_name_frequency.csv"
        frequency = pd.read_csv(frequency_path)
        if "attribute_name" not in frequency or frequency["attribute_name"].duplicated().any():
            raise ValueError("Frozen attribute frequency table is invalid")
        key_rank = {{
            str(key): rank
            for rank, key in enumerate(frequency["attribute_name"].tolist())
        }}
        product_text = [
            serialize_product(
                row.name,
                parse_attributes(row.attributes),
                SERIALIZATION_VARIANT,
                set(),
                key_rank,
            )
            for row in raw_items.itertuples(index=False)
        ]
        if any(not text for text in product_text):
            raise ValueError("S2 serialization produced empty product text")
        prepared_items = pd.DataFrame({{
            "id": raw_items["id"].to_numpy(),
            "name": raw_items["name"].astype(str).to_numpy(),
            "category": raw_items["category"].astype(str).to_numpy(),
            "product_text": product_text,
        }})
        prepared_items_path = PREPARED_DIR / "items.parquet"
        prepared_items.to_parquet(
            prepared_items_path, index=False, compression="zstd"
        )

        pair_frames = {{
            name: pd.read_parquet(path, columns=["id1", "id2", "target"])
            for name, path in {{
                "train": PREPARED_DIR / "train_pairs.parquet",
                "iid": PREPARED_DIR / "iid_validation_pairs.parquet",
                "hard": PREPARED_DIR / "hard_validation_pairs.parquet",
                "ood": PREPARED_DIR / "ood_validation_pairs.parquet",
            }}.items()
        }}
        if not pair_frames["train"]["target"].isin([0.0, 1.0]).all():
            raise ValueError("Human train labels are not binary")
        train_ids = set(pair_frames["train"][["id1", "id2"]].to_numpy().reshape(-1))
        validation_ids = set(
            np.concatenate([
                pair_frames[name][["id1", "id2"]].to_numpy().reshape(-1)
                for name in ("iid", "hard", "ood")
            ])
        )
        if leaked := train_ids & validation_ids:
            raise ValueError(f"Frozen train/validation item leakage: {{len(leaked)}} IDs")
        S2_PREP_SECONDS = time.perf_counter() - serialization_started
        SERIALIZATION_REPORT = {{
            "variant": SERIALIZATION_VARIANT,
            "serialization_sha256": SERIALIZATION_SHA256,
            "frequency_file_sha256": file_sha256(frequency_path),
            "prepared_items_sha256": file_sha256(prepared_items_path),
            "items": len(prepared_items),
            "train_pairs": len(pair_frames["train"]),
            "validation_pairs": {{
                name: len(pair_frames[name]) for name in ("iid", "hard", "ood")
            }},
            "prep_seconds": S2_PREP_SECONDS,
        }}
        print(json.dumps(SERIALIZATION_REPORT, ensure_ascii=False, indent=2))
        """
    )

    artifacts_heading = heading_index(notebook, "## Артефакты и completion report")
    notebook.cells[artifacts_heading + 1] = code(
        f"""
        report_path = OUTPUT_DIR / "training_report.json"
        if not report_path.is_file():
            raise RuntimeError(f"Training finished without report: {{report_path}}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected_splits = {{"iid", "hard", "ood"}}
        if set(report.get("validation_splits", {{}})) != expected_splits:
            raise RuntimeError(
                f"Expected three validation splits, got "
                f"{{sorted(report.get('validation_splits', {{}}))}}"
            )
        predictions = {{
            split: f"{{OUTPUT_DIR.name}}/"
            f"{{report['validation_splits'][split]['predictions_file']}}"
            for split in ("iid", "hard", "ood")
        }}
        completion = {{
            "status": "complete",
            "run_id": EXPERIMENT_RUN_ID,
            "started_at_utc": EXPERIMENT_STARTED_AT_UTC,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ).replace("+00:00", "Z"),
            "experiment": OUTPUT_DIR.name,
            "architecture": {str(profile['architecture'])!r},
            "model": {str(profile['model'])!r},
            "dataset_ref": EXPECTED_DATASET_REF,
            "raw_dataset_ref": RAW_DATASET_REF,
            "initial_checkpoint_ref": INITIAL_CHECKPOINT_REF or {str(profile['model'])!r},
            "initial_checkpoint_manifest_sha256": (
                EXPECTED_CHECKPOINT_MANIFEST_SHA256 or None
            ),
            "kaggle_kernel_ref": (
                os.getenv("KAGGLE_KERNEL_RUN_ID")
                or os.getenv("KAGGLE_KERNEL_INFERENCE_RUN_ID")
                or ""
            ),
            "code_bundle_sha256": EMBEDDED_SOURCE_SHA256,
            "serialization": SERIALIZATION_VARIANT,
            "serialization_sha256": SERIALIZATION_SHA256,
            "serialization_prep_seconds": S2_PREP_SECONDS,
            "precision": report.get("amp_dtype"),
            "effective_batch": (
                int(TRAIN_CONFIG["batch_size"])
                * int(TRAIN_CONFIG["gradient_accumulation"])
                * int(report.get("world_size", 2))
            ),
            "technical_notes": {str(profile['technical_notes'])!r},
            "training_wall_seconds": training_wall_seconds,
            "serialization_report": SERIALIZATION_REPORT,
            "training_report": report,
            "artifacts": {{
                "checkpoint": OUTPUT_DIR.name,
                "training_report": f"{{OUTPUT_DIR.name}}/training_report.json",
                "training_config": f"{{OUTPUT_DIR.name}}/training_config.json",
                "predictions": predictions,
                "training_log": TRAIN_LOG.name,
            }},
        }}
        completion_path = WORKING_ROOT / "notebook_completed.json"
        completion_path.write_text(
            json.dumps(completion, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        summary = {{
            name: {{
                "macro_ap": metrics["macro_average_precision"],
                "overall_ap": metrics["overall_average_precision"],
                "roc_auc": metrics["roc_auc"],
                "inference_seconds": report["validation_seconds_by_split"][name],
                "predictions": predictions[name],
            }}
            for name, metrics in report["validation_splits"].items()
        }}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        """
    )
    notebook.cells[-2:] = shared.google_sheets_tracking_cells(
        sync_target="architecture"
    )
    for cell in notebook.cells:
        cell.metadata.setdefault("tags", ["architecture-baseline"])
    for index, cell in enumerate(notebook.cells):
        cell.id = hashlib.sha256(
            f"architecture-baseline-v1:{profile_name}:{index}".encode("utf-8")
        ).hexdigest()[:12]
    notebook.metadata["product_matching_training"].update(
        {
            "template": "architecture_baseline_v1",
            "profile": profile_name,
            "architecture": profile["architecture"],
            "serialization": SERIALIZATION_VARIANT,
            "serialization_sha256": serialization_hash,
            "raw_dataset": RAW_DATASET_REF,
            "google_sheet": "architecture_exps",
            "effective_batch": int(protocol["target_effective_batch"]),
        }
    )
    nbf.validate(notebook)
    return notebook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=["all", *NOTEBOOK_FILENAMES],
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_validation_dataset()
    configuration = load_configuration()
    protocol = configuration["protocol"]
    profiles = configuration["profiles"]
    selected = list(NOTEBOOK_FILENAMES) if args.profile == "all" else [args.profile]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for profile_name in selected:
        notebook = build_notebook(
            dataset,
            protocol,
            profile_name,
            profiles[profile_name],
        )
        destination = args.output_dir / NOTEBOOK_FILENAMES[profile_name]
        nbf.write(notebook, destination)
        print(f"Wrote notebook: {destination}")
    print(f"Frozen validation Dataset: {VALIDATION_DATASET_REF}")
    print(f"Raw human Dataset: {RAW_DATASET_REF}")
    print(f"Protocol SHA-256: {canonical_sha256(configuration)}")


if __name__ == "__main__":
    main()
