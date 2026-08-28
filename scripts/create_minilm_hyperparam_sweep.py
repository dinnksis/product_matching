"""Create one Kaggle notebook per MiniLM hyperparameter configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "notebooks/minilm_5ep_team_ablation/minilm_5ep_team_ablation_2xt4.ipynb"
DEFAULT_OUTPUT = ROOT / "notebooks/minilm_5ep_hyperparam_sweep"


def canonical_sha256(value: dict[str, object]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def configurations() -> list[tuple[int, float, int]]:
    result: list[tuple[int, float, int]] = []
    for max_length in (384, 512):
        result.extend(
            [
                (max_length, 2e-5, 1),
                (max_length, 5e-6, 1),
                (max_length, 1e-5, 1),
                (max_length, 3e-5, 1),
                (max_length, 5e-5, 1),
                (max_length, 2e-5, 2),
                (max_length, 2e-5, 3),
            ]
        )
    return result


def slug_number(index: int, max_length: int, learning_rate: float, epochs: int) -> str:
    lr = f"{learning_rate:.0e}".replace("-", "m")
    return f"sweep_{index:02d}_len{max_length}_lr{lr}_ep{epochs}"


def build_config(template: dict[str, object], *, max_length: int, learning_rate: float, epochs: int) -> dict[str, object]:
    notebook = json.loads(json.dumps(template))
    config_cell = next(
        cell for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "CANONICAL_TRAIN_CONFIG" in "".join(cell.get("source", []))
    )
    source = "".join(config_cell["source"])
    start = source.index("CANONICAL_TRAIN_CONFIG = ") + len("CANONICAL_TRAIN_CONFIG = ")
    end = source.index("\nEXPECTED_TRAIN_RECIPE_SHA256", start)
    config_text = source[start:end]
    config = json.loads(json.dumps(eval(config_text, {"__builtins__": {}}, {})))
    config["max_length"] = max_length
    config["learning_rate"] = learning_rate
    config["epochs"] = epochs
    recipe_hash = canonical_sha256(config)
    replacement = (
        f"CANONICAL_TRAIN_CONFIG = {config!r}"
        f"\nEXPECTED_TRAIN_RECIPE_SHA256 = {recipe_hash!r}"
    )
    config_cell["source"] = source[: source.index("CANONICAL_TRAIN_CONFIG = ")] + replacement + source[source.index("\ndef canonical_json_sha256", end):]

    index = int(template["_sweep_index"])
    experiment_slug = slug_number(index, max_length, learning_rate, epochs)
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        text = "".join(cell.get("source", []))
        for old in (
            "minilm_5ep_team_data_loss_ablation",
            "minilm_5ep_team_data_loss_ablation.log",
        ):
            text = text.replace(old, f"minilm_5ep_{experiment_slug}")
        cell["source"] = text
    notebook["metadata"].setdefault("product_matching_training", {})["sweep"] = {
        "index": index,
        "max_length": max_length,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "experiment": experiment_slug,
        "recipe_sha256": recipe_hash,
    }
    notebook.pop("_sweep_index", None)
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    template = json.loads(args.template.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    for index, (max_length, learning_rate, epochs) in enumerate(configurations(), start=1):
        source = dict(template)
        source["_sweep_index"] = index
        notebook = build_config(
            source,
            max_length=max_length,
            learning_rate=learning_rate,
            epochs=epochs,
        )
        path = args.output / f"{slug_number(index, max_length, learning_rate, epochs)}.ipynb"
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{index:02d} {path.name} max_length={max_length} learning_rate={learning_rate:g} epochs={epochs}")


if __name__ == "__main__":
    main()
