#!/usr/bin/env python3
"""Build the MiniLM 5ep SFT experiment with 18 category-specific heads."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from textwrap import dedent, indent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
BASE_NOTEBOOK = (
    ROOT
    / "notebooks"
    / "minilm_5ep_team_ablation"
    / "minilm_5ep_team_ablation_2xt4.ipynb"
)
DEFAULT_OUTPUT = (
    ROOT
    / "notebooks"
    / "minilm_5ep_category_heads"
    / "minilm_5ep_category_heads_18_2xt4.ipynb"
)
EXPERIMENT_LABEL = "minilm_5ep_category_heads_18"
EXPERIMENT_SHEET = "sft_exps"
EXPERIMENT_NOTES = (
    "18 category-specific logits; train and IID/hard inference select the pair's "
    "category head; unseen OOD categories use the mean logit across all 18 heads."
)
OUTPUT_DIRECTORY_NAME = EXPERIMENT_LABEL
CATEGORY_HEAD_NAMES = (
    "Автотовары",
    "Аптека",
    "Бытовая химия",
    "Галантерея и аксессуары",
    "Детские товары",
    "Дом и сад",
    "Канцелярские товары",
    "Красота и гигиена",
    "Мебель",
    "Музыкальные инструменты",
    "Обувь",
    "Продукты питания",
    "Спорт и отдых",
    "Строительство и ремонт",
    "Товары для животных",
    "Хобби и творчество",
    "Электроника",
    "Ювелирные изделия",
)
OOD_CATEGORIES = ("Бытовая техника", "Одежда")
UNSEEN_CATEGORY_FALLBACK = "mean_logit"


def _replace_once(source: str, old: str, new: str, description: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(
            f"Expected exactly one {description} block in the frozen trainer, "
            f"found {count}"
        )
    return source.replace(old, new, 1)


def _python_block(value: str, indentation: int) -> str:
    return indent(dedent(value).strip("\n"), " " * indentation) + "\n"


def patch_training_source(source: str) -> str:
    """Turn the frozen single-logit trainer into a routed 18-head trainer."""
    category_config = {
        "count": len(CATEGORY_HEAD_NAMES),
        "names": list(CATEGORY_HEAD_NAMES),
        "training_routing": "category_1",
        "known_validation_routing": "category_1",
        "unseen_validation_fallback": UNSEEN_CATEGORY_FALLBACK,
        "expected_unseen_ood_categories": list(OOD_CATEGORIES),
    }
    source = _replace_once(
        source,
        'DEFAULT_CONFIG = ROOT / "configs" / "cross_encoder_minilm.json"\n',
        (
            'DEFAULT_CONFIG = ROOT / "configs" / "cross_encoder_minilm.json"\n'
            f"CATEGORY_HEAD_NAMES = {CATEGORY_HEAD_NAMES!r}\n"
            "CATEGORY_TO_HEAD = {\n"
            "    category: index for index, category in enumerate(CATEGORY_HEAD_NAMES)\n"
            "}\n"
            f"CATEGORY_HEAD_CONFIG = {category_config!r}\n"
        ),
        "default config",
    )

    helper = dedent(
        '''

        def select_category_logits(
            logits: torch.Tensor,
            pair_indices: list[int] | torch.Tensor,
            categories: list[str],
            *,
            allow_unseen: bool,
        ) -> torch.Tensor:
            """Select the one trained head associated with each pair category."""
            if logits.ndim != 2 or logits.shape[1] != len(CATEGORY_HEAD_NAMES):
                raise RuntimeError(
                    "Category-head model returned logits with shape "
                    f"{tuple(logits.shape)}; expected [batch, {len(CATEGORY_HEAD_NAMES)}]"
                )
            positions = (
                pair_indices.detach().cpu().tolist()
                if isinstance(pair_indices, torch.Tensor)
                else pair_indices
            )
            category_ids = [
                CATEGORY_TO_HEAD.get(str(categories[int(position)]), -1)
                for position in positions
            ]
            unseen = sorted(
                {
                    str(categories[int(position)])
                    for position, category_id in zip(positions, category_ids)
                    if category_id < 0
                }
            )
            if unseen and not allow_unseen:
                raise RuntimeError(f"Training contains categories without heads: {unseen}")
            head_indices = torch.tensor(
                category_ids, dtype=torch.long, device=logits.device
            )
            gathered = logits.gather(
                1, head_indices.clamp_min(0).unsqueeze(1)
            ).squeeze(1)
            if unseen:
                # Frozen OOD policy: the two held-out categories have no trained head.
                # Average logits (not probabilities) so every OOD pair still receives
                # one deterministic continuous score.
                gathered = torch.where(
                    head_indices >= 0,
                    gathered,
                    logits.mean(dim=1),
                )
            return gathered
        '''
    )
    source = _replace_once(
        source,
        "\ndef evaluate(\n",
        helper + "\n\ndef evaluate(\n",
        "evaluate definition",
    )

    source = _replace_once(
        source,
        _python_block(
            '''
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = relevance_logits(model(**batch), model_backend)
            probabilities = logits.float().sigmoid().cpu().tolist()
            ''',
            12,
        ),
        _python_block(
            """
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                all_logits = relevance_logits(model(**batch), model_backend)
                logits = select_category_logits(
                    all_logits,
                    pair_indices,
                    categories,
                    allow_unseen=True,
                )
            probabilities = logits.float().sigmoid().cpu().tolist()
            """,
            12,
        ),
        "validation logit selection",
    )

    source = _replace_once(
        source,
        _python_block(
            """
            if logits.ndim == 2 and logits.shape[-1] == 1:
                logits = logits[:, 0]
            if logits.ndim != 1:
                raise RuntimeError(
                    f"Model backend {model_backend!r} returned scores with shape "
                    f"{tuple(logits.shape)}; expected one score per pair"
                )
            return logits
            """,
            4,
        ),
        _python_block(
            """
            if logits.ndim == 2 and logits.shape[-1] == 1:
                logits = logits[:, 0]
            if logits.ndim not in {1, 2}:
                raise RuntimeError(
                    f"Model backend {model_backend!r} returned scores with shape "
                    f"{tuple(logits.shape)}; expected one or category-specific scores per pair"
                )
            return logits
            """,
            4,
        ),
        "relevance logit shape contract",
    )

    source = _replace_once(
        source,
        _python_block(
            """
            validation_categories = {
                name: validation["category_1"].astype(str).tolist()
                for name, validation in validations.items()
            }
            lexical_similarities = (
            """,
            4,
        ),
        _python_block(
            """
            validation_categories = {
                name: validation["category_1"].astype(str).tolist()
                for name, validation in validations.items()
            }
            observed_train_categories = tuple(sorted(set(train_categories)))
            if observed_train_categories != CATEGORY_HEAD_NAMES:
                raise RuntimeError(
                    "Frozen train categories differ from the 18-head mapping: "
                    f"{observed_train_categories}"
                )
            expected_unseen_ood = set(
                CATEGORY_HEAD_CONFIG["expected_unseen_ood_categories"]
            )
            observed_unseen_ood = set(validation_categories.get("ood", ())) - set(
                CATEGORY_HEAD_NAMES
            )
            if observed_unseen_ood != expected_unseen_ood:
                raise RuntimeError(
                    "Frozen OOD categories differ from the declared fallback set: "
                    f"{sorted(observed_unseen_ood)}"
                )
            lexical_similarities = (
            """,
            4,
        ),
        "category arrays",
    )

    source = _replace_once(
        source,
        _python_block(
            """
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model,
                num_labels=1,
                attn_implementation=args.attention_implementation,
                trust_remote_code=args.trust_remote_code,
                **extra_model_load_kwargs,
            )
            model.config.id2label = {0: "MATCH_SCORE"}
            model.config.label2id = {"MATCH_SCORE": 0}
            """,
            8,
        ),
        _python_block(
            """
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model,
                num_labels=len(CATEGORY_HEAD_NAMES),
                ignore_mismatched_sizes=True,
                attn_implementation=args.attention_implementation,
                trust_remote_code=args.trust_remote_code,
                **extra_model_load_kwargs,
            )
            model.config.id2label = {
                index: category
                for index, category in enumerate(CATEGORY_HEAD_NAMES)
            }
            model.config.label2id = {
                category: index
                for index, category in enumerate(CATEGORY_HEAD_NAMES)
            }
            model.config.category_head_names = list(CATEGORY_HEAD_NAMES)
            model.config.category_head_routing = "category_1"
            model.config.unseen_category_fallback = "mean_logit"
            """,
            8,
        ),
        "sequence classification model initialization",
    )

    source = _replace_once(
        source,
        _python_block(
            """
            logits = relevance_logits(
                training_model(**batch), args.model_backend
            )
            raw_loss, loss_metrics = loss_hook.compute(
                logits=logits.float(),
            """,
            20,
        ),
        _python_block(
            """
            all_logits = relevance_logits(
                training_model(**batch), args.model_backend
            )
            logits = select_category_logits(
                all_logits,
                pair_indices,
                train_categories,
                allow_unseen=False,
            )
            raw_loss, loss_metrics = loss_hook.compute(
                logits=logits.float(),
            """,
            20,
        ),
        "training logit selection",
    )

    source = _replace_once(
        source,
        '                    "loss_hook": loss_hook.metadata,\n',
        (
            '                    "loss_hook": loss_hook.metadata,\n'
            '                    "category_heads": CATEGORY_HEAD_CONFIG,\n'
        ),
        "startup category-head report",
    )
    source = _replace_once(
        source,
        (
            '            "training_loss_weighting": args.loss_weighting,\n'
            '            "loss_hook": loss_hook.metadata,\n'
        ),
        (
            '            "training_loss_weighting": args.loss_weighting,\n'
            '            "loss_hook": loss_hook.metadata,\n'
            '            "category_heads": CATEGORY_HEAD_CONFIG,\n'
            '            "validation_unseen_categories": {\n'
            '                name: sorted(set(categories) - set(CATEGORY_HEAD_NAMES))\n'
            '                for name, categories in validation_categories.items()\n'
            '            },\n'
        ),
        "final category-head report",
    )
    compile(source, "patched_train_cross_encoder.py", "exec")
    return source


def _heading_index(notebook: nbf.NotebookNode, heading: str) -> int:
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown" and cell.source.strip().splitlines()[0] == heading:
            return index
    raise ValueError(f"Notebook heading is missing: {heading}")


def _embedded_training_source(notebook: nbf.NotebookNode) -> str:
    for cell in notebook.cells:
        if cell.cell_type != "code" or not cell.source.startswith("EMBEDDED_SOURCES = "):
            continue
        module = ast.parse(cell.source)
        assignment = module.body[0]
        if not isinstance(assignment, ast.Assign):
            continue
        sources = ast.literal_eval(assignment.value)
        source = sources.get("scripts/train_cross_encoder.py")
        if not isinstance(source, str):
            raise ValueError("Frozen notebook lacks scripts/train_cross_encoder.py")
        return source
    raise ValueError("Frozen notebook lacks the EMBEDDED_SOURCES cell")


def build_category_heads_notebook(
    base_notebook: Path = BASE_NOTEBOOK,
) -> nbf.NotebookNode:
    notebook = nbf.read(base_notebook, as_version=4)
    patched_training_source = patch_training_source(
        _embedded_training_source(notebook)
    )
    patch_sha256 = hashlib.sha256(
        patched_training_source.encode("utf-8")
    ).hexdigest()
    category_config = {
        "count": len(CATEGORY_HEAD_NAMES),
        "names": list(CATEGORY_HEAD_NAMES),
        "training_routing": "category_1",
        "known_validation_routing": "category_1",
        "unseen_validation_fallback": UNSEEN_CATEGORY_FALLBACK,
        "expected_unseen_ood_categories": list(OOD_CATEGORIES),
    }

    notebook.cells[0].source = dedent(
        """
        # MiniLM 5ep: 18 category-specific outputs

        Контролируемый SFT-эксперимент на baseline human-данных. Encoder стартует
        с того же frozen MiniLM checkpoint после пяти эпох pretraining, но вместо
        одного logit используется 18 выходов — по одному на каждую train-категорию.
        BCE обновляет только выбранную категорийную голову. Для двух невиданных OOD
        категорий итоговый logit равен среднему по всем 18 головам.
        """
    ).strip()

    routing_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code"
        and "experiment-routing" in cell.metadata.get("tags", [])
    ]
    if len(routing_cells) != 1:
        raise ValueError(f"Expected one experiment routing cell, found {len(routing_cells)}")
    routing_cells[0].source = dedent(
        f"""
        EXPERIMENT_LABEL = {EXPERIMENT_LABEL!r}
        EXPERIMENT_SHEET = {EXPERIMENT_SHEET!r}  # pretrain_exps | sft_exps | data_exps
        EXPERIMENT_NOTES = {EXPERIMENT_NOTES!r}
        """
    ).strip()

    training_heading = _heading_index(
        notebook, "## 🔒 Training and IID/hard/OOD validation"
    )
    patch_cell = nbf.v4.new_code_cell(
        dedent(
            f"""
            # Controlled architecture patch for this SFT experiment.
            PATCHED_TRAINING_SOURCE = {patched_training_source!r}
            CATEGORY_HEAD_PATCH_SHA256 = {patch_sha256!r}
            CATEGORY_HEAD_CONFIG = {category_config!r}
            patched_script_path = PROJECT_ROOT / "scripts/train_cross_encoder.py"
            patched_script_path.write_text(PATCHED_TRAINING_SOURCE, encoding="utf-8")
            if file_sha256(patched_script_path) != CATEGORY_HEAD_PATCH_SHA256:
                raise RuntimeError("Category-head trainer hash differs after materialization")
            OUTPUT_DIR = WORKING_ROOT / {OUTPUT_DIRECTORY_NAME!r}
            TRAIN_LOG = WORKING_ROOT / {f'{OUTPUT_DIRECTORY_NAME}.log'!r}
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            print(json.dumps({{
                "category_head_patch_sha256": CATEGORY_HEAD_PATCH_SHA256,
                "category_heads": CATEGORY_HEAD_CONFIG,
                "output_dir": str(OUTPUT_DIR),
            }}, ensure_ascii=False, indent=2))
            """
        ).strip()
    )
    patch_cell["id"] = "category-heads-patch"
    patch_cell.metadata["tags"] = ["variant-setup", "category-heads"]
    patch_heading = nbf.v4.new_markdown_cell("## 18-head category routing")
    patch_heading["id"] = "category-heads-heading"
    patch_heading.metadata["tags"] = ["variant-setup", "category-heads"]
    notebook.cells[training_heading:training_heading] = [
        patch_heading,
        patch_cell,
    ]

    artifacts_heading = _heading_index(notebook, "## 🔒 Artifacts and completion report")
    completion_cell = notebook.cells[artifacts_heading + 1]
    completion_cell.source = _replace_once(
        completion_cell.source,
        '    "train_data": TRAIN_DATA_REPORT,\n',
        (
            '    "train_data": TRAIN_DATA_REPORT,\n'
            '    "category_head_patch_sha256": CATEGORY_HEAD_PATCH_SHA256,\n'
            '    "category_heads": CATEGORY_HEAD_CONFIG,\n'
        ),
        "completion category-head provenance",
    )

    metadata = notebook.metadata["product_matching_training"]
    metadata.update(
        {
            "experiment": EXPERIMENT_LABEL,
            "default_experiment_sheet": EXPERIMENT_SHEET,
            "template": "minilm_5ep_category_heads_18_v1",
            "category_heads": category_config,
            "category_head_patch_sha256": patch_sha256,
        }
    )
    nbf.validate(notebook)
    return notebook


def main() -> None:
    notebook = build_category_heads_notebook()
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, DEFAULT_OUTPUT)
    print(f"Wrote notebook: {DEFAULT_OUTPUT}")
    print(f"Experiment: {EXPERIMENT_LABEL}")
    print(f"Comparison sheet: {EXPERIMENT_SHEET}")
    print(f"Category heads: {len(CATEGORY_HEAD_NAMES)}")


if __name__ == "__main__":
    main()
