"""Build a Kaggle notebook and its private 10k inference sample.

Kaggle limits notebook source to 1 MiB, so the fixed, stratified sample is
written as a Parquet file for a private Kaggle Dataset instead of being
embedded in the notebook source.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "notebooks" / "qwen3_vllm_inference_10k.ipynb"
DATASET_DIR = ROOT / ".kaggle" / "datasets" / "product-matching-qwen3-sample"
SAMPLE_PATH = DATASET_DIR / "qwen3_inference_sample_10k.parquet"
SAMPLE_SIZE = 10_000
RANDOM_STATE = 42


WORKER_SOURCE = r"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

# FlashInfer tries to JIT-link against an unversioned libcuda.so that is absent
# from the Kaggle T4 image. vLLM's Triton backend supports all attention types
# and CUDA compute capabilities without that FlashInfer linker path.
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from vllm import LLM


WORKING_DIR = Path("/kaggle/working")
INPUT_PATH = WORKING_DIR / "reranker_input.csv.gz"
PREDICTION_PATH = WORKING_DIR / "qwen3_reranker_predictions.parquet"
RUNTIME_PATH = WORKING_DIR / "qwen3_reranker_runtime.json"

MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
MAX_ITEM_TOKENS = 448
MAX_MODEL_LEN = 1024
EXPECTED_GPU_COUNT = 2

CHAT_TEMPLATE = r'''<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Determine whether the Query and Document describe the same marketplace product. Treat wording and schema differences as irrelevant, but distinguish genuinely different models or material product configurations.
<Query>: {{ messages | selectattr("role", "eq", "query") | map(attribute="content") | first }}
<Document>: {{ messages | selectattr("role", "eq", "document") | map(attribute="content") | first }}<|im_end|>
<|im_start|>assistant
<think>

</think>
'''


def batched_truncate(tokenizer, texts: list[str], max_tokens: int, batch_size: int = 128):
    truncated_texts: list[str] = []
    original_lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, add_special_tokens=False, truncation=False)["input_ids"]
        original_lengths.extend(len(token_ids) for token_ids in encoded)
        truncated_texts.extend(
            tokenizer.batch_decode(
                [token_ids[:max_tokens] for token_ids in encoded],
                skip_special_tokens=True,
            )
        )
    return truncated_texts, np.asarray(original_lengths, dtype=np.int32)


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    gpu_count = torch.cuda.device_count()
    if gpu_count != EXPECTED_GPU_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_GPU_COUNT} GPUs, got {gpu_count}")

    frame = pd.read_csv(INPUT_PATH, compression="gzip")
    queries = frame["text1"].astype(str).tolist()
    documents = frame["text2"].astype(str).tolist()

    tokenizer_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    queries, query_lengths = batched_truncate(tokenizer, queries, MAX_ITEM_TOKENS)
    documents, document_lengths = batched_truncate(tokenizer, documents, MAX_ITEM_TOKENS)
    tokenization_seconds = time.perf_counter() - tokenizer_start

    load_start = time.perf_counter()
    llm = LLM(
        model=MODEL_NAME,
        runner="pooling",
        hf_overrides={
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
        tensor_parallel_size=gpu_count,
        distributed_executor_backend="mp",
        dtype="float16",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=128,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        seed=42,
    )
    model_load_seconds = time.perf_counter() - load_start

    inference_start = time.perf_counter()
    outputs = llm.score(queries, documents, chat_template=CHAT_TEMPLATE)
    inference_seconds = time.perf_counter() - inference_start
    scores = np.asarray([output.outputs.score for output in outputs], dtype=np.float32)
    if len(scores) != len(frame) or not np.isfinite(scores).all():
        raise RuntimeError("vLLM returned missing or non-finite scores")

    predictions = frame[["pair_index", "id1", "id2", "target", "category"]].copy()
    predictions["predict"] = scores
    predictions.to_parquet(PREDICTION_PATH, index=False)

    gpu_info = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
    all_lengths = np.concatenate([query_lengths, document_lengths])
    runtime = {
        "model": MODEL_NAME,
        "attention_backend": os.environ["VLLM_ATTENTION_BACKEND"],
        "pairs": int(len(frame)),
        "gpu_count": gpu_count,
        "gpu_info_after_inference": gpu_info,
        "max_item_tokens": MAX_ITEM_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        "tokenization_seconds": tokenization_seconds,
        "model_load_seconds": model_load_seconds,
        "inference_seconds": inference_seconds,
        "pairs_per_second": len(frame) / inference_seconds,
        "token_length_p50": float(np.quantile(all_lengths, 0.50)),
        "token_length_p95": float(np.quantile(all_lengths, 0.95)),
        "token_length_p99": float(np.quantile(all_lengths, 0.99)),
        "token_length_max": int(all_lengths.max()),
        "item_truncation_rate": float((all_lengths > MAX_ITEM_TOKENS).mean()),
    }
    RUNTIME_PATH.write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(runtime, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
"""


def markdown(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def load_sample() -> pd.DataFrame:
    items = pd.read_parquet(ROOT / "data" / "items_human.parquet").set_index("id")
    matches = pd.read_parquet(ROOT / "data" / "matches.parquet")
    categories = matches["id1"].map(items["category"])
    strata = categories.astype(str) + "__" + matches["target"].astype(str)
    _, sample_indices = train_test_split(
        matches.index,
        test_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        stratify=strata,
    )
    sample_matches = matches.loc[sample_indices].copy().sort_index()
    left = items.loc[sample_matches["id1"], ["name", "attributes", "category"]]
    right = items.loc[sample_matches["id2"], ["name", "attributes"]]
    sample = pd.DataFrame(
        {
            "pair_index": sample_matches.index.to_numpy(),
            "id1": sample_matches["id1"].to_numpy(),
            "id2": sample_matches["id2"].to_numpy(),
            "target": sample_matches["target"].astype(int).to_numpy(),
            "category": left["category"].to_numpy(),
            "name1": left["name"].to_numpy(),
            "attributes1": left["attributes"].to_numpy(),
            "name2": right["name"].to_numpy(),
            "attributes2": right["attributes"].to_numpy(),
        }
    )
    if len(sample) != SAMPLE_SIZE or sample.isna().any().any():
        raise RuntimeError("Failed to create a complete 10k sample")
    return sample


def build_notebook(sample: pd.DataFrame) -> nbf.NotebookNode:
    cells = [
        markdown(
            """
            # Qwen3-Reranker-0.6B: vLLM inference на 10 000 товарных пар

            Notebook проверяет zero-shot inference исходной модели
            `Qwen/Qwen3-Reranker-0.6B` на стратифицированной выборке ручной
            разметки. Обучения здесь нет. Цели эксперимента:

            - проверить совместимость оригинального checkpoint с `vLLM.score`;
            - измерить model-load time, throughput и длины входов на двух T4;
            - получить overall и macro average precision до fine-tuning;
            - сохранить predictions и hard examples в `/kaggle/working`.

            Выборка содержит только 10 000 пар и подключается из отдельного
            приватного Kaggle Dataset. Notebook приватный; Kaggle-токен и полные
            parquet-файлы в него не включаются.
            """
        ),
        code(
            r"""
            import importlib.metadata
            import json
            import os
            import subprocess
            import sys
            import time
            from pathlib import Path

            WORKING_DIR = Path("/kaggle/working")
            WORKING_DIR.mkdir(parents=True, exist_ok=True)
            VLLM_VERSION = "0.14.0"

            try:
                installed_vllm = importlib.metadata.version("vllm")
            except importlib.metadata.PackageNotFoundError:
                installed_vllm = None

            if installed_vllm != VLLM_VERSION:
                print(f"Installing vLLM {VLLM_VERSION}; found {installed_vllm!r}")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", f"vllm=={VLLM_VERSION}"],
                    check=True,
                )
            else:
                print(f"vLLM {installed_vllm} is already installed")

            print(subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True).stdout)
            """
        ),
        code(
            """
            import json
            import re

            import pandas as pd

            sample_candidates = list(
                Path("/kaggle/input").glob("**/qwen3_inference_sample_10k.parquet")
            )
            if len(sample_candidates) != 1:
                raise RuntimeError(
                    "Expected one attached qwen3_inference_sample_10k.parquet, "
                    f"found {sample_candidates}"
                )
            pairs = pd.read_parquet(sample_candidates[0])
            assert len(pairs) == 10_000
            assert set(pairs["target"].unique()) == {0, 1}
            print(
                f"Loaded {len(pairs):,} pairs from {sample_candidates[0]}; "
                f"positive rate={pairs.target.mean():.2%}"
            )
            display(pd.crosstab(pairs["category"], pairs["target"], margins=True))
            """
        ),
        markdown(
            """
            ## Сериализация карточек

            Ключи не удаляются по глобальной частоте. Сначала идут потенциальные
            идентификаторы и variant-поля, затем все остальные атрибуты в
            детерминированном порядке. Очень длинное отдельное значение и общий
            текст ограничиваются защитными лимитами; точное token-truncation до 448
            токенов на карточку выполняется внутри inference worker.
            """
        ),
        code(
            r"""
            PRIORITY_PATTERNS = [
                r"артикул|партномер|oem|штрих|ean|gtin|код товара|код производителя|sku",
                r"бренд|производитель|модель",
                r"тип|вид ",
                r"размер|цвет|объем|объём|вес|количество|комплектац",
            ]
            MAX_VALUE_CHARS = 500
            MAX_ITEM_CHARS = 8_000


            def clean(value):
                return " ".join(str(value).replace("ё", "е").split())


            def key_priority(key):
                normalized = clean(key).lower()
                for priority, pattern in enumerate(PRIORITY_PATTERNS):
                    if re.search(pattern, normalized):
                        return priority, normalized
                return len(PRIORITY_PATTERNS), normalized


            def serialize_item(category, name, raw_attributes):
                attributes = json.loads(raw_attributes)
                lines = [f"Категория: {clean(category)}", f"Название: {clean(name)}", "Атрибуты:"]
                for key, value in sorted(attributes.items(), key=lambda item: key_priority(item[0])):
                    value = clean(value)
                    if not value:
                        continue
                    lines.append(f"{clean(key)}: {value[:MAX_VALUE_CHARS]}")
                return "\n".join(lines)[:MAX_ITEM_CHARS]


            pairs["text1"] = [
                serialize_item(category, name, attributes)
                for category, name, attributes in zip(pairs.category, pairs.name1, pairs.attributes1)
            ]
            pairs["text2"] = [
                serialize_item(category, name, attributes)
                for category, name, attributes in zip(pairs.category, pairs.name2, pairs.attributes2)
            ]
            display(pairs[["text1", "text2", "target"]].head(2))
            print(pairs[["text1", "text2"]].map(len).describe(percentiles=[.5, .9, .95, .99]))

            input_columns = ["pair_index", "id1", "id2", "target", "category", "text1", "text2"]
            input_path = WORKING_DIR / "reranker_input.csv.gz"
            pairs[input_columns].to_csv(input_path, index=False, compression="gzip")
            print(f"Wrote {input_path} ({input_path.stat().st_size / 2**20:.2f} MiB)")
            """
        ),
        markdown(
            """
            ## vLLM inference

            Worker запускается отдельным Python-процессом. Это изолирует vLLM и
            его CUDA/PyTorch зависимости от уже запущенного Jupyter kernel. Исходный
            Qwen checkpoint подключается к pooling runner через официальные
            `hf_overrides`; `tensor_parallel_size=2` задействует обе T4. Backend
            внимания явно задан как `TRITON_ATTN`: он совместим с T4 и не требует
            проблемной FlashInfer JIT-сборки в Kaggle image.
            """
        ),
        code(
            f'''
            WORKER_PATH = WORKING_DIR / "run_qwen3_vllm_worker.py"
            WORKER_PATH.write_text({WORKER_SOURCE!r}, encoding="utf-8")
            print(f"Worker written to {{WORKER_PATH}}")

            worker_start = time.perf_counter()
            subprocess.run([sys.executable, str(WORKER_PATH)], check=True)
            worker_wall_seconds = time.perf_counter() - worker_start
            print(f"Worker wall time: {{worker_wall_seconds:.1f}} seconds")
            '''
        ),
        markdown("## Метрики и диагностические примеры"),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            from sklearn.metrics import average_precision_score

            predictions = pd.read_parquet(WORKING_DIR / "qwen3_reranker_predictions.parquet")
            runtime = json.loads((WORKING_DIR / "qwen3_reranker_runtime.json").read_text())
            assert len(predictions) == 10_000
            assert np.isfinite(predictions["predict"]).all()

            per_category_ap = predictions.groupby("category").apply(
                lambda part: average_precision_score(part["target"], part["predict"]),
                include_groups=False,
            ).sort_values()
            macro_ap = float(per_category_ap.mean())
            overall_ap = float(average_precision_score(predictions["target"], predictions["predict"]))

            metrics = {
                **runtime,
                "overall_average_precision": overall_ap,
                "macro_average_precision": macro_ap,
                "score_min": float(predictions["predict"].min()),
                "score_median": float(predictions["predict"].median()),
                "score_max": float(predictions["predict"].max()),
                "per_category_average_precision": per_category_ap.to_dict(),
            }
            (WORKING_DIR / "qwen3_reranker_metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            display(pd.Series({
                "macro AP": macro_ap,
                "overall AP": overall_ap,
                "pairs/second": runtime["pairs_per_second"],
                "inference seconds": runtime["inference_seconds"],
                "model load seconds": runtime["model_load_seconds"],
                "item truncation rate": runtime["item_truncation_rate"],
            }).to_frame("value"))
            display(per_category_ap.to_frame("average_precision"))

            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            for target, label, color in [(0, "не дубль", "#64748b"), (1, "дубль", "#f97316")]:
                values = predictions.loc[predictions.target == target, "predict"]
                axes[0].hist(values, bins=50, density=True, histtype="step", linewidth=2,
                             label=label, color=color)
            axes[0].set(title="Zero-shot score distribution", xlabel="Qwen3 reranker score", ylabel="density")
            axes[0].legend()
            per_category_ap.plot.barh(ax=axes[1], color="#0f766e")
            axes[1].set(title="Average precision по категориям", xlabel="AP", ylabel="")
            fig.tight_layout()
            figure_path = WORKING_DIR / "qwen3_reranker_diagnostics.png"
            fig.savefig(figure_path, dpi=150, bbox_inches="tight")
            plt.show()
            """
        ),
        code(
            """
            review = predictions.merge(
                pairs[["pair_index", "name1", "name2"]], on="pair_index", how="left", validate="one_to_one"
            )
            hard_negatives = review[review.target == 0].nlargest(20, "predict").assign(slice="high_scored_negative")
            hard_positives = review[review.target == 1].nsmallest(20, "predict").assign(slice="low_scored_positive")
            examples = pd.concat([hard_negatives, hard_positives], ignore_index=True)
            examples.to_csv(WORKING_DIR / "qwen3_reranker_hard_examples.csv", index=False)
            display(examples[["slice", "category", "target", "predict", "name1", "name2"]])

            completion = {
                "status": "complete",
                "pairs": len(predictions),
                "macro_average_precision": macro_ap,
                "overall_average_precision": overall_ap,
            }
            (WORKING_DIR / "notebook_completed.json").write_text(
                json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps(completion, ensure_ascii=False, indent=2))
            """
        ),
        markdown(
            """
            ## Интерпретация

            Это zero-shot проверка retrieval-reranker на задаче product identity,
            поэтому качество не является baseline обученного решения. Главные
            результаты notebook — работоспособность `vLLM.score`, фактическая
            скорость на 2×T4, доля truncation и распределение scores. Следующий
            честный эксперимент — fine-tuning на component-disjoint train folds и
            повторный inference на зафиксированном holdout.
            """
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "product_matching_sample": {
                "rows": len(sample),
                "random_state": RANDOM_STATE,
                "positive_rate": float(sample["target"].mean()),
            },
        }
    )
    return notebook


def main() -> None:
    sample = load_sample()
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(SAMPLE_PATH, index=False)
    notebook = build_notebook(sample)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT_PATH)
    print(
        f"Wrote {OUTPUT_PATH.relative_to(ROOT)}: {len(sample):,} pairs, "
        f"positive_rate={sample.target.mean():.2%}, "
        f"notebook={OUTPUT_PATH.stat().st_size / 2**20:.2f} MiB\n"
        f"Wrote private dataset payload {SAMPLE_PATH.relative_to(ROOT)}: "
        f"{SAMPLE_PATH.stat().st_size / 2**20:.2f} MiB"
    )


if __name__ == "__main__":
    main()
