from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from item_pipeline.cli import provider_credentials, read_env_value
from item_pipeline.generate import run_generation
from item_pipeline.validation import validate_generated_dataset


class FakeQwenClient:
    model = "fake-qwen"

    def generate(self, prompt, *, category, attribute_keys, seed):
        values = {}
        for key in attribute_keys:
            if key == "тип":
                values[key] = "картридж"
            else:
                values[key] = f"новое-{seed}-{key}"
        return {
            "item": {
                "name": f"новый тестовый картридж модель-{seed}",
                "attributes": values,
                "category": category,
            },
            "request_attempts": 1,
            "latency_seconds": 0.01,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "response_id": f"response-{seed}",
        }


class FlakyQwenClient(FakeQwenClient):
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def generate(self, prompt, *, category, attribute_keys, seed):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("synthetic transient failure")
        return super().generate(
            prompt,
            category=category,
            attribute_keys=attribute_keys,
            seed=seed,
        )


def build_index(path: Path) -> None:
    records = []
    for index in range(12):
        attributes = {
            "тип": "картридж",
            "бренд": f"brand-{index}",
            "модель": f"model-{index}",
            "цвет товара": "черный",
        }
        name = f"brand-{index} картридж model-{index} черный"
        records.append(
            {
                "id": index + 1,
                "name": name,
                "attributes": json.dumps(attributes, ensure_ascii=False),
                "category": "Электроника",
                "subtype": "картридж",
                "retrieval_text": name,
                "sample_hash": index,
            }
        )
    path.mkdir(parents=True)
    pd.DataFrame(records).to_parquet(path / "exemplar_bank.parquet", index=False)
    (path / "profile.json").write_text(
        json.dumps({"version": "test", "bank_rows": len(records)}), encoding="utf-8"
    )


class PipelineTest(unittest.TestCase):
    def test_provider_credentials_reads_env_file_without_exposing_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "TEST_PROVIDER_SECRET='secret-value'\nMODEL=z-ai/glm-5.3-flash\n",
                encoding="utf-8",
            )
            self.assertEqual(
                read_env_value(env_file, "TEST_PROVIDER_SECRET"), "secret-value"
            )
            args = type("Args", (), {
                "api_key_env": "TEST_PROVIDER_SECRET",
                "env_file": env_file,
                "model": None,
            })()
            self.assertEqual(
                provider_credentials(args),
                ("secret-value", "z-ai/glm-5.3-flash"),
            )

    def test_failed_task_is_requeued_with_a_new_seed_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_dir = root / "index"
            output_dir = root / "generated"
            build_index(index_dir)
            client = FlakyQwenClient(failures=2)
            summary = run_generation(
                index_dir=index_dir,
                output_dir=output_dir,
                client=client,
                system_prompt="test prompt",
                count=1,
                seed=17,
                id_start=-1,
                example_count=3,
                categories=None,
                workers=1,
                generation_attempts=1,
                checkpoint_every=1,
                task_retries=2,
            )
            self.assertEqual(summary["generated"], 1)
            self.assertEqual(summary["task_retry_events"], 2)
            self.assertEqual(client.calls, 3)
            metadata = pd.read_parquet(output_dir / "generation_metadata.parquet")
            self.assertEqual(int(metadata.iloc[0]["task_retry_round"]), 2)

    def test_end_to_end_generation_checkpoint_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_dir = root / "index"
            output_dir = root / "generated"
            build_index(index_dir)
            arguments = dict(
                index_dir=index_dir,
                output_dir=output_dir,
                client=FakeQwenClient(),
                system_prompt="test prompt",
                count=3,
                seed=17,
                id_start=-1,
                example_count=3,
                categories=None,
                workers=2,
                generation_attempts=2,
                checkpoint_every=1,
            )
            summary = run_generation(**arguments)
            self.assertEqual(summary["generated"], 3)
            items = pd.read_parquet(output_dir / "items.parquet")
            self.assertEqual(
                items.columns.tolist(), ["id", "name", "attributes", "category"]
            )
            self.assertEqual(items["id"].tolist(), [-1, -2, -3])

            # The exact same run must resume without new calls or duplicate rows.
            resumed = run_generation(**arguments)
            self.assertEqual(resumed["generated"], 3)
            report = validate_generated_dataset(
                output_dir / "items.parquet",
                reference_path=index_dir / "exemplar_bank.parquet",
                metadata_path=output_dir / "generation_metadata.parquet",
            )
            self.assertTrue(report["valid"])
            self.assertEqual(report["new_attribute_keys"], [])


if __name__ == "__main__":
    unittest.main()
