from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import push_bge_pretrain_checkpoint_dataset as uploader


class PushBgePretrainCheckpointDatasetTest(unittest.TestCase):
    @staticmethod
    def _source_payloads() -> dict[str, bytes]:
        return {
            "config.json": b'{"model_type":"xlm-roberta"}\n',
            "tokenizer.json": b'{"version":"1.0"}\n',
            "tokenizer_config.json": b'{"tokenizer_class":"XLMRobertaTokenizer"}\n',
            "special_tokens_map.json": b'{"pad_token":"<pad>"}\n',
            "model.safetensors": b"0123456789abcdefghijklmnop",
        }

    @classmethod
    def _write_source(cls, directory: Path) -> dict[str, dict[str, object]]:
        payloads = cls._source_payloads()
        expected: dict[str, dict[str, object]] = {}
        directory.mkdir(parents=True, exist_ok=True)
        for filename, payload in payloads.items():
            (directory / filename).write_bytes(payload)
            expected[filename] = {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        return expected

    def test_frozen_production_contract(self) -> None:
        self.assertEqual(uploader.DATASET_SLUG, "product-matching-bge-pretrain-2ep")
        self.assertEqual(uploader.MODEL_PART_BYTES, 64 * 1024 * 1024)
        self.assertEqual(
            uploader.EXPECTED_SOURCE_FILES,
            {
                "config.json": {
                    "bytes": 764,
                    "sha256": "3f143138299caf72270c79814d1f0e3c38fc168e2d30f1e6cc4c70f6cdb481f5",
                },
                "tokenizer.json": {
                    "bytes": 17_082_900,
                    "sha256": "8bf8afbfd11306bd872018c53bfdf2e160a56f8edbcf49933324404791c148d3",
                },
                "tokenizer_config.json": {
                    "bytes": 1_203,
                    "sha256": "b87c8703482b0300d3da30e201519aa641f6a450f5eb5bf1e624afbf70c74d80",
                },
                "special_tokens_map.json": {
                    "bytes": 964,
                    "sha256": "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835",
                },
                "model.safetensors": {
                    "bytes": 2_271_071_852,
                    "sha256": "c21ccfcd5de310ca0328620bf8ba09e838dbe3f6394be656bd7fec16ad8377d1",
                },
            },
        )

    def test_build_payload_records_exact_bytes_and_unverified_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            stage = root / "stage"
            expected = self._write_source(source)

            manifest = uploader.build_payload(
                source,
                stage,
                "testowner",
                expected_files=expected,
                model_part_bytes=8,
            )

            self.assertEqual(
                manifest["dataset"],
                "testowner/product-matching-bge-pretrain-2ep",
            )
            self.assertIs(manifest["is_private"], True)
            self.assertEqual(manifest["checkpoint_files"], expected)
            self.assertEqual(manifest["source"], "external_user_supplied_checkpoint")
            provenance = manifest["provenance"]
            self.assertEqual(provenance["status"], "user_supplied_unverified")
            self.assertIs(provenance["training_lineage_verified"], False)
            self.assertIs(provenance["pretraining_data_verified"], False)
            self.assertIs(provenance["base_model_revision_verified"], False)

            reconstruction = manifest["reconstruction"]
            model_payload = self._source_payloads()["model.safetensors"]
            self.assertEqual(reconstruction["bytes"], len(model_payload))
            self.assertEqual(
                reconstruction["sha256"],
                hashlib.sha256(model_payload).hexdigest(),
            )
            self.assertEqual(reconstruction["part_bytes"], 8)
            self.assertEqual(
                reconstruction["parts"],
                [
                    "model.safetensors.part000",
                    "model.safetensors.part001",
                    "model.safetensors.part002",
                    "model.safetensors.part003",
                ],
            )
            reconstructed = b"".join(
                (stage / filename).read_bytes()
                for filename in reconstruction["parts"]
            )
            self.assertEqual(reconstructed, model_payload)
            self.assertFalse((stage / "model.safetensors").exists())

            metadata = json.loads(
                (stage / "dataset-metadata.json").read_text(encoding="utf-8")
            )
            self.assertIs(metadata["isPrivate"], True)
            self.assertEqual(metadata["licenses"], [{"name": "unknown"}])
            self.assertIn("remain unverified", metadata["description"])

            staged = {
                path.name
                for path in stage.iterdir()
                if path.name != "dataset-metadata.json"
            }
            self.assertEqual(staged, set(manifest["files"]) | {uploader.MANIFEST_NAME})
            for filename in uploader.DIRECT_CHECKPOINT_FILES:
                source_stat = os.stat(source / filename)
                staged_stat = os.stat(stage / filename)
                self.assertNotEqual(
                    (source_stat.st_dev, source_stat.st_ino),
                    (staged_stat.st_dev, staged_stat.st_ino),
                )
            staged_config = (stage / "config.json").read_bytes()
            changed_source_config = bytes(
                [self._source_payloads()["config.json"][0] ^ 1]
            ) + self._source_payloads()["config.json"][1:]
            (source / "config.json").write_bytes(changed_source_config)
            self.assertEqual((stage / "config.json").read_bytes(), staged_config)

    def test_source_stage_overlap_and_symlink_alias_fail_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            expected = self._write_source(source)
            original_model = (source / "model.safetensors").read_bytes()

            with self.assertRaises(SystemExit):
                uploader.build_payload(
                    source,
                    source,
                    "testowner",
                    expected_files=expected,
                    model_part_bytes=8,
                )
            self.assertEqual((source / "model.safetensors").read_bytes(), original_model)

            descendant = source / "nested-stage"
            with self.assertRaises(SystemExit):
                uploader.build_payload(
                    source,
                    descendant,
                    "testowner",
                    expected_files=expected,
                    model_part_bytes=8,
                )
            self.assertFalse(descendant.exists())
            self.assertEqual((source / "model.safetensors").read_bytes(), original_model)

            with self.assertRaises(SystemExit):
                uploader.build_payload(
                    source,
                    root,
                    "testowner",
                    expected_files=expected,
                    model_part_bytes=8,
                )
            self.assertEqual((source / "model.safetensors").read_bytes(), original_model)

            alias = root / "stage-alias"
            alias.symlink_to(source, target_is_directory=True)
            with self.assertRaises(SystemExit):
                uploader.build_payload(
                    source,
                    alias,
                    "testowner",
                    expected_files=expected,
                    model_part_bytes=8,
                )
            self.assertEqual((source / "model.safetensors").read_bytes(), original_model)

    def test_same_size_source_mutation_cannot_publish_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            stage = root / "stage"
            expected = self._write_source(source)
            real_validate = uploader.validate_source_checkpoint

            def validate_then_mutate(*args: object, **kwargs: object) -> object:
                ledger = real_validate(*args, **kwargs)
                path = source / "config.json"
                original = path.read_bytes()
                path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
                self.assertEqual(path.stat().st_size, expected["config.json"]["bytes"])
                return ledger

            with (
                mock.patch.object(
                    uploader,
                    "validate_source_checkpoint",
                    side_effect=validate_then_mutate,
                ),
                self.assertRaises(SystemExit),
            ):
                uploader.build_payload(
                    source,
                    stage,
                    "testowner",
                    expected_files=expected,
                    model_part_bytes=8,
                )
            self.assertFalse((stage / uploader.MANIFEST_NAME).exists())

    def test_same_size_staged_mutation_is_caught_before_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            stage = root / "stage"
            expected = self._write_source(source)
            real_split = uploader.split_model

            def split_then_mutate(*args: object, **kwargs: object) -> object:
                result = real_split(*args, **kwargs)
                path = stage / "config.json"
                original = path.read_bytes()
                path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
                return result

            with (
                mock.patch.object(
                    uploader,
                    "split_model",
                    side_effect=split_then_mutate,
                ),
                self.assertRaises(SystemExit),
            ):
                uploader.build_payload(
                    source,
                    stage,
                    "testowner",
                    expected_files=expected,
                    model_part_bytes=8,
                )
            self.assertFalse((stage / uploader.MANIFEST_NAME).exists())

    def test_preupload_rehash_rejects_same_size_staged_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            stage = root / "stage"
            expected = self._write_source(source)
            manifest = uploader.build_payload(
                source,
                stage,
                "testowner",
                expected_files=expected,
                model_part_bytes=8,
            )
            part = stage / manifest["reconstruction"]["parts"][0]
            original = part.read_bytes()
            part.write_bytes(bytes([original[0] ^ 1]) + original[1:])
            self.assertEqual(part.stat().st_size, len(original))
            with self.assertRaises(SystemExit):
                uploader.verify_payload_for_upload(
                    stage,
                    manifest,
                    expected_checkpoint_files=expected,
                )

    def test_payload_rejects_hash_drift_and_unexpected_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            expected = self._write_source(source)
            expected["config.json"] = {
                **expected["config.json"],
                "sha256": "0" * 64,
            }
            with self.assertRaises(SystemExit):
                uploader.build_payload(
                    source,
                    root / "hash-stage",
                    "testowner",
                    expected_files=expected,
                    model_part_bytes=8,
                )

            expected = self._write_source(source)
            (source / "training_state.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                uploader.build_payload(
                    source,
                    root / "extra-stage",
                    "testowner",
                    expected_files=expected,
                    model_part_bytes=8,
                )

    def test_payload_rejects_stale_staging_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            stage = root / "stage"
            expected = self._write_source(source)
            stage.mkdir()
            (stage / "stale.bin").write_bytes(b"stale")
            with self.assertRaises(SystemExit):
                uploader.build_payload(
                    source,
                    stage,
                    "testowner",
                    expected_files=expected,
                    model_part_bytes=8,
                )

    def test_main_dry_run_never_resolves_or_calls_kaggle_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            stage = root / "stage"
            env_file = root / ".env"
            env_file.write_text("KAGGLE_USERNAME=testowner\n", encoding="utf-8")
            expected = self._write_source(source)
            args = argparse.Namespace(
                env_file=env_file,
                source_dir=source,
                stage_dir=stage,
                dataset_slug=uploader.DATASET_SLUG,
                message="test",
                dry_run=True,
            )
            with (
                mock.patch.object(uploader, "parse_args", return_value=args),
                mock.patch.object(uploader, "EXPECTED_SOURCE_FILES", expected),
                mock.patch.object(uploader, "MODEL_PART_BYTES", 8),
                mock.patch.dict(
                    os.environ,
                    {"KAGGLE_USERNAME": "testowner"},
                    clear=False,
                ),
                mock.patch.object(
                    uploader.kaggle,
                    "kaggle_command",
                    side_effect=AssertionError("dry-run contacted Kaggle CLI"),
                ),
                mock.patch.object(
                    uploader.subprocess,
                    "run",
                    side_effect=AssertionError("dry-run spawned upload"),
                ),
            ):
                uploader.main()


if __name__ == "__main__":
    unittest.main()
