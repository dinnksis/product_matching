#!/usr/bin/env python3
"""Package the 60% MiniLM / 5% RuModern compact one-way submission."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import build_bge_minilm_rumodern_fast_oneway_submit as base


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "submits/bge-minilm-rumodern-fast-oneway-60-5-st-v2"
ARCHIVE = ROOT / "submits/bge-minilm-rumodern-fast-oneway-60-5-st-v2.zip"


def main() -> int:
    base.SUBMIT = SUBMIT
    base.ARCHIVE = ARCHIVE
    base.EXPECTED_IMAGE = "dinakepech/ecup26-bge-minilm-rumodern-router60-sdpa:1.0"
    base.BGE_MODEL = SUBMIT / "models/bge-reranker-v2-m3-human-ft-v1"
    base.MINILM_MODEL = SUBMIT / "models/minilm-5ep-human-ft-v1"
    base.RU_MODEL = SUBMIT / "models/rumodernbert-base-human-ft-v1"
    base.stage()
    shutil.copy2(
        ROOT / "submits/bge-minilm-rumodern-fast-oneway-40-5-st-v2/run.py",
        SUBMIT / "run.py",
    )
    base.write_manifest()
    manifest_path = SUBMIT / "SUBMISSION_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["experiment"] = "bge_minilm_rumodern_fast_oneway_60_5_st_v2"
    manifest["routing"]["mini_coverage"] = 0.60
    manifest["offline_validation"] = {
        "note": "one-way compact 60/5 was not selected by validation; use as runtime/quality probe"
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    base.archive()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
