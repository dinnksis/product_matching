#!/usr/bin/env python3
"""Assemble the trained embedding/CatBoost model into a submission ZIP."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"artifacts/kaggle/product-matching-qwen-embedding-boosting/embedding_boosting"
SUBMIT=ROOT/"submits/qwen-embedding-catboost"
ARCHIVE=ROOT/"submits/qwen-embedding-catboost.zip"


def sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8*1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def copy_training_artifacts():
    model=SOURCE/"03_names_qwen_attributes/model.cbm"; keys=SOURCE/"selected_attribute_keys.json"
    if not model.is_file() or not keys.is_file(): raise FileNotFoundError("Kaggle artifacts are incomplete")
    (SUBMIT/"models").mkdir(parents=True,exist_ok=True)
    shutil.copy2(model,SUBMIT/"models/matching_model.cbm"); shutil.copy2(keys,SUBMIT/"selected_attribute_keys.json")


def verify():
    json.loads((SUBMIT/"metadata.json").read_text(encoding="utf-8"))


def archive():
    if ARCHIVE.exists(): ARCHIVE.unlink()
    with zipfile.ZipFile(ARCHIVE,"w",allowZip64=True) as output:
        for path in sorted(SUBMIT.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts: continue
            compression=zipfile.ZIP_STORED if path.suffix in {".safetensors",".cbm"} else zipfile.ZIP_DEFLATED
            output.write(path,path.relative_to(SUBMIT).as_posix(),compress_type=compression)
    with zipfile.ZipFile(ARCHIVE) as source:
        if source.testzip(): raise RuntimeError("broken archive")
        if not {"metadata.json","run.py"}.issubset(source.namelist()): raise RuntimeError("root files missing")
    print(f"Created {ARCHIVE} ({ARCHIVE.stat().st_size:,} bytes)\nSHA-256: {sha256(ARCHIVE)}")


def main():
    parser=argparse.ArgumentParser(); parser.parse_args()
    copy_training_artifacts()
    verify(); archive()


if __name__=="__main__": main()
