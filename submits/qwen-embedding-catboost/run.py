#!/usr/bin/env python3
"""Deadline-aware offline Qwen item embeddings + structured CatBoost submission."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("OMP_NUM_THREADS", "20")

import numpy as np
import pandas as pd
from rapidfuzz import fuzz


ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("PM_MODEL_DIR", "/opt/models/qwen3-embedding-0.6b"))
if not MODEL_DIR.is_dir():
    MODEL_DIR = ROOT / "models" / "qwen_embedding_model"
CATBOOST_PATH = ROOT / "models" / "matching_model.cbm"
KEYS_PATH = ROOT / "selected_attribute_keys.json"
MAX_LENGTH = int(os.getenv("PM_MAX_LENGTH", "96"))
EMBEDDING_DIMENSION = 256
BATCH_SIZE = int(os.getenv("PM_BATCH_SIZE", "1024"))
SPACE_RE = re.compile(r"\s+")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
FAMILY_PATTERNS = {
    "brand": re.compile(r"бренд|brand|производител"),
    "model": re.compile(r"модель|model|серия|линейка"),
    "identifier": re.compile(r"артикул|партномер|part.?number|sku|mpn|oem|код товара"),
    "size": re.compile(r"размер|длина|ширина|высота|диаметр|толщина|габарит"),
    "quantity": re.compile(r"количеств|комплект|упаков|штук|шт\.?$|объ.м|вес"),
    "color": re.compile(r"цвет|оттенок"),
    "material": re.compile(r"материал|состав|сырь"),
    "country": re.compile(r"страна|производств"),
    "seller_noise": re.compile(r"продав|магазин|поставщик|валюта|цена|достав|гарант"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items_path", "--items-path", "-i", required=True, type=Path)
    parser.add_argument("--matches_path", "--matches-path", "-m", required=True, type=Path)
    parser.add_argument("--output_path", "--output-path", "-o", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def log(message, started): print(f"[{time.perf_counter()-started:7.1f}s] {message}", flush=True)


def clean(value):
    if value is None: return ""
    if isinstance(value, (dict, list)): value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return SPACE_RE.sub(" ", str(value)).strip().casefold().replace("ё", "е")


def parse_attributes(raw):
    if not isinstance(raw, str) or not raw: return {}
    value = json.loads(raw)
    if not isinstance(value, dict): return {}
    result = {}
    for key, item in value.items():
        key, item = clean(key), clean(item)
        if key and item: result[key] = item
    return result


def family_for_key(key):
    for family, pattern in FAMILY_PATTERNS.items():
        if pattern.search(key): return family
    return None


def load_data(args):
    matches = pd.read_parquet(args.matches_path, columns=["id1", "id2"])
    if args.limit: matches = matches.head(args.limit).copy()
    required = pd.unique(matches[["id1", "id2"]].to_numpy().reshape(-1))
    items = pd.read_parquet(args.items_path, columns=["id", "name", "attributes", "category"])
    items = items[items.id.isin(required)].reset_index(drop=True)
    if len(items) != len(required): raise ValueError("matches reference missing item IDs")
    lookup = pd.Series(np.arange(len(items), dtype=np.int32), index=items.id.to_numpy())
    left = lookup.loc[matches.id1].to_numpy(); right = lookup.loc[matches.id2].to_numpy()
    categories = items.category.astype(str).to_numpy()[left]
    return matches, items, left, right, categories


def name_features(names, left, right, categories):
    rows = []
    for lp, rp in zip(left, right):
        first, second = clean(names[lp]), clean(names[rp])
        fn, sn = set(NUMBER_RE.findall(first)), set(NUMBER_RE.findall(second)); union = fn | sn
        longest = max(len(first), len(second))
        rows.append((fuzz.ratio(first,second)/100, fuzz.token_set_ratio(first,second)/100,
                     fuzz.token_sort_ratio(first,second)/100, float(first==second),
                     min(len(first),len(second))/longest if longest else 1.0,
                     len(fn&sn)/max(1,len(union)), float(bool(fn) and bool(sn)), abs(len(first)-len(second))))
    frame = pd.DataFrame(rows, columns=["name_ratio","name_token_set_ratio","name_token_sort_ratio","name_exact","name_length_ratio","name_numeric_jaccard","name_numbers_both","name_length_delta"], dtype=np.float32)
    return pd.concat([frame, pd.get_dummies(pd.Series(categories,name="category"),prefix="category",dtype=np.float32)], axis=1)


def attribute_features(categories, left, right, attrs, selected, per_category=5):
    rows=[]; families=list(FAMILY_PATTERNS)
    for category,lp,rp in zip(categories,left,right):
        first,second=attrs[lp],attrs[rp]; fk,sk=set(first),set(second); common=fk&sk; union=fk|sk
        equal=sum(first[k]==second[k] for k in common); conflict=len(common)-equal
        ff,sf=defaultdict(set),defaultdict(set)
        for k,v in first.items(): ff[family_for_key(k)].add(v)
        for k,v in second.items(): sf[family_for_key(k)].add(v)
        row=[len(common),len(common)/max(1,len(union)),equal,conflict,equal/max(1,len(common)),conflict/max(1,len(common)),abs(len(first)-len(second)),float(first==second)]
        for family in families:
            a,b=ff[family],sf[family]; row += [float(bool(a and b)),float(bool(a&b)),float(bool(a and b and not(a&b)))]
        keys=selected.get(str(category),[])
        for rank in range(per_category):
            key=keys[rank] if rank<len(keys) else None; both=bool(key and key in first and key in second)
            row += [float(both),float(both and first[key]==second[key]),float(both and first[key]!=second[key]),float(bool(key) and ((key in first)!=(key in second)))]
        rows.append(row)
    columns=["attr_shared_keys","attr_key_jaccard","attr_equal_values","attr_conflicting_values","attr_equal_ratio","attr_conflict_ratio","attr_count_delta","attr_exact"]
    for family in families: columns += [f"{family}_both",f"{family}_match",f"{family}_conflict"]
    for rank in range(per_category): columns += [f"exact_key_{rank}_both",f"exact_key_{rank}_match",f"exact_key_{rank}_conflict",f"exact_key_{rank}_one_missing"]
    return pd.DataFrame(rows,columns=columns,dtype=np.float32)


def encode_names(names, started):
    import torch
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen embedding inference requires CUDA")
    model=SentenceTransformer(
        str(MODEL_DIR),
        model_kwargs={"torch_dtype": torch.float16},
        local_files_only=True,
    )
    model.max_seq_length=MAX_LENGTH
    log(f"Loaded SentenceTransformer; batch={BATCH_SIZE}, max_length={MAX_LENGTH}",started)
    embeddings=model.encode(
        names,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
        device="cuda:0",
        truncate_dim=EMBEDDING_DIMENSION,
    ).astype(np.float32,copy=False)
    norms=np.linalg.norm(embeddings,axis=1,keepdims=True)
    embeddings/=np.maximum(norms,1e-12)
    log(f"Embedded all {len(names):,} items",started)
    return embeddings.astype(np.float16)


def embedding_features(embeddings,left,right):
    first=np.asarray(embeddings[left],np.float32); second=np.asarray(embeddings[right],np.float32)
    absolute=np.abs(first-second); product=first*second
    data={"embedding_cosine":np.einsum("ij,ij->i",first,second),"embedding_l1_mean":absolute.mean(1),"embedding_l2":np.sqrt(np.square(first-second).sum(1)),"embedding_abs_max":absolute.max(1)}
    for i in range(EMBEDDING_DIMENSION): data[f"embedding_abs_{i:03d}"]=absolute[:,i]; data[f"embedding_product_{i:03d}"]=product[:,i]
    return pd.DataFrame(data,dtype=np.float32)


def align(frame, feature_names):
    missing=set(feature_names)-set(frame.columns)
    for column in missing: frame[column]=np.float32(0)
    return frame.loc[:,feature_names]


def main():
    args=parse_args(); started=time.perf_counter(); matches,items,left,right,categories=load_data(args)
    cache_root=args.output_path.parent/".cache"
    cache_root.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault("HF_HOME",str(cache_root/"huggingface"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME",str(cache_root/"sentence_transformers"))
    log(f"Loaded {len(matches):,} pairs and {len(items):,} required items",started)
    names=items.name.fillna("").astype(str).to_numpy(); lexical=name_features(names,left,right,categories)
    attrs=[parse_attributes(raw) for raw in items.attributes]; selected=json.loads(KEYS_PATH.read_text(encoding="utf-8")); structured=attribute_features(categories,left,right,attrs,selected)
    embeddings=encode_names(names.tolist(),started)
    from catboost import CatBoostClassifier
    model=CatBoostClassifier(); model.load_model(CATBOOST_PATH)
    features=pd.concat([lexical,embedding_features(embeddings,left,right),structured],axis=1)
    scores=model.predict_proba(align(features,model.feature_names_))[:,1]
    log("Applied Qwen + attributes CatBoost to every pair",started)
    if len(scores)!=len(matches) or not np.isfinite(scores).all(): raise RuntimeError("invalid scores")
    output=matches[["id1","id2"]].copy(); output["predict"]=scores
    args.output_path.parent.mkdir(parents=True,exist_ok=True); output.to_csv(args.output_path,index=False)
    log(f"Saved {len(output):,} rows",started); return 0


if __name__=="__main__": raise SystemExit(main())
