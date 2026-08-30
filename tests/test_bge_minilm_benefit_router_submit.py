from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_PATH = ROOT / "submits/bge-minilm-benefit-router-40-st-v1/run.py"


def load_runtime():
    spec = importlib.util.spec_from_file_location("benefit_router_submit_run", RUN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_route_top_budget_is_exact_and_deterministic(monkeypatch):
    monkeypatch.syspath_prepend(str(RUN_PATH.parent))
    runtime = load_runtime()
    pairs = pd.DataFrame({"id1": [5, 4, 3, 2, 1], "id2": [1, 1, 1, 1, 1]})
    priority = np.ones(5)
    routed = runtime.route_top_budget(priority, pairs, coverage=0.40)
    assert routed.sum() == 2
    assert routed.tolist() == [False, False, False, True, True]


def test_symmetric_score_contract(monkeypatch):
    monkeypatch.syspath_prepend(str(RUN_PATH.parent))
    runtime = load_runtime()
    logits = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
    probability = runtime._sigmoid(logits)
    assert np.allclose(probability, [0.11920292, 0.5, 0.880797], atol=1e-6)
