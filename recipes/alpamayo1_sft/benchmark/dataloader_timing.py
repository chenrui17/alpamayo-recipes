# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wall-clock timing for the training data pipeline (dataset → collate → GPU)."""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, get_worker_info

_LOCK = threading.Lock()
# step_index -> phase -> list[seconds]
_STEP_BUCKETS: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
_CURRENT_STEP = 0


def local_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def set_timing_step(step: int) -> None:
    global _CURRENT_STEP
    _CURRENT_STEP = step


def record(phase: str, seconds: float, step: int | None = None) -> None:
    idx = _CURRENT_STEP if step is None else step
    with _LOCK:
        _STEP_BUCKETS[idx][phase].append(seconds)


def apply_batch_timing(inputs: dict[str, Any]) -> None:
    """Record worker-side timings attached to a batch on the training rank."""
    timing = inputs.pop("_batch_timing", None)
    if not timing:
        return
    for phase, value in timing.items():
        if isinstance(value, list):
            for sec in value:
                record(phase, float(sec))
        else:
            record(phase, float(value))


def collect_sample_timings(samples: list[dict[str, Any]]) -> dict[str, list[float]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        timing = sample.pop("_dl_timing", None)
        if not timing:
            continue
        for phase, sec in timing.items():
            buckets[f"dataset.{phase}"].append(float(sec))
    return dict(buckets)


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "total_ms": 0.0, "avg_ms": 0.0, "max_ms": 0.0}
    return {
        "n": len(values),
        "total_ms": 1000 * sum(values),
        "avg_ms": 1000 * sum(values) / len(values),
        "max_ms": 1000 * max(values),
    }


def summarize_steps(step_ids: list[int] | None = None) -> dict[str, Any]:
    with _LOCK:
        keys = step_ids if step_ids is not None else sorted(_STEP_BUCKETS.keys())
        out: dict[str, Any] = {}
        for step in keys:
            if step not in _STEP_BUCKETS:
                continue
            phases = {k: _stats(v) for k, v in sorted(_STEP_BUCKETS[step].items())}
            fetch = phases.get("fetch.next", {}).get("total_ms", 0.0)
            collate = phases.get("collate.total", {}).get("total_ms", 0.0)
            ds_load = phases.get("dataset.load", {}).get("total_ms", 0.0)
            ds_pre = phases.get("dataset.preprocess", {}).get("total_ms", 0.0)
            gpu = phases.get("train.gpu", {}).get("total_ms", 0.0)
            step_wall = phases.get("step.wall", {}).get("total_ms", 0.0)
            worker_wait = max(fetch - collate - ds_load - ds_pre, 0.0)
            other = max(step_wall - fetch - gpu, 0.0)
            # Serial Trainer: GPU idle at step start ≈ fetch; unoverlap ≈ max(0, fetch+other - 0)
            gpu_idle_serial_est = fetch + other
            cpu_pipeline = fetch + other
            out[str(step)] = {
                "phases": phases,
                "derived_ms": {
                    "fetch_total": fetch,
                    "collate_total": collate,
                    "dataset_load_total": ds_load,
                    "dataset_preprocess_total": ds_pre,
                    "worker_wait_est": worker_wait,
                    "gpu_total": gpu,
                    "step_wall": step_wall,
                    "other_est": other,
                    "gpu_idle_serial_est": gpu_idle_serial_est,
                    "cpu_pipeline_est": cpu_pipeline,
                },
            }
        return out


def write_report(path: str | Path, meta: dict[str, Any], step_ids: list[int] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "rank": local_rank(),
        "steps": summarize_steps(step_ids),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


_LOAD_PATCHED = False


def _patch_load_physical_aiavdataset() -> None:
    global _LOAD_PATCHED
    if _LOAD_PATCHED:
        return
    import alpamayo.data.pai as pai_mod
    import alpamayo_r1.load_physical_aiavdataset as load_mod

    original = load_mod.load_physical_aiavdataset

    def timed_load(*args: Any, **kwargs: Any):
        t0 = time.perf_counter()
        out = original(*args, **kwargs)
        out.setdefault("_dl_timing", {})["load"] = time.perf_counter() - t0
        return out

    load_mod.load_physical_aiavdataset = timed_load
    pai_mod.load_physical_aiavdataset = timed_load
    _LOAD_PATCHED = True


def wrap_dataset_for_timing(dataset: Dataset, log_every: int = 20) -> Dataset:
    """Patch I/O + VLA preprocess so worker subprocesses record per-sample timings."""
    _patch_load_physical_aiavdataset()

    preprocess = getattr(dataset, "vla_preprocess_func", None)
    if preprocess is not None and not getattr(preprocess, "_timing_wrapped", False):
        local_calls = {"n": 0}

        def timed_preprocess(*args: Any, **kwargs: Any):
            t0 = time.perf_counter()
            out = preprocess(*args, **kwargs)
            pre_sec = time.perf_counter() - t0
            sample = kwargs.get("data")
            if sample is None and args:
                sample = args[0]
            if isinstance(sample, dict):
                sample.setdefault("_dl_timing", {})["preprocess"] = pre_sec
            local_calls["n"] += 1
            if log_every > 0 and local_calls["n"] % log_every == 0:
                wi = get_worker_info()
                wid = wi.id if wi is not None else -1
                print(
                    f"[DL-TIMING] rank={local_rank()} worker={wid} "
                    f"preprocess={pre_sec * 1000:.1f}ms",
                    flush=True,
                )
            return out

        timed_preprocess._timing_wrapped = True  # type: ignore[attr-defined]
        dataset.vla_preprocess_func = timed_preprocess  # type: ignore[attr-defined]

    return dataset


def ingest_sample_timings(samples: list[dict[str, Any]]) -> None:
    """Deprecated: use collect_sample_timings + attach_batch_timing instead."""
    for phase, values in collect_sample_timings(samples).items():
        for sec in values:
            record(phase, sec)
