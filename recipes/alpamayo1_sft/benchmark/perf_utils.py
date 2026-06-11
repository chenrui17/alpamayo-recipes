# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for reading performance / benchmark optimization flags."""

from __future__ import annotations

from functools import partial
from typing import Any

import hydra.utils as hyu
from omegaconf import DictConfig, OmegaConf

from alpamayo1_sft.benchmark.collate_cache import collate_fn_from_model_config_cached
from alpamayo1_sft.benchmark.collate_nvtx import collate_fn_from_model_config_cached_nvtx
from alpamayo1_sft.benchmark.collate_timing import collate_fn_from_model_config_cached_timed


def perf_plain(cfg: DictConfig) -> dict[str, Any]:
    """Merge ``performance.*`` with ``benchmark.optimizations`` (latter wins)."""
    perf: dict[str, Any] = {}
    if cfg.get("performance") is not None:
        perf.update(OmegaConf.to_container(cfg.performance, resolve=True))  # type: ignore[arg-type]
    bench_opt = (cfg.get("benchmark") or {}).get("optimizations")
    if bench_opt is not None:
        perf.update(OmegaConf.to_container(bench_opt, resolve=True))  # type: ignore[arg-type]
    return perf


def _wrap_collate_total_timing(collate):
    import time

    from alpamayo1_sft.benchmark.dataloader_timing import collect_sample_timings

    def wrapped(batch):
        batch_timing: dict[str, Any] = dict(collect_sample_timings(batch))
        t0 = time.perf_counter()
        out = collate(batch)
        batch_timing["collate.total"] = time.perf_counter() - t0
        out["_batch_timing"] = batch_timing
        return out

    return wrapped


def build_collate_fn(cfg: DictConfig, model, perf: dict[str, Any]):
    use_nvtx = bool(perf.get("nvtx_markers", False))
    use_timing = bool(perf.get("dataloader_timing", False))
    chat_template_version = "r1"
    collate_cfg = cfg.data.get("collate_fn") or {}
    if "chat_template_version" in collate_cfg:
        chat_template_version = collate_cfg.chat_template_version

    if perf.get("collate_cache", False):
        if use_nvtx:
            fn = collate_fn_from_model_config_cached_nvtx
        elif use_timing:
            fn = collate_fn_from_model_config_cached_timed
        else:
            fn = collate_fn_from_model_config_cached
        collate = partial(
            fn,
            model_config=model.config,
            chat_template_version=chat_template_version,
        )
    else:
        collate = hyu.instantiate(cfg.data.collate_fn, _convert_="partial", model_config=model.config)
        if use_timing:
            collate = _wrap_collate_total_timing(collate)

    # Outer collate_fn NVTX is injected on the live DataLoader in trainer_nvtx.py.
    return collate
