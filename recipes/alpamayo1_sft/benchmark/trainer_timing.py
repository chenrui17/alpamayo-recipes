# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Patch Trainer to record fetch / collate / GPU phase timings."""

from __future__ import annotations

import time
from typing import Any

from alpamayo1_sft.benchmark.dataloader_timing import apply_batch_timing, record, set_timing_step


_active_step = 0
_step_wall_t0: float | None = None


def active_step() -> int:
    return _active_step


def consume_step_wall_seconds() -> float | None:
    global _step_wall_t0
    if _step_wall_t0 is None:
        return None
    elapsed = time.perf_counter() - _step_wall_t0
    _step_wall_t0 = None
    return elapsed


def _resolve_collate_fn(dataloader: Any):
    base = getattr(dataloader, "base_dataloader", None)
    target = base if base is not None else dataloader
    return target, getattr(target, "collate_fn", None)


def _sync_collate_timing(dataloader: Any, collate: Any) -> Any:
    if collate is None or getattr(collate, "_timing_dataloader_collate_wrapped", False):
        return dataloader

    def collate_with_timing(features):
        return collate(features)

    collate_with_timing._timing_dataloader_collate_wrapped = True  # type: ignore[attr-defined]
    target, _ = _resolve_collate_fn(dataloader)
    target.collate_fn = collate_with_timing

    iterator = getattr(target, "_iterator", None)
    if iterator is not None:
        if hasattr(iterator, "_collate_fn"):
            iterator._collate_fn = collate_with_timing
        fetcher = getattr(iterator, "_dataset_fetcher", None)
        if fetcher is not None and hasattr(fetcher, "collate_fn"):
            fetcher.collate_fn = collate_with_timing
    return dataloader


def patch_trainer_timing(trainer: Any) -> None:
    if getattr(trainer, "_timing_patched", False):
        return

    original_get_batch_samples = trainer.get_batch_samples
    original_training_step = trainer.training_step
    original_get_train_dataloader = trainer.get_train_dataloader
    original_collate = trainer.data_collator

    def timed_collate(features):
        return original_collate(features)

    timed_collate._timing_dataloader_collate_wrapped = True  # type: ignore[attr-defined]
    trainer.data_collator = timed_collate

    def get_train_dataloader():
        dataloader = original_get_train_dataloader()
        return _sync_collate_timing(dataloader, timed_collate)

    def get_batch_samples(epoch_iterator, num_batches, device):
        global _active_step, _step_wall_t0
        _step_wall_t0 = time.perf_counter()
        _active_step = trainer.state.global_step + 1
        set_timing_step(_active_step)
        _sync_collate_timing(original_get_train_dataloader(), timed_collate)

        batch_samples = []
        for _ in range(num_batches):
            t0 = time.perf_counter()
            try:
                batch_samples.append(next(epoch_iterator))
            except StopIteration:
                break
            record("fetch.next", time.perf_counter() - t0)

        num_items_in_batch = trainer._get_num_items_in_batch(batch_samples, device)
        return batch_samples, num_items_in_batch

    def training_step(model, inputs, num_items_in_batch=None):
        set_timing_step(_active_step)
        apply_batch_timing(inputs)
        t0 = time.perf_counter()
        loss = original_training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        record("train.gpu", time.perf_counter() - t0)
        return loss

    trainer.get_batch_samples = get_batch_samples  # type: ignore[method-assign]
    trainer.training_step = training_step  # type: ignore[method-assign]
    trainer.get_train_dataloader = get_train_dataloader  # type: ignore[method-assign]
    trainer._timing_patched = True
