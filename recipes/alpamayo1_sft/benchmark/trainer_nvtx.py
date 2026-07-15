# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Patch Trainer with per-rank NVTX ranges for collate and training_step."""

from __future__ import annotations

from typing import Any

from alpamayo1_sft.benchmark.nvtx_markers import local_rank, nvtx_range


def _resolve_collate_fn(dataloader: Any):
    """Return (target, collate_fn) where target accepts collate_fn assignment."""
    base = getattr(dataloader, "base_dataloader", None)
    target = base if base is not None else dataloader
    return target, getattr(target, "collate_fn", None)


def _sync_collate_nvtx(dataloader: Any, rank: int) -> Any:
    """Inject NVTX on collate_fn and any active DataLoader iterator snapshots."""
    target, collate = _resolve_collate_fn(dataloader)
    if collate is None or getattr(collate, "_nvtx_dataloader_collate_wrapped", False):
        return dataloader

    def collate_with_nvtx(features):
        with nvtx_range(f"rank{rank}.collate_fn"):
            return collate(features)

    collate_with_nvtx._nvtx_dataloader_collate_wrapped = True  # type: ignore[attr-defined]
    target.collate_fn = collate_with_nvtx

    iterator = getattr(target, "_iterator", None)
    if iterator is not None:
        if hasattr(iterator, "_collate_fn"):
            iterator._collate_fn = collate_with_nvtx
        fetcher = getattr(iterator, "_dataset_fetcher", None)
        if fetcher is not None and hasattr(fetcher, "collate_fn"):
            fetcher.collate_fn = collate_with_nvtx
    return dataloader


def patch_trainer_nvtx(trainer: Any) -> None:
    """Wrap ``training_step``, ``get_batch_samples``, and dataloader ``collate_fn``."""
    if getattr(trainer, "_nvtx_training_step_patched", False):
        return

    rank = local_rank()
    original_training_step = trainer.training_step
    original_get_batch_samples = trainer.get_batch_samples
    original_get_train_dataloader = trainer.get_train_dataloader

    def training_step(model, inputs, num_items_in_batch=None):
        with nvtx_range(f"rank{rank}.training_step"):
            return original_training_step(model, inputs, num_items_in_batch=num_items_in_batch)

    def get_batch_samples(epoch_iterator, num_batches, device):
        _sync_collate_nvtx(original_get_train_dataloader(), rank)
        with nvtx_range(f"rank{rank}.get_batch_samples"):
            return original_get_batch_samples(epoch_iterator, num_batches, device)

    def get_train_dataloader():
        dataloader = original_get_train_dataloader()
        return _sync_collate_nvtx(dataloader, rank)

    trainer.training_step = training_step  # type: ignore[method-assign]
    trainer.get_batch_samples = get_batch_samples  # type: ignore[method-assign]
    trainer.get_train_dataloader = get_train_dataloader  # type: ignore[method-assign]
    trainer._nvtx_training_step_patched = True
