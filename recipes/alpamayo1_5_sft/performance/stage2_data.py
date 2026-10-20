# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in training data preparation for small, deterministic Stage-2 datasets.

Call :func:`apply_stage2_training_data` on the training dataset before creating
Trainer or DataLoader workers. Preloading freezes the result of one read per
unique dataset index, so it is appropriate only when those reads are
deterministic. Random augmentations or sampling need a different cache policy.
This caches CPU input data only; model activations and KV states are not cached.
"""

from __future__ import annotations

import copy
import numbers
import operator
import os
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any

from .stage2_optimizations import stage2_flag


DEFAULT_PRELOAD_MAX_BYTES = 4 * 1024**3


def _drop_raw(sample: Any) -> Any:
    if isinstance(sample, Mapping) and sample.get("tokenized_data") is not None:
        # The source dataset may retain or reuse its dictionary.
        sample = dict(sample)
        sample.pop("image_frames", None)
    return sample


def _copy_cpu_tree(value: Any, torch: Any, numpy: Any) -> Any:
    """Clone supported sample trees, including every mutable tensor/array leaf."""
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, Mapping):
        return {
            copy.deepcopy(key): _copy_cpu_tree(item, torch, numpy) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_copy_cpu_tree(item, torch, numpy) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_cpu_tree(item, torch, numpy) for item in value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        # Hydra keeps label_components as an OmegaConf ListConfig when using
        # _convert_='partial'. Normalize sequence metadata to an independent list.
        return [_copy_cpu_tree(item, torch, numpy) for item in value]
    if isinstance(value, numpy.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Stage-2 preload does not support numpy object arrays")
        return value.copy()
    if value is None or isinstance(value, (str, bytes, numbers.Number, numpy.generic)):
        return copy.deepcopy(value)
    raise TypeError(f"Unsupported Stage-2 preload sample value: {type(value).__name__}")


def _tree_bytes(value: Any, torch: Any, numpy: Any) -> tuple[int, int]:
    """Conservative object/payload estimate and the exact cloned tensor bytes.

    Count each occurrence separately because the cache clones each occurrence.
    Python allocator overhead is outside this estimate. Array payloads may be
    counted twice by getsizeof; that conservatism keeps the configured bound safe.
    """
    estimated = sys.getsizeof(value)
    tensor_bytes = 0
    if torch.is_tensor(value):
        tensor_bytes = value.numel() * value.element_size()
        estimated += tensor_bytes
    elif isinstance(value, Mapping):
        for key, item in value.items():
            key_size, key_tensors = _tree_bytes(key, torch, numpy)
            item_size, item_tensors = _tree_bytes(item, torch, numpy)
            estimated += key_size + item_size
            tensor_bytes += key_tensors + item_tensors
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            item_size, item_tensors = _tree_bytes(item, torch, numpy)
            estimated += item_size
            tensor_bytes += item_tensors
    elif isinstance(value, numpy.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Stage-2 preload does not support numpy object arrays")
        estimated += value.nbytes
    elif value is not None and not isinstance(value, (str, bytes, numbers.Number, numpy.generic)):
        raise TypeError(f"Unsupported Stage-2 preload sample value: {type(value).__name__}")
    return estimated, tensor_bytes


class Stage2TrainingDataset:
    """A map-style torch Dataset protocol wrapper around the original indices.

    PyTorch accepts the ``__len__``/``__getitem__`` protocol without inheritance.
    Keeping torch imports lazy also leaves the disabled path dependency-free.
    Repetition changes only the index mapping and length, never the source data.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        repeat_factor: int = 1,
        preload: bool = False,
        drop_raw: bool = False,
        preload_max_bytes: int = DEFAULT_PRELOAD_MAX_BYTES,
    ) -> None:
        self.dataset = dataset
        self.repeat_factor = operator.index(repeat_factor)
        if self.repeat_factor < 1:
            raise ValueError("repeat_factor must be a positive integer")
        self.unique_samples = len(dataset)
        self.drop_raw = drop_raw
        self.preprocessing_seconds = 0.0
        self.cache_tensor_bytes = 0
        self.cache_estimated_bytes = 0
        self.preload_max_bytes = operator.index(preload_max_bytes)
        if self.preload_max_bytes < 1:
            raise ValueError("preload_max_bytes must be a positive integer")
        self._cache: list[Any] | None = None
        if preload:
            import numpy
            import torch

            if torch.utils.data.get_worker_info() is not None:
                raise RuntimeError("Stage-2 preload must run before DataLoader workers are created")
            started = time.perf_counter()
            cache = []
            for index in range(self.unique_samples):
                sample = dataset[index]
                if drop_raw:
                    sample = _drop_raw(sample)
                estimated, tensor_bytes = _tree_bytes(sample, torch, numpy)
                if self.cache_estimated_bytes + estimated > self.preload_max_bytes:
                    raise MemoryError(
                        f"Stage-2 preload at sample {index} exceeds "
                        f"preload_max_bytes={self.preload_max_bytes}; "
                        f"estimated_bytes={self.cache_estimated_bytes + estimated}"
                    )
                cache.append(_copy_cpu_tree(sample, torch, numpy))
                self.cache_estimated_bytes += estimated
                self.cache_tensor_bytes += tensor_bytes
            self._cache = cache
            self.preprocessing_seconds = time.perf_counter() - started

    def __len__(self) -> int:
        return self.unique_samples * self.repeat_factor

    def __getitem__(self, index: int) -> Any:
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        unique_index = index % self.unique_samples
        if self._cache is not None:
            import numpy
            import torch

            # Collators and model forwards may pop keys or modify tensors.
            return _copy_cpu_tree(self._cache[unique_index], torch, numpy)
        sample = self.dataset[unique_index]
        return _drop_raw(sample) if self.drop_raw else sample


def apply_stage2_training_data(
    dataset: Any,
    *,
    training: bool = True,
    repeat_factor: int | None = None,
    preload_max_bytes: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Return a training-only wrapper plus a JSON-compatible preparation report.

    ``APPLY_REPEAT_DATA`` is a positive integer multiplier (default 1); use the
    same repeat setting for both benchmark arms. An explicit ``repeat_factor``
    argument overrides that environment variable. ``APPLY_PRELOAD=1`` enables eager
    unique-index preparation. ``APPLY_DROP_RAW=1`` removes ``image_frames`` only
    after tokenized input exists. All three switches are off by default.
    ``APPLY_PRELOAD_MAX_BYTES`` sets a positive estimated cache-byte budget;
    an explicit ``preload_max_bytes`` argument takes precedence.
    """
    if repeat_factor is None:
        repeat_factor = int(os.environ.get("APPLY_REPEAT_DATA", "1"))
    repeat_factor = operator.index(repeat_factor)
    if repeat_factor < 1:
        raise ValueError("APPLY_REPEAT_DATA / repeat_factor must be a positive integer")
    requested = {
        "repeat_data": repeat_factor > 1,
        "preload": stage2_flag("APPLY_PRELOAD"),
        "drop_raw": stage2_flag("APPLY_DROP_RAW"),
    }
    report: dict[str, Any] = {
        **{name: False for name in requested},
        "requested": requested,
        "unique_samples": len(dataset),
        "effective_samples": len(dataset),
        "repeat_factor": 1,
        "preprocessing_seconds": 0.0,
        "cache_tensor_bytes": 0,
        "cache_estimated_bytes": 0,
        "preload_max_bytes": None,
    }
    if not training:
        report["skipped_reason"] = "evaluation_dataset"
        return dataset, report
    if not any(requested.values()):
        return dataset, report
    if requested["preload"] and preload_max_bytes is None:
        preload_max_bytes = int(
            os.environ.get("APPLY_PRELOAD_MAX_BYTES", DEFAULT_PRELOAD_MAX_BYTES)
        )
    wrapped = Stage2TrainingDataset(
        dataset,
        repeat_factor=repeat_factor if requested["repeat_data"] else 1,
        preload=requested["preload"],
        drop_raw=requested["drop_raw"],
        preload_max_bytes=(
            preload_max_bytes if preload_max_bytes is not None else DEFAULT_PRELOAD_MAX_BYTES
        ),
    )
    report.update(
        **requested,
        effective_samples=len(wrapped),
        repeat_factor=wrapped.repeat_factor,
        preprocessing_seconds=wrapped.preprocessing_seconds,
        cache_tensor_bytes=wrapped.cache_tensor_bytes,
        cache_estimated_bytes=wrapped.cache_estimated_bytes,
        preload_max_bytes=wrapped.preload_max_bytes if requested["preload"] else None,
    )
    return wrapped, report
