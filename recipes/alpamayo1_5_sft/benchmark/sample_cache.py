# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wrap a map-style dataset with an in-process sample cache."""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset


class InMemorySampleCache(Dataset):
    """Cache ``dataset[i]`` after first access (per worker process)."""

    def __init__(self, dataset: Dataset, preload: bool = False) -> None:
        self.dataset = dataset
        self._cache: dict[int, Any] = {}
        if preload:
            for i in range(len(dataset)):
                self._cache[i] = dataset[i]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        if idx not in self._cache:
            self._cache[idx] = self.dataset[idx]
        return self._cache[idx]
