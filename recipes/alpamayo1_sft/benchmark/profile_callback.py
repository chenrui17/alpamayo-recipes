# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainer callback to bracket Nsight capture on selected optimizer steps."""

from __future__ import annotations

from typing import Any

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class ProfileRangeCallback(TrainerCallback):
    """Start/stop CUDA profiler API around [profile_start, profile_end] steps.

    Capture starts at the **end** of step ``profile_start - 2`` so the next
    ``get_batch_samples`` / ``collate_fn`` (including DataLoaderShard's one-batch
    prefetch) is inside the nsys window.
    """

    def __init__(self, profile_start: int = 5, profile_end: int = 7) -> None:
        self.profile_start = profile_start
        self.profile_end = profile_end
        self._started = False

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        step = state.global_step
        # Start one step earlier: DataLoaderShard keeps one collated batch prefetched
        # before the next optimizer step's get_batch_samples runs.
        if step == self.profile_start - 2 and not self._started:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            self._started = True
        if step == self.profile_end and self._started:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            self._started = False
