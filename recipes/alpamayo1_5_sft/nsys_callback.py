# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
HF TrainerCallback that:
  - wraps each optimizer step with an NVTX range "step_N"
  - calls cudaProfilerStart() at capture_start_step and cudaProfilerStop() at capture_end_step
"""
import torch
import torch.cuda.nvtx as nvtx
from transformers import TrainerCallback


class NsysCallback(TrainerCallback):
    def __init__(self, capture_start_step: int = 6, capture_end_step: int = 9):
        self.capture_start = capture_start_step
        self.capture_end = capture_end_step
        self._step = 0
        self._profiling = False

    def on_step_begin(self, args, state, control, **kw):
        self._step += 1
        nvtx.range_push(f"step_{self._step}")
        if self._step == self.capture_start:
            print(f'[NSYS] cudaProfilerStart at step {self._step}', flush=True)
            torch.cuda.profiler.start()
            self._profiling = True

    def on_step_end(self, args, state, control, **kw):
        nvtx.range_pop()  # close step_N
        if self._profiling and self._step >= self.capture_end:
            print(f'[NSYS] cudaProfilerStop at step {self._step}', flush=True)
            torch.cuda.profiler.stop()
            self._profiling = False
