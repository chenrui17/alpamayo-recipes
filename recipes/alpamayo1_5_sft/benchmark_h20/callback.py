# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synchronized optimizer-step timings and optional Nsight capture for Trainer.

This callback never reads a batch, model output, optimizer, scheduler, or RNG.
Only ``profile=True`` enables NVTX and CUDA profiler API calls. CUDA synchronization
at every step end is intentional measurement overhead and is identical on all
ranks. Do not compare profile-mode timings against timing-mode results.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback


class DistributedStepCallback(TrainerCallback):
    """Measure synchronized step-end intervals; exclude step 1 from intervals."""

    def __init__(
        self,
        output_path: str,
        run_id: str,
        steady_start_step: int = 5,
        steady_end_step: int = 20,
        expected_max_steps: int = 20,
        profile: bool = False,
        profile_start_step: int = 5,
        profile_end_step: int = 8,
        global_samples_per_step: float | None = None,
        global_tokens_per_step: float | None = None,
    ) -> None:
        if not 2 <= steady_start_step <= steady_end_step <= expected_max_steps:
            raise ValueError("Require 2 <= steady_start_step <= steady_end_step <= expected_max_steps")
        if profile and not 1 <= profile_start_step <= profile_end_step <= expected_max_steps:
            raise ValueError("Profile window must fit inside expected_max_steps")
        if not run_id:
            raise ValueError("A unique run_id is required")
        for value in (global_samples_per_step, global_tokens_per_step):
            if value is not None and value <= 0:
                raise ValueError("Throughput counts must be positive when supplied")
        self.output_path = Path(output_path)
        self.run_id = str(run_id)
        self.steady_start_step = int(steady_start_step)
        self.steady_end_step = int(steady_end_step)
        self.expected_max_steps = int(expected_max_steps)
        self.profile = bool(profile)
        self.profile_start_step = int(profile_start_step)
        self.profile_end_step = int(profile_end_step)
        self.global_samples_per_step = global_samples_per_step
        self.global_tokens_per_step = global_tokens_per_step
        self.records: list[dict[str, Any]] = []
        self.logs: list[dict[str, Any]] = []
        self._last_step_end: float | None = None
        self._capture_active = False
        self._nvtx_open = False
        self._capture_started_at_step: int | None = None
        self._capture_stopped_at_step: int | None = None

    @staticmethod
    def _distributed() -> bool:
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    @staticmethod
    def _cuda() -> bool:
        return torch.cuda.is_available()

    @staticmethod
    def _profiler_call(method: str) -> None:
        # Use the CUDA runtime API explicitly so nsys --capture-range=cudaProfilerApi
        # can delimit the same optimizer-step window on every process.
        result = getattr(torch.cuda.cudart(), method)()
        error = result[0] if isinstance(result, tuple) else result
        if error is not None and int(error) != 0:
            raise RuntimeError(f"{method} failed with CUDA error {error}")

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.records.clear()
        self.logs.clear()
        self._last_step_end = None
        self._capture_started_at_step = None
        self._capture_stopped_at_step = None
        if self.profile and not self._cuda():
            raise RuntimeError("profile=True requires CUDA")
        if self._cuda():
            # Includes allocations already alive when training begins, while
            # excluding an unrelated earlier peak from model/checkpoint loading.
            torch.cuda.reset_peak_memory_stats()

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if not self.profile:
            return
        step = int(state.global_step) + 1
        if step == self.profile_start_step:
            self._profiler_call("cudaProfilerStart")
            self._capture_active = True
            self._capture_started_at_step = step
        torch.cuda.nvtx.range_push(f"optimizer_step_{step}")
        self._nvtx_open = True

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        before_sync = time.perf_counter()
        cuda_available = self._cuda()
        if cuda_available:
            torch.cuda.synchronize()
        now = time.perf_counter()
        step = int(state.global_step)
        record: dict[str, Any] = {
            "step": step,
            "sync_seconds": now - before_sync,
            "step_seconds": None if self._last_step_end is None else now - self._last_step_end,
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if cuda_available else None,
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()) if cuda_available else None,
        }
        self.records.append(record)
        self._last_step_end = now
        if self.profile:
            if self._nvtx_open:
                torch.cuda.nvtx.range_pop()
                self._nvtx_open = False
            if step == self.profile_end_step and self._capture_active:
                self._profiler_call("cudaProfilerStop")
                self._capture_active = False
                self._capture_stopped_at_step = step

    def on_log(
        self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        # Trainer already materialized these values. Never pull tensors from the
        # model or force .item() on another tensor just for instrumentation.
        if logs:
            values = {key: value for key, value in logs.items() if isinstance(value, (int, float))}
            self.logs.append({**values, "step": int(state.global_step)})

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self._nvtx_open:
            torch.cuda.nvtx.range_pop()
            self._nvtx_open = False
        if self._capture_active:
            self._profiler_call("cudaProfilerStop")
            self._capture_active = False
            # An early stop must not be mistaken for the requested full capture.
        distributed = self._distributed()
        rank = int(torch.distributed.get_rank()) if distributed else 0
        world_size = int(torch.distributed.get_world_size()) if distributed else 1
        cuda_available = self._cuda()
        payload = {
            "rank": rank,
            "global_step": int(state.global_step),
            "records": self.records,
            "logs": self.logs,
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if cuda_available else None,
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()) if cuda_available else None,
            "capture_started_at_step": self._capture_started_at_step,
            "capture_stopped_at_step": self._capture_stopped_at_step,
        }
        ranks: list[Any] = [None] * world_size
        if distributed:
            # All ranks participate, including ranks other than world-process-zero.
            torch.distributed.all_gather_object(ranks, payload)
        else:
            ranks[0] = payload
        if rank != 0:
            return
        ranks.sort(key=lambda item: item["rank"])
        by_rank = [{int(item["step"]): item for item in item_rank["records"]} for item_rank in ranks]
        common_steps = set(by_rank[0])
        for records in by_rank[1:]:
            common_steps.intersection_update(records)
        rank_max_records = []
        for step in sorted(common_steps):
            records = [records[step] for records in by_rank]
            intervals = [record["step_seconds"] for record in records]
            slowest_rank = None
            if all(value is not None for value in intervals):
                slowest_index = max(range(len(intervals)), key=lambda index: intervals[index])
                interval = intervals[slowest_index]
                slowest_rank = ranks[slowest_index]["rank"]
            else:
                interval = None
            rank_max_records.append({
                "step": step,
                "step_seconds": interval,
                "slowest_rank": slowest_rank,
                "max_sync_seconds": max(record["sync_seconds"] for record in records),
            })
        requested_steps = set(range(self.steady_start_step, self.steady_end_step + 1))
        available_steps = {record["step"] for record in rank_max_records if record["step_seconds"] is not None}
        complete = all(item["global_step"] >= self.expected_max_steps for item in ranks)
        complete = complete and requested_steps.issubset(available_steps)
        if self.profile:
            complete = complete and all(
                item["capture_started_at_step"] == self.profile_start_step
                and item["capture_stopped_at_step"] == self.profile_end_step
                for item in ranks
            )
        result = {
            "schema_version": 1,
            "run_id": self.run_id,
            "mode": "profile" if self.profile else "timing",
            "complete": complete,
            "world_size": world_size,
            "expected_max_steps": self.expected_max_steps,
            "steady_start_step": self.steady_start_step,
            "steady_end_step": self.steady_end_step,
            "first_interval_excluded": True,
            "measurement": "CUDA-synchronized optimizer-step-end wall intervals; rank maximum per step",
            "global_samples_per_step": self.global_samples_per_step,
            "global_tokens_per_step": self.global_tokens_per_step,
            "profile_window": [self.profile_start_step, self.profile_end_step] if self.profile else None,
            "ranks": ranks,
            "rank_max_records": rank_max_records,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_name(self.output_path.name + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, self.output_path)
