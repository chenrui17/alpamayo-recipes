# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainer callback that records per-optimizer-step wall time."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class StepTimerCallback(TrainerCallback):
    """Record wall-clock duration of each optimizer step (after grad accumulation)."""

    def __init__(
        self,
        output_path: str,
        run_name: str = "unknown",
        config: dict[str, Any] | None = None,
        stable_tail: int = 40,
    ) -> None:
        self.output_path = Path(output_path)
        self.run_name = run_name
        self.config = config or {}
        self.stable_tail = stable_tail
        self._step_start: float | None = None
        self._step_times: list[float] = []
        self._warmup_steps = 0

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._warmup_steps = 0
        self._step_times = []
        self._step_start = time.perf_counter()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        now = time.perf_counter()
        if self._step_start is not None:
            self._step_times.append(now - self._step_start)
        self._step_start = now

        if state.global_step >= args.max_steps:
            control.should_training_stop = True

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not state.is_world_process_zero:
            return

        tail = self._step_times[-self.stable_tail :] if self._step_times else []
        avg_tail = sum(tail) / len(tail) if tail else float("nan")
        avg_all = sum(self._step_times) / len(self._step_times) if self._step_times else float("nan")

        result = {
            "run_name": self.run_name,
            "config": self.config,
            "max_steps": args.max_steps,
            "global_steps_recorded": len(self._step_times),
            "step_times_sec": self._step_times,
            "avg_step_sec_all": avg_all,
            "avg_step_sec_stable_tail": avg_tail,
            "stable_tail_n": len(tail),
            "per_device_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_workers": args.dataloader_num_workers,
            "pin_memory": args.dataloader_pin_memory,
            "persistent_workers": args.dataloader_persistent_workers,
            "prefetch_factor": args.dataloader_prefetch_factor,
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(
            f"[BENCHMARK] run={self.run_name} "
            f"avg_stable={avg_tail:.3f}s ({len(tail)} steps) "
            f"avg_all={avg_all:.3f}s ({len(self._step_times)} steps) "
            f"-> {self.output_path}",
            flush=True,
        )
