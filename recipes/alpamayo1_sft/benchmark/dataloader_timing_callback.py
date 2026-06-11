# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Log per-step data-pipeline timing and write a JSON report at train end."""

from __future__ import annotations

from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from alpamayo1_sft.benchmark.dataloader_timing import (
    local_rank,
    record,
    summarize_steps,
    write_report,
)
from alpamayo1_sft.benchmark.trainer_timing import consume_step_wall_seconds


class DataloaderTimingCallback(TrainerCallback):
    def __init__(
        self,
        output_path: str,
        log_start_step: int = 5,
        log_every: int = 1,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.output_path = output_path
        self.log_start_step = log_start_step
        self.log_every = log_every
        self.meta = meta or {}
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        wall = consume_step_wall_seconds()
        if wall is not None:
            record("step.wall", wall)

        step = state.global_step
        if step < self.log_start_step or step % self.log_every != 0:
            return

        summary = summarize_steps([step]).get(str(step), {})
        derived = summary.get("derived_ms", {})
        phases = summary.get("phases", {})
        if not derived:
            return

        collate_tok = phases.get("collate.tokenizer", {}).get("avg_ms", 0.0)
        ds_load = derived.get("dataset_load_total", 0.0)
        ds_pre = derived.get("dataset_preprocess_total", 0.0)
        gpu_idle = derived.get("gpu_idle_serial_est", 0.0)

        print(
            f"[DL-TIMING] rank={local_rank()} step={step} "
            f"wall={derived.get('step_wall', 0):.0f}ms "
            f"fetch={derived.get('fetch_total', 0):.0f}ms "
            f"(load={ds_load:.0f} pre={ds_pre:.0f} collate={derived.get('collate_total', 0):.0f} "
            f"ipc~={derived.get('worker_wait_est', 0):.0f}) "
            f"gpu={derived.get('gpu_total', 0):.0f}ms "
            f"opt/sync~={derived.get('other_est', 0):.0f}ms "
            f"gpu_idle~={gpu_idle:.0f}ms",
            flush=True,
        )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not state.is_world_process_zero and local_rank() != 0:
            # Each rank writes its own report for straggler comparison.
            pass
        tail_start = max(self.log_start_step, 1)
        step_ids = list(range(tail_start, state.global_step + 1))
        write_report(
            self.output_path.replace(".json", f"_rank{local_rank()}.json"),
            meta={
                **self.meta,
                "global_steps": state.global_step,
                "log_start_step": self.log_start_step,
            },
            step_ids=step_ids,
        )
        if local_rank() == 0:
            print(f"[DL-TIMING] wrote {self.output_path.replace('.json', '_rank0.json')}", flush=True)
