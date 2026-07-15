# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import time
from pathlib import Path
from transformers import TrainerCallback
import torch


class StepTimerCallback(TrainerCallback):
    def __init__(self, warmup_steps=10, result_path='benchmark/results/result.json', run_name='run', stable_tail=40):
        self.warmup_steps = warmup_steps
        self.result_path = Path(result_path)
        self.run_name = run_name
        self.stable_tail = stable_tail
        self._step_start = None
        self._step_times = []
        self._current_step = 0

    def on_train_begin(self, args, state, control, **kw):
        torch.cuda.reset_peak_memory_stats()
        self._step_times = []
        self._current_step = 0
        self._step_start = time.perf_counter()

    def on_step_end(self, args, state, control, **kw):
        now = time.perf_counter()
        self._current_step += 1
        if self._current_step > self.warmup_steps:
            self._step_times.append(now - self._step_start)
        self._step_start = now

    def on_train_end(self, args, state, control, **kw):
        if not args.local_rank in (-1, 0):
            return
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        tail = self._step_times[-self.stable_tail:] if self._step_times else []
        avg_tail = sum(tail) / len(tail) if tail else float('nan')
        avg_all = sum(self._step_times) / len(self._step_times) if self._step_times else float('nan')
        result = {
            'run_name': self.run_name,
            'step_times_sec': self._step_times,
            'avg_step_sec_all': avg_all,
            'avg_step_sec_stable_tail': avg_tail,
            'stable_tail_n': len(tail),
            'peak_gpu_mem_gb': peak_mem_gb,
            'num_measured_steps': len(self._step_times),
            'warmup_steps': self.warmup_steps,
        }
        self.result_path.parent.mkdir(parents=True, exist_ok=True)
        with self.result_path.open('w') as f:
            json.dump(result, f, indent=2)
        tp = 1.0 / avg_tail if avg_tail > 0 else 0
        print(f'\n[BENCHMARK] run={self.run_name}\n'
              f'  avg_step (stable tail {len(tail)} steps): {avg_tail:.3f}s = {avg_tail*1000:.1f}ms\n'
              f'  throughput: {tp:.3f} steps/s\n'
              f'  peak GPU memory (rank 0): {peak_mem_gb:.2f} GB', flush=True)
