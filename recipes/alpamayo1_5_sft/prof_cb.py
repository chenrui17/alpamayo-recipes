import time
import torch
from transformers import TrainerCallback


class TorchProfCallback(TrainerCallback):
    """Times dataloader-wait vs step compute. on_step_begin fires after the batch is
    fetched; the gap between previous on_step_end and this on_step_begin is data wait."""

    def __init__(self, warmup=6, nprof=8, result_path="/tmp/prof_breakdown.txt"):
        self.warmup = warmup
        self.nprof = nprof
        self.result_path = result_path
        self._prev_end = None
        self._step_begin = None
        self._data_waits = []
        self._step_computes = []
        self._n = 0

    def on_step_begin(self, args, state, control, **kw):
        now = time.perf_counter()
        if self._prev_end is not None:
            self._data_waits.append(now - self._prev_end)
        self._step_begin = now

    def on_step_end(self, args, state, control, **kw):
        torch.cuda.synchronize()
        now = time.perf_counter()
        if self._step_begin is not None:
            self._step_computes.append(now - self._step_begin)
        self._prev_end = now
        self._n += 1

    def on_train_end(self, args, state, control, **kw):
        if args.local_rank not in (-1, 0):
            return
        dw = self._data_waits[self.warmup:]
        sc = self._step_computes[self.warmup:]
        def avg(x):
            return sum(x) / len(x) if x else float("nan")
        msg = (f"[PROF-BREAKDOWN] n={len(sc)} "
               f"data_wait(before step)={avg(dw)*1000:.1f}ms "
               f"step_compute(begin->end sync)={avg(sc)*1000:.1f}ms "
               f"total={(avg(dw)+avg(sc))*1000:.1f}ms")
        print(msg, flush=True)
        with open(self.result_path, "w") as f:
            f.write(msg + "\n")
            f.write("data_waits_ms=" + ",".join(f"{x*1000:.0f}" for x in dw) + "\n")
            f.write("step_computes_ms=" + ",".join(f"{x*1000:.0f}" for x in sc) + "\n")
