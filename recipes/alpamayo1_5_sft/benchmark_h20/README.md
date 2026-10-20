# Stage2 benchmark harness

Hydra target: `alpamayo1_5_sft.benchmark_h20.callback.DistributedStepCallback`.
All ranks must instantiate the callback. Required constructor arguments are
`output_path` and a unique `run_id`. The training entry point should instantiate
the callbacks in its Hydra `callbacks` mapping and pass them to Trainer.

## Timing contract

Each optimizer-step end calls `torch.cuda.synchronize()` on every rank and records
the wall interval since that rank's previous synchronized step end. The first
step has no interval. The default steady window contains **steps 5 through 20,
inclusive (16 intervals)**, including dataloading/collation, logging, and the
CUDA work between those boundaries. Step 20's subsequent logging is outside the
last interval. Step-begin synchronization is intentionally absent. The metric is
the **mean of the per-step maximum across ranks**; this differs from the maximum
of each rank's mean. This instrumentation includes its own small CPU bookkeeping
overhead and the deliberate device synchronization cost.

`steps.json` preserves every rank's per-step interval, synchronization call time,
peak allocated/reserved CUDA bytes, and already-materialized numeric Trainer
logs (including `loss`). Memory peaks reset when training begins; allocations
already alive remain counted. There are no extra forward passes, tensor reads,
RNG calls, batch reads, optimizer changes, sampler changes, or scheduler changes.
Every rank participates in one `all_gather_object` at train end; only rank zero
writes the aggregate JSON atomically. An early or incomplete run is labeled
`complete=false`.

`analyze.py` uses only the standard library. It recomputes rank maxima from the
raw rank records, verifies the complete steady window, and reports mean, median,
linearly interpolated p90, per-rank timings, loss logs, memory, and optional
sample/token throughput. Global throughput is the explicit global count per
optimizer step divided by mean rank-max seconds. Per-rank throughput assumes
the global count is evenly divided among ranks. Token counts are supplied by
the caller; the harness does not equate padded sequence length with true tokens.

## Portable runner

Run one variant per invocation from an environment where the recipe package and
training dependencies are importable. Replace all illustrative CLI values with
actual inputs. `--python` selects the interpreter that launches both Hydra
composition and `torch.distributed.run`; no machine or virtualenv path is built
into the runner. `--work-dir` is the subprocess working directory and `--code-root`
selects the Git checkout to snapshot (defaults to the working directory).

```bash
python -m alpamayo1_5_sft.benchmark_h20.run \
  --python /path/to/training/python \
  --work-dir /path/to/alpamayo-recipes \
  --result-dir /path/to/benchmark-results \
  --run-id baseline-b1-r1 \
  --config-path pkg://alpamayo1_5_sft/configs \
  --config-name sft_stage2_nav \
  --nproc-per-node 8 --global-samples-per-step 8 \
  --override model.pretrained_model_name_or_path=/path/to/checkpoint \
  --override data.train_dataset.local_dir=/path/to/pai \
  --override data.train_dataset.annotations_path=/path/to/navigation.json \
  --override data.val_dataset.local_dir=/path/to/pai \
  --override data.val_dataset.annotations_path=/path/to/navigation.json \
  --override trainer.per_device_train_batch_size=1 \
  --override trainer.gradient_accumulation_steps=1 \
  --override '++trainer.save_strategy=no' \
  --override '++trainer.eval_strategy=no'
```

Pass each performance/optimizer/seed choice explicitly using `--override`; the
runner does not choose optimization variants. It sets only max steps, logging
frequency (default 1), output directories, and callback measurement settings.
`--global-samples-per-step` must reflect microbatch × accumulation × world size.
An optional `--global-tokens-per-step` must reflect the actual workload definition.
Child environment overrides use repeated `--env NAME=VALUE`. The manifest records
an allowlist of benchmark switches and runtime settings; other override values
are omitted. The resolved Hydra config itself is saved in full.

Each launch creates a fresh run directory and refuses an existing `--run-id`.
Artifacts include the exact subprocess command, resolved Hydra configuration and
its SHA256, code Git HEAD/status/tracked patch, a verbatim snapshot and SHA256 of
this harness, training log, all-rank measurements, summary, and a manifest.
Untracked code outside this harness is listed by Git status but is not copied;
commit or separately preserve those files when making a fully reproducible run.
Success requires a zero training exit status, matching run ID/mode/world size,
the exact final optimizer step on every rank, and all 16 steady intervals. A
preexisting JSON can never satisfy a new launch. Failed launches keep their
diagnostic artifacts and get `status=failed`.

## Separate Nsight capture

Use a separate fresh invocation with `--mode profile`. The runner wraps torchrun
with `nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop`.
The callback calls `cudaProfilerStart` at `on_step_begin(global_step=4)` and
`cudaProfilerStop` after synchronizing `on_step_end(global_step=8)`, so the CUDA
capture covers optimizer steps 5–8. NVTX ranges are named `optimizer_step_N`.
The default profile run still executes 20 steps; to stop after the capture, pass
`--max-steps 8 --steady-end-step 8`. All other model/data/optimizer overrides must
match the corresponding timing run. Profile-mode timings are labeled separately
and must not be used as speedup measurements. A successful profile launch also
requires a `.nsys-rep` artifact. Set `--nsys /path/to/nsys` or pass extra options
as `--nsys-option=--sample=none` if necessary.

```bash
python -m alpamayo1_5_sft.benchmark_h20.analyze /path/to/run/steps.json \
  --output /path/to/run/summary-recomputed.json

python -m unittest alpamayo1_5_sft.benchmark_h20.test_harness -v
```

The contract tests use fake Torch/Trainer implementations and need no GPU,
PyTorch, Transformers, Hydra, or Nsight installation.

## Real-input checks

In the training environment, run the numerical checker on an existing local
checkpoint and one real navigation sample. It creates no optimizer and compares
all trainable gradients against the same loaded baseline weights. Prefix and
trajectory are isolated before Liger is patched; `combined` is checked last.
Trajectory checks warm both fixed factors and require actual reuse in the
measured forward. The report states explicit BF16 error tolerances.

```bash
python -m alpamayo1_5_sft.benchmark_h20.check_correctness \
  --checkpoint "$CHECKPOINT" --data "$DATA" --annotations "$ANNOTATIONS" \
  --chunk-ids "$CHUNK_IDS" --variants skiphead prefix trajectory liger combined \
  --output /work/results/correctness-new.json

python -m alpamayo1_5_sft.benchmark_h20.check_data \
  --checkpoint "$CHECKPOINT" --data "$DATA" --annotations "$ANNOTATIONS" \
  --chunk-ids "$CHUNK_IDS" --expected-unique-samples 20 \
  --output /work/results/data-equivalence-new.json
```

The second command hides GPUs and loads configuration only. It compares all
filtered real samples, ordinary/cached collation, exact model-input tensors,
read determinism and cache-return isolation. It permits equivalent sequence
metadata containers (Hydra ListConfig versus a plain list). CPU preload is
appropriate only for deterministic preprocessing and has an explicit memory
budget; it should not freeze random augmentation in a general training run.
