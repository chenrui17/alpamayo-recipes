# AR1.5 SFT Throughput Benchmark

Updated: 2026-05-29T09:31:11.761612+00:00
Hardware: 8x GPU, single node
Method: 15 steps/run, average of last 10 stable steps
Baseline avg step: **26.480s** (gradient_accumulation_steps=4)
Target (+80% throughput): **14.711s** avg step

> ⚠️ `vs Baseline` is only meaningful for runs with the same `grad_accum` as baseline.
> Rows with a different `grad_accum` show `n/a*` and are listed in section B for reference.

## A. Apples-to-apples (grad_accum=4, comparable to baseline)

| Run | Avg Step (s) | vs Baseline | grad_accum | grad_ckpt | workers | Optimizations |
|-----|-------------:|------------:|-----------:|:---------:|--------:|---------------|
| best_preload_ga4 | 8.836 | +199.7% | 4 | off | 8 | zip,collate,tf32,cudnn,preload |
| production_opt | 11.968 | +121.3% | 4 | on | 8 | zip,collate,tf32,cudnn |
| dl8_zip_collate_tf32 | 11.994 | +120.8% | 4 | on | 8 | zip,collate,tf32,cudnn,matmul=high |
| dl8_zip_collate_cache | 12.001 | +120.6% | 4 | on | 8 | zip,collate |
| dl8_no_cam_text | 15.646 | +69.2% | 4 | on | 8 | zip,collate,tf32,cudnn |
| dl_workers8_persistent | 17.413 | +52.1% | 4 | on | 8 | - |
| baseline | 26.480 | +0.0% | 4 | on | 2 | - |

## B. Different batch semantics (NOT directly comparable to baseline)

Each `step` only covers 1 micro-batch instead of 4; raw step time looks lower but the per-optimizer-step work is smaller.

| Run | Avg Step (s) | vs Baseline | grad_accum | grad_ckpt | workers | Optimizations |
|-----|-------------:|------------:|-----------:|:---------:|--------:|---------------|
| best_preload_cache | 2.204 | n/a* | 1 | off | 4 | zip,collate,tf32,cudnn,preload |
| best_no_deepspeed | 2.303 | n/a* | 1 | off | 4 | zip,collate,tf32,cudnn,preload,no_ds |
| dl8_channels_last | 3.400 | n/a* | 1 | off | 8 | zip,collate,tf32,cudnn,ch_last |
| dl8_no_grad_ckpt | 4.920 | n/a* | 1 | off | 8 | zip,collate,tf32,cudnn |
| dl8_grad_accum1 | 5.687 | n/a* | 1 | on | 8 | zip,collate,tf32,cudnn |

## D. Known incompatibilities / caveats

| Item | Detail |
|------|--------|
| `dl8_torch_compile` | `torch.compile(inductor)` fails on Qwen3-VL because Dynamo cannot fake-trace `flash_attn._flash_attn_varlen_forward`. Re-enable after transformers/flash-attn add Dynamo support. See `results/dl8_torch_compile.log`. |
| `channels_last` on Qwen3-VL visual encoder | Qwen3-VL ViT uses Linear patch embedding (no Conv2d). `apply_model_optimizations` reports `channels_last_visual_conv2d=0`, so the flag is effectively a **no-op** on this model. The speedup observed in `dl8_channels_last` comes from its other knobs (`grad_accum=1` + `grad_ckpt=off`), not from channels_last. |
| `preload_dataset_cache` | Loads the **entire** train dataset into per-worker RAM. Safe only for this 20-sample nav demo; do NOT enable in production. |
| `dl8_no_cam_text` regression vs `dl8_zip_collate_tf32` | The former omits `matmul_precision=high` (intentional, see `EXPERIMENTS`), which is why disabling `include_camera_ids/frame_nums` does NOT translate into a speedup here. |

**Best comparable (grad_accum=4):** `best_preload_ga4` @ 8.836s (+199.7% throughput vs baseline).

**Production-safe recommendation:** `production_opt` (no train-semantics change, no out-of-RAM risk, no grad-ckpt trade-off).
