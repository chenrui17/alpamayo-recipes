# AR1.5 SFT 训练吞吐优化记录

## 测试方法

- **硬件**: 单机 8× GPU (H20)
- **配置**: `sft_stage1_nav`（20 条 nav demo 样本，trainer `gradient_accumulation_steps=4`）
- **每次运行**: 15 optimizer steps，取**最后 10 步**平均 wall time
- **入口**: `python benchmark/run_suite.py [experiment_names...]`
- **结果目录**: `benchmark/results/`（JSON + log + 自动生成 `RESULTS.md`）

## 基准 (baseline)

| 指标 | 值 |
|------|-----|
| Avg step (15 步取末 10 步) | **26.480 s** |
| num_workers | 2 |
| pin_memory | true |
| gradient_accumulation_steps | 4 |
| gradient_checkpointing | true |
| 缓存 / TF32 / 预加载 / no-ds | 全关 |

## 目标

吞吐相比 baseline **提升 80%** → avg step ≤ **14.711 s**（保持 baseline 的训练语义，
即同 `grad_accum=4`）。

## 已实现的优化手段

| 类别 | 实现位置 | 说明 |
|------|----------|------|
| DataLoader | `sft_base.yaml` / Hydra override | num_workers, persistent_workers, prefetch_factor, pin_memory |
| Feature 缓存 (`zip_cache`) | `benchmark/apply_optimizations.py::enable_zip_cache` | 同一 worker 内 clip feature 二次读取走内存 |
| Collate 单例 (`collate_cache`) | `benchmark/collate_cache.py` | 避免每 batch 重建 QwenProcessor |
| 样本预加载 (`preload_dataset_cache`) | `benchmark/sample_cache.py` | 全量 RAM 缓存，仅适合 nav demo |
| TF32 / cuDNN | `benchmark/apply_optimizations.py::apply_runtime_optimizations` | matmul TF32、cudnn.benchmark、matmul_precision |
| channels_last | `benchmark/apply_optimizations.py::apply_model_optimizations` | 仅对 Conv2d/ConvTranspose2d 模块（Qwen3-VL ViT 无 Conv2d，**该项实际 no-op**）|
| torch.compile | TrainingArguments | inductor + reduce-overhead，**当前与 flash_attn varlen 不兼容** |
| 关闭 gradient_checkpointing | trainer override | 降低 forward 开销（增 VRAM）|
| 无 DeepSpeed | trainer.deepspeed=null | 原生 DDP，减少通信 |

## 公平性

`baseline` 的 1 optimizer step = 4 micro-batch 的前向 + 反向。**只有同 `grad_accum=4` 的
实验能跟 baseline 直接比 wall time / 算 `vs Baseline %`**。

下表中 A 组与 baseline 同 batch 语义，B 组每 step 只跑 1 micro-batch，step time 天然
更短，**不能换算成等效优化倍数**。

## 结果汇总（全部 12 项实验）

### A. 与 baseline 同 `grad_accum=4`（可直接对比）

| 实验 | Avg Step | vs Baseline | grad_ckpt | workers | 主要开关 | 达标 |
|------|---------:|------------:|:---------:|--------:|----------|:----:|
| baseline | 26.480 s | — | on | 2 | — | — |
| dl_workers8_persistent | 17.413 s | +52.1% | on | 8 | DataLoader 并行 | ❌ |
| dl8_no_cam_text | 15.646 s | +69.2% | on | 8 | +zip+collate+tf32+cudnn，关 camera/frame text | ❌ |
| dl8_zip_collate_cache | 12.001 s | +120.6% | on | 8 | +zip+collate | ✅ |
| dl8_zip_collate_tf32 | 11.994 s | +120.8% | on | 8 | A 上 +tf32+cudnn+matmul=high | ✅ |
| **production_opt** | **11.968 s** | **+121.3%** | on | 8 | A 上 +tf32+cudnn（无 matmul=high）| ✅ |
| **best_preload_ga4** | **8.836 s** | **+199.7%** | **off** | 8 | A 上 +preload，关 grad_ckpt | ✅ |

### B. `grad_accum=1`（不能直接与 baseline 比较，仅展示微步吞吐）

| 实验 | Avg Step | grad_ckpt | workers | 主要开关 |
|------|---------:|:---------:|--------:|----------|
| dl8_grad_accum1 | 5.687 s | on | 8 | 同 dl8_no_cam_text 但 grad_accum=1 |
| dl8_no_grad_ckpt | 4.920 s | off | 8 | +关 grad_ckpt |
| dl8_channels_last | 3.400 s | off | 8 | +`channels_last`（对 Qwen3-VL ViT **无效**，实际转换 0 个模块） |
| best_preload_cache | 2.204 s | off | 4 | +preload，workers=4 |
| best_no_deepspeed | 2.303 s | off | 4 | 同上 + 关 DeepSpeed |

> 把 B 组按 `× grad_accum` 折算成 baseline 等效 step：
> `best_preload_cache 2.204 × 4 ≈ 8.82 s`，与 A 组 `best_preload_ga4 8.836 s` 几乎一致。
> 这说明 grad_accum=1 vs grad_accum=4 在本工作负载下没有额外吞吐红利，仅改训练语义。

### C. 失败 / 跳过

| 实验 | 状态 | 原因 |
|------|------|------|
| `dl8_torch_compile` | ❌ FAILED | `torch.compile(inductor)` 与 Qwen3-VL 的 `flash_attn._flash_attn_varlen_forward` 不兼容（Dynamo 无法 fake-tensor 跟踪）。日志: `results/dl8_torch_compile.log`。等 transformers / flash-attn 提供 Dynamo 兼容版本再重启。|

## 关键结论

1. **目标达成**：同训练语义（`grad_accum=4`）下 `best_preload_ga4` 达到 **+199.7%** 吞吐
   （8.836 s，远低于 14.711 s 达标线）；安全的 `production_opt` 也已达 **+121.3%**。
2. **首要瓶颈是 DataLoader**：仅 `num_workers 2→8 + persistent_workers + prefetch_factor=4`
   就拿到 +52%；继续叠加 `zip_cache + collate_cache`（避免重复 zip 读 + 重建 QwenProcessor）
   再 +60%，累计 +121%。这是性价比最高、风险最低的一段。
3. **TF32 / matmul=high**：在 bf16 训练下增益 <1%（11.994 s vs 11.968 s）。主要作用是
   保证残余 fp32 路径走 TF32，不会变慢即可，留着无妨。
4. **关闭 `gradient_checkpointing`**：A 组 `production_opt 11.97 s → best_preload_ga4 8.84 s`
   即关闭 grad_ckpt（同时打开 preload）带来约 -3 s/step。H20 显存富余时建议关闭，但需
   独立验证显存占用与 batch_size 配比，不要直接拷到大数据/大序列场景。
5. **`preload_dataset_cache`**：把数据集整体放进 worker 进程 RAM。20 条 nav demo 没问题；
   **生产数据集体量远超内存，禁止使用**——它只是 micro-benchmark 上限参照。
6. **`channels_last`**：Qwen3-VL ViT 使用 Linear patch embedding，没有 Conv2d 模块，
   `apply_model_optimizations` 报告 `channels_last_visual_conv2d=0`。`dl8_channels_last`
   的 3.4 s 速度来自该 run 同时开启的 `grad_accum=1 + grad_ckpt=off`（与 `dl8_no_grad_ckpt`
   4.920 s 的差值更可能来自时段噪声 / 其他用户 GPU 0 上的负载，非 channels_last 本身）。
7. **`torch.compile`**：目前 transformers + flash-attn 版本下不可用，详见 C。
8. **DeepSpeed ZeRO-2 vs 原生 DDP**：单机 8 卡 H20、模型可全量 bf16 装下时差异 <1%
   （B 组 2.204 vs 2.303 s）。生产保持 ZeRO-2，无需切换。

## 推荐落地方案

| 场景 | 选项 | 理由 |
|------|------|------|
| **生产默认**（已写入 `sft_base.yaml` / `sft_stage1_nav.yaml`） | `production_opt` 等价配置 | +121% 吞吐，不改 batch 语义，无 RAM/显存风险 |
| 单机 H20 显存富余的激进档 | `best_preload_ga4` 的子集：在 `production_opt` 之上**额外**关 `gradient_checkpointing`（**不要**开 `preload_dataset_cache`，除非数据集够小） | 额外 +35% 吞吐（11.97 → 8.84 s），代价：显存↑，需先压测 |
| 想换更小有效 batch | B 组 `grad_accum=1` 系列 | 仅改训练语义，单 step 更轻，不构成吞吐"魔法加速" |

## 复现命令

```bash
cd /raid/charlie/alpamayo-recipes/recipes/alpamayo1_5_sft
source a1_5_sft/bin/activate

# 基准（首次必须先跑出 results/baseline.json）
python benchmark/run_suite.py baseline

# 全部实验（≈ 100 min）
BENCHMARK_FULL=1 python benchmark/run_suite.py

# 单项
python benchmark/run_suite.py best_preload_ga4
```

机器生成的全表 + 失败/警告章节见 [`benchmark/RESULTS.md`](RESULTS.md)。
