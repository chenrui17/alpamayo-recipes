# Alpamayo-1 SFT 训练吞吐优化

基于 `alpamayo1_5_sft` 的数据管线优化，合入 `alpamayo1_sft` 并对 **Stage-1 CoC disabled / CoC enabled** 两个任务做 benchmark。

与 A1.5 nav demo（20 样本、epoch 极短）不同，A1 SFT 在 19 chunks 下有 **~1877 clips/epoch**，部分优化收益与风险不同。

---

## 1. 优化项与适用性

| 优化 | 实现 | A1.5 nav | A1 SFT（多 step/epoch） | 生产默认 |
|------|------|----------|-------------------------|----------|
| DataLoader 8w + persistent + prefetch | `sft_base.yaml` | ✅ | ✅ | **ON** |
| collate_cache | `benchmark/collate_cache.py` | ✅ | ✅ 与 clip 数无关 | **ON** |
| zip_cache | `apply_optimizations.py` | ✅ | ⚠️ 小数据集收益大；大 epoch 占 RAM | **ON**（见建议） |
| TF32 + cudnn.benchmark | runtime opts | ✅ | ✅ | **ON** |
| preload_dataset_cache | 全量 RAM | ✅ 仅 20 样本 | ❌ | **OFF** |
| channels_last | ViT Conv2d | no-op | ❌ | **OFF** |

### zip_cache 说明

- 1877 clips × 4 camera × 8 workers：cache 会随见过的 clip 增长，**persistent_workers** 下跨 epoch 保留。
- Benchmark 显示：16-clip CoC 任务 zip+collate 收益显著；1877-clip no_coc 任务 **仅 DataLoader 并行已接近最优**，zip_cache 可能略拖慢或占 RAM。
- 全量 `chunks 0-99` 训练时建议监控 CPU RAM，必要时 `performance.zip_cache=false`。

---

## 2. 代码改动

```
alpamayo1_sft/
├── benchmark/
│   ├── apply_optimizations.py
│   ├── collate_cache.py      # 使用 alpamayo.processor.QwenProcessor
│   ├── perf_utils.py
│   ├── run_suite.py          # 支持 skip 已完成、--master_port
│   └── step_timer.py
├── bench_hf.py               # 50 step 定长 benchmark
├── train_hf.py               # 读取 performance.*
└── configs/sft_base.yaml     # DataLoader + performance 默认 ON
```

生产训练读取 `configs/sft_base.yaml`：

```yaml
trainer:
  dataloader_num_workers: 8
  dataloader_persistent_workers: true
  dataloader_prefetch_factor: 4

performance:
  zip_cache: true
  collate_cache: true
  tf32: true
  cudnn_benchmark: true
```

Benchmark 用 `+benchmark.optimizations.*=true/false` 覆盖（优先级高于 `performance`）。

---

## 3. 测试方法

| 项目 | 值 |
|------|-----|
| 硬件 | 8× H20-3e |
| 模型 | Alpamayo-R1-10B |
| 数据 | `/raid/charlie/pai_dataset`（19 chunks / CoC 16 clips） |
| 指标 | 50 steps，后 40 steps 平均 wall time |
| grad_accum | 4（与 Stage-1 一致） |

```bash
cd recipes/alpamayo1_sft && source a1_sft/bin/activate
python benchmark/run_suite.py   # 全矩阵
python benchmark/run_suite.py no_coc_production_opt coc_production_opt  # 单项
```

---

## 4. 实验结果（2026-06-10）

完整表格见 [`RESULTS.md`](RESULTS.md)。

### CoC disabled（1877 clips，19 chunks）

| 实验 | Avg Step | vs Baseline | 优化项 |
|------|---------:|------------:|--------|
| baseline | **19.02 s** | — | workers=2 |
| dl_workers8 | **10.29 s** | **+84.7%** | workers=8 |
| production_opt | 11.92 s | +59.5% | workers+zip+collate+tf32 |

**结论**：大 epoch 下 **DataLoader 并行是最大头**，单独启用即可接近最优；zip_cache 在本数据集上未进一步降 latency（可能因 cache 填充开销 / RAM 压力）。

### CoC enabled（16 reasoning clips）

| 实验 | Avg Step | vs Baseline | 优化项 |
|------|---------:|------------:|--------|
| baseline | **16.00 s** | — | workers=2 |
| dl_workers8 | 15.43 s | +3.7% | workers=8 |
| production_opt | **10.15 s** | **+57.6%** | workers+zip+collate+tf32 |

**结论**：小数据集、高重复访问时，**zip_cache + collate_cache 与 A1.5 一致，收益显著**。

---

## 5. 生产建议

| 场景 | 推荐配置 |
|------|----------|
| CoC / 小 clip 集合 | `performance` 全开（当前 `sft_base.yaml` 默认） |
| 全量 PAI、多 chunk | workers=8 + collate_cache + tf32；`zip_cache` 先开，OOM 则关 |
| 禁止 | preload_dataset_cache、改 grad_accum/grad_ckpt 做 benchmark |

---

## 6. 故障排查

| 现象 | 原因 | 处理 |
|------|------|------|
| `EADDRINUSE port 29500` | 上次 torchrun 未退出 | `pkill -9 -f alpamayo1_sft.bench_hf`，`run_suite.py` 已加 `--master_port=29501` |
| 进程看似卡住 | 50 step × ~12s ≈ 10min/实验；zip_cache 预热首几步慢 | 看 GPU 利用率；勿重复起 run |
| collate_cache TypeError | 误用 `alpamayo_r1` QwenProcessor | 已改为 `alpamayo.processor` |

---

## 7. 与 A1.5 对比

| | A1.5 nav (20 samples) | A1 no_coc (1877 clips) | A1 coc (16 clips) |
|--|----------------------|------------------------|-------------------|
| baseline | ~26.5 s | 19.0 s | 16.0 s |
| production_opt | ~12.0 s (+121%) | 11.9 s (+59%) | **10.2 s (+58%)** |
| 瓶颈 | DataLoader | ** mainly DataLoader** | DataLoader + collate + zip |

*CoC enabled 使用本地合成的 16-clip reasoning 数据（无 HF gated 数据），timing 可比。*
