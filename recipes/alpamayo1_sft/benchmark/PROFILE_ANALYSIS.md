# Alpamayo-1 SFT：Profile 与 DataLoader 分析

Updated: 2026-06-10  
硬件：8× H20，DeepSpeed ZeRO-2，配置 `grad_accum=4`  
数据来源：Nsight Systems（step 5–7）、50-step benchmark（`RESULTS.md`）、dataloader timing（step 5–11）

---

## 核心结论

| | no_coc（~1877 clips） | coc（16 reasoning clips） |
|--|----------------------|---------------------------|
| **瓶颈** | IO 并行不足 → `cudaStreamSynchronize` 等待 | PAI load + CPU preprocess → **step 开头 GPU idle** |
| **workers=8** | **+84.7%**（19.0s → 10.3s） | 仅 **+3.7%**（16.0s → 15.4s） |
| **最优 step** | ~11.9s（production_opt） | **5.5s**（load_cache）/ ~8.9s（preprocess_cache） |
| **GPU 利用率** | ~86%（算得多，sync 也多） | ~50%（profile 下 fetch 占半幅墙钟） |

**no_coc**：大 epoch，加 workers 能 overlap 读盘。  
**coc**：每 rank 每 epoch 仅 2 clips，无法喂饱 worker 池；Trainer **先 fetch 再 GPU**（无 overlap），瓶颈在数据准备而非 IO 并行。

---

## 1. CoC 单步时间结构

HF Trainer 执行顺序（无 fetch/compute overlap）：

```
get_batch_samples (fetch, GPU idle)  →  training_step × N (GPU busy)  →  optimizer
```

`coc_production_opt_preprocess` dataloader timing（跨 rank 平均）：

| 阶段 | 耗时 | 说明 |
|------|-----:|------|
| fetch（2× `next(dataloader)`） | ~3.2 s | 主线程阻塞 ≈ step 内 GPU idle |
| └ PAI load | ~2.5 s/样本 | `load_physical_aiavdataset` |
| └ preprocess | ~77 ms/样本 | `preprocess_cache` 命中后 |
| └ collate | ~68 ms | `collate_cache` 后可忽略 |
| train.gpu | ~5.8 s | 2× micro-batch FWD/BWD |
| **step.wall** | **~8.9 s** | fetch + gpu（串行） |

`load_cache` 将 fetch 压至 ≈0 后，step ≈ **5.5 s**（纯 GPU 下限）。因 `fetch(3.2s) < gpu(5.8s)`，跨 step prefetch 理论上也可达 ~5.8s；`load_cache` 额外消除 rank straggler。

### 有效 GA = 2（非配置中的 4）

CoC 仅 16 clips / 8 GPU → 每 rank 每 epoch **2 样本**。每 optimizer step 实际 **2 次** FWD/BWD（非 4 次）。`get_batch_samples` 在 `on_step_begin` 之前执行，step 开头 GPU idle 即整段 fetch 窗口。

---

## 2. Benchmark 关键数据

完整表格见 [`RESULTS.md`](RESULTS.md)。摘要（50 step，tail 40 平均）：

| Run | Avg Step | 要点 |
|-----|--------:|------|
| no_coc_baseline → workers8 | 19.0 → **10.3 s** | workers 是关键 |
| no_coc_production_opt | 11.9 s | zip 收益有限 |
| coc_baseline → workers8 | 16.0 → 15.4 s | workers 几乎无效 |
| coc_dl_workers8 → +collate | 15.4 → **10.2 s** | collate_cache **+50.6%** |
| coc_production_opt_preprocess | 8.9 s | preprocess_cache |
| **coc_production_opt_loadcache** | **5.5 s** | 当前最优 |

CoC 单因素 ablation（workers=8 基准 15.4s）：`collate` +50.6%，`zip` **-6.8%**（应关闭）。

---

## 3. Nsight Profile（production_opt，step 5–7）

### 复现

```bash
cd recipes/alpamayo1_sft && source a1_sft/bin/activate
python benchmark/run_profile.py no_coc_production_opt coc_production_opt
python benchmark/analyze_profile.py
```

产出：`benchmark/profiles/{no_coc,coc}_production_opt.nsys-rep`

### 墙钟与 GPU 计算

| | no_coc | coc |
|--|-------:|----:|
| 3-step 平均墙钟 | 11.2 s | 10.9 s |
| GPU kernel 总计（3 step × 8 GPU） | 230.6 s | 131.3 s |
| GEMM（nvjet+cutlass）占比 | ~80% | ~71% |
| FlashAttention 占比 | ~11% | ~10% |
| NCCL AllReduce 占比 | 5.6% | 15.5%（含 straggler） |
| 估算 GPU 利用率 | **86%** | **50%** |

两者 production_opt 墙钟均 ~11 s，但成因不同：no_coc GPU 算得满、被 sync 拖住；coc 计算量少一半、大量时间在等数据。

### CPU / 流水线等待（profile 核心差异）

| 指标 | no_coc | coc | 比值 |
|------|-------:|----:|-----:|
| `cudaStreamSynchronize` | 126 s | 66 s | 1.9× |
| H2D 数据量（3 step） | 11 GB | 5.5 GB | 2× |

no_coc 的 stream sync 等待约为 coc 的 2 倍，反映大 epoch 下 DataLoader 喂不饱 GPU。coc 在 production_opt 下 sync 较少，但 dataloader timing 显示 **Trainer 串行 fetch** 仍是主要墙钟来源（profile 与 timing 互补：profile 看 GPU 侧等待，timing 量化 fetch 子阶段）。

---

## 4. 优化建议

### CoC

| 优先级 | 优化 | 收益 |
|--------|------|------|
| P0 | `collate_cache` | +50.6% |
| P0 | `load_cache` | 8.9s → **5.5s** |
| P1 | `preprocess_cache` | 10.2s → 8.9s（被 load_cache 包含时可不开） |
| P1 | fetch/compute overlap | 理论 ~5.8s，需改 Trainer |
| 关闭 | `zip_cache` | CoC 负收益 |
| 可选 | workers 2~8 | 差异 <1% |

**推荐**：`collate_cache + load_cache + tf32`，`zip_cache=false`。

### no_coc

**推荐**：`workers=8 + collate_cache + tf32`；`zip_cache` 视 RAM 酌情开启。进一步可考虑 `prefetch_factor`、NUMA 绑核、本地 SSD 缓存。

---

## 5. 工具索引

| 用途 | 路径 |
|------|------|
| Benchmark 跑分 | `benchmark/run_suite.py` |
| 分阶段耗时 | `benchmark/run_dataloader_timing.py` |
| nsys 采集 / 分析 | `run_profile.py`, `analyze_profile.py` |
| Cache 实现 | `collate_cache.py`, `preprocess_cache.py`, `load_cache.py` |
| 优化说明 | `OPTIMIZATION.md` |
| 跑分数据表 | `RESULTS.md` |

**测量注意**：勿并行多个 `torchrun`；跑前 `python benchmark/run_suite.py --check-gpu-only`；残留进程 `pkill -9 -f 'alpamayo1_sft.bench_hf'`。

`bench_hf.py` 可选：`+benchmark.profile_start=5` `+benchmark.profile_end=7` 控制 nsys 捕获窗口。
