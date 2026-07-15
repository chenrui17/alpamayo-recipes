# Alpamayo 1.5 GRPO 吞吐优化实验报告

> 日期：2026-06-12  
> 环境：单节点 8× NVIDIA H20-3e (140GB)  
> 框架：Cosmos-RL + GRPO  
> 模型：Alpamayo 1.5 (`/raid/charlie/cc_run/alpamayo_1_5_converted`)  
> 实验目录：`/raid/charlie/cc_run/`  
> 分析脚本：`/raid/charlie/cc_run/analyze_run.py`、`analyze2.py`

---

## 1. 背景与目标

在 Alpamayo 1.5 VLA 的 GRPO 强化学习 post-training 中，单节点 8 GPU 稳态吞吐约 **2.5 samples/s**（~19s/step）。目标是提升系统吞吐，在不影响 RL 收敛的前提下缩短 wall-clock 训练时间。

**默认训练拓扑**：

| 组件 | GPU | 配置 |
|------|-----|------|
| Policy (FSDP) | 6 (GPU 0–5) | `dp_shard_size=6` |
| Rollout (vLLM) | 2 (GPU 6–7) | `n_init_replicas=1` × 2 |
| Global batch | — | `train_batch_per_replica=48` |
| Mini batch | — | `mini_batch=1`（48 次串行 forward/step） |

**启动命令**：

```bash
bash /raid/charlie/cc_run/train_run.sh <toml_basename> <tag> <policy_replicas> <rollout_replicas>
# 例：bash /raid/charlie/cc_run/train_run.sh exp_minib2.toml exp_minib2 1 2
```

---

## 2. Baseline 性能

| 运行 | 奖励模式 | 步数 | iter_time (稳态) | samples/s | rollout generate | pending 终值 |
|------|----------|------|------------------|-----------|----------------|--------------|
| `run_motion15` | motion-only | 60 | **18.62s** | **2.58** | ~12.9s / 276 tok/s | 1008 ↑ |
| `run_joint15` | joint (Lingo) | 60 | **18.66s** | **2.57** | ~13.5s / 263 tok/s | 960 ↑ |
| `run_exp_base` | motion (1.5) | 24 | **18.71s** | **2.56** | ~13.5s / 272 tok/s | 432 ↑ |

### 2.1 瓶颈诊断

```
Rollout (2 GPU, ~13s)  ──→  pending rollouts ↑↑  ──→  Policy (6 GPU, ~19s) ← BOTTLENECK
```

- **Policy step (~19s) 决定 wall-clock**，rollout 单次 generate 仅 ~13s
- `pending rollouts` 持续堆积 → rollout 比 policy 快，但 policy 才是限速环节
- Policy GPU 利用率 68–87%；Rollout GPU 仅 50–54%（资源分配不合理）
- 每 step 48 个 sample 因 `mini_batch=1` 串行 forward，GPU 算力利用不充分

---

## 3. 实验总览

### 3.1 配置/算子优化（`mini_batch=1` 不变）

| 实验 TOML | 关键改动 | 步数 | iter (稳态) | samples/s | 结论 |
|-----------|----------|------|-------------|-----------|------|
| `exp_base` | baseline | 24 | 18.71s | 2.56 | 参照 |
| `exp_noreshard` | `fsdp_reshard_after_forward=never` | 10 | **18.19s** | **2.64** | 略好 (~3%) |
| `exp_sync` | `sync_weight_interval` 2→8 | 8 | 18.60s | 2.58 | 无收益 |
| `exp_nogc` | `model_gradient_checkpointing=false` | 10 | 18.54s | 2.59 | 无收益（短跑） |
| `exp_prefetch` | `capacity=96`, `num_workers=16` | 8 | 18.60s | 2.58 | 无收益 |
| `exp_compile` | `compile=true` (LM + visual) | 18 | 18.65s | 2.57 | 无收益 |
| `exp_fp8only` | Policy FP8 (`train.fp8.enable_fp8`) | 8 | **21.21s** | **2.26** | ❌ 慢 15% |
| `exp_fp8tw` | FP8 train + rollout quant | — | — | — | 未完成 |
| `exp_batch96` | `train_batch_per_replica=96` | 5 | **36.90s** | **1.30** | ❌ iter 翻倍 |
| `exp_best` | fp8 rollout + batch56 + sync8 | 0 | — | — | ❌ 启动失败 |
| `run_dp4` | 4 GPU policy | 0 | — | — | ❌ 未完成 |

### 3.2 方向 1：Policy `mini_batch` 批处理

| 实验 TOML | 配置 | 步数 | iter (稳态) | samples/s | 结论 |
|-----------|------|------|-------------|-----------|------|
| baseline (`exp_base`) | mb=1, 6p+2r, batch=48 | 24 | 18.71s | 2.56 | — |
| **`exp_minib2`** | **mb=2**, 6p+2r, batch=48 | **24** | **17.58s** | **2.73** | ✅ **+6.6%** |
| **`exp_minib2_noreshard`** | **mb=2 + reshard=never**, 6p+2r | **24** | **17.27s** | **2.78** | ✅ **+8.6%**（当前最优） |
| `exp_minib4` | mb=4, 6p+2r, batch=48 | 0 | — | — | ❌ CUDA OOM |
| `exp_minib4_7p1r` | mb=4, 7p+1r, batch=56 | 0 | — | — | ❌ CUDA OOM |
| `exp_minib8` | mb=8, 6p+2r, batch=48 | — | — | — | 未跑 |

**Cosmos-RL 整除约束**：

```
train_batch_per_replica % (dp_shard_size × mini_batch) == 0
```

| dp_shard | mini_batch | 所需 batch 倍数 | batch=48 | batch=56 |
|----------|------------|-----------------|----------|----------|
| 6 | 1 | 6 | ✅ | — |
| 6 | 2 | 12 | ✅ | — |
| 6 | 4 | 24 | ✅ (但 OOM) | — |
| 7 | 4 | 28 | ❌ | ✅ (但 OOM) |

---

## 4. 最佳结果详情

### 4.1 `exp_minib2_noreshard`（当前最优）

**配置**：`toml/exp_minib2_noreshard.toml` — `mini_batch=2` + `fsdp_reshard_after_forward=never`, 6p+2r, batch=48  
**日志**：`/raid/charlie/cc_run/run_exp_minib2_noreshard/`

| 指标 | baseline (mb=1) | minib2 (mb=2) | minib2+noreshard | vs baseline |
|------|-----------------|---------------|------------------|-------------|
| iter_time 稳态 | 18.71s | 17.58s | **17.27s** | **-7.7%** |
| samples/s | 2.56 | 2.73 | **2.78** | **+8.6%** |
| iter_time median | ~18.6s | 17.43s | **17.15s** | — |
| pending 终值 | 432 | 432 | 432 | 持平 |
| rollout generate | ~13.5s | ~13.3s | **13.30s** | 持平 |
| RL traj_L2 (首3/末3) | 2.06 / — | 2.06 / 1.87 | 2.06 / **1.87** | 正常收敛 |
| 24 steps wall clock | ~11.6 min | ~11.9 min | **~11.5 min** | — |

**叠加效果**：单独 `noreshard` 约 +3%，单独 `mb=2` 约 +6.6%，组合后相对 baseline **+8.6%**，两项优化近似可加。

### 4.2 `exp_minib2`（仅 mini_batch=2）

**配置**：`toml/exp_minib2.toml` — `mini_batch=2`, 6p+2r, batch=48  
**日志**：`/raid/charlie/cc_run/run_exp_minib2/`

| 指标 | baseline (mb=1) | minib2 (mb=2) | 变化 |
|------|-----------------|---------------|------|
| iter_time 稳态 | 18.71s | **17.58s** | **-6.0%** |
| samples/s | 2.56 | **2.73** | **+6.6%** |
| iter_time median | ~18.6s | **17.43s** | — |
| pending 终值 | 432 | 432 | 持平 |
| RL traj_L2 (首3/末3) | 2.06 / — | 2.06 / 1.87 | 正常收敛 |

**为何收益有限**：`mini_batch` 只减少 forward 次数（48→24），但以下开销不随 mini_batch 缩小：
- 48 次串行 `get_policy_input`（data pack）
- FSDP all-reduce / grad sync
- Policy→Rollout weight sync

---

## 5. 代码改动

### 5.1 已合入 recipe 的改动

| 文件 | 改动说明 |
|------|----------|
| `models/reasoning_vla/data_packer.py` | 新增 `_pad_row_to_len()`；`policy_collate_fn` 对 `input_ids`/`labels_mask`/`attention_mask`/`position_ids` 做左 padding，支持 `mini_batch>1`；`pixel_values`/`image_grid_thw` 不做 padding |
| `utils/fsdp.py` | `maybe_compile_lm_layers()`、`maybe_fp8_convert_lm()`、`fsdp_reshard_after_forward` 可配置 |
| `models/reasoning_vla/cosmos_wrapper.py` | 接入 compile / fp8 优化 hook |
| `models/reasoning_vla/trainer.py` | CUDA Event 计时（`train/iteration_time`） |
| `models/reasoning_vla/rollout.py` | vLLM 环境变量 override、`rollout_profile_s` 分段日志 |

### 5.2 新增实验 TOML（`toml/`）

```
exp_base.toml          # baseline
exp_noreshard.toml     # FSDP reshard=never
exp_sync.toml          # sync_weight_interval=8
exp_nogc.toml          # 关 gradient checkpointing
exp_compile.toml       # torch.compile
exp_fp8only.toml       # Policy FP8
exp_fp8tw.toml         # FP8 train + rollout
exp_prefetch.toml      # prefetch 调参
exp_batch96.toml       # batch=96
exp_best.toml          # 组合最优（未跑通）
exp_minib2.toml        # mini_batch=2
exp_minib2_noreshard.toml  # mb=2 + reshard=never ✅ 推荐
exp_minib4.toml        # mini_batch=4
exp_minib8.toml        # mini_batch=8
exp_minib4_7p1r.toml   # mb=4, 7p+1r, batch=56
```

---

## 6. 失败模式与排障

| 错误 | 场景 | 原因 | 处理 |
|------|------|------|------|
| `Sizes of tensors must match` | mb=4 首次运行 | collate 未 padding 不同长度序列 | 已修复 `data_packer.py` |
| `CUDA out of memory` | mb=4 | VLA 多模态 batch forward 超 140GB | 降至 mb=2 或减 batch |
| `Free memory < gpu_util` | rollout GPU 6/7 | policy 占显存后 rollout 起不来 | 7p+1r 或降 `gpu_memory_utilization` |
| `NCCL timeout` | 多实验并行 | 僵尸 GPU 进程 | `pkill -9 -f cosmos-rl` 后确认 `nvidia-smi` 全 0 |
| `batch % (dp×mb) != 0` | 7p1r + batch=48 | 48 不能被 28 整除 | batch 改为 56 |
| `exit 137 (SIGKILL)` | 后台任务 | pkill 误杀 / OOM killer | 训练与 kill 分开执行 |

**跑实验前检查**：

```bash
pkill -9 -f "cosmos-rl|torchrun" 2>/dev/null
sleep 8
nvidia-smi --query-gpu=index,memory.used --format=csv   # 应全为 0 MiB
```

---

## 7. 结论

### 7.1 已验证有效

1. **`mini_batch=2` + `fsdp_reshard_after_forward=never`**（`exp_minib2_noreshard.toml`）— 吞吐 **+8.6%**（2.78 samples/s），RL 收敛正常，**当前推荐配置**
2. **`mini_batch=2` 单独**（`exp_minib2.toml`）— 吞吐 **+6.6%**，可作为不改动 FSDP reshard 时的备选
3. **`fsdp_reshard_after_forward=never` 单独**（`exp_noreshard.toml`）— 约 +3%，建议与 mb=2 叠加使用

### 7.2 已验证无效或负收益

- `torch.compile`（LM + visual）
- Policy FP8 训练（`train.fp8.enable_fp8`）
- 单独增大 `train_batch_per_replica`（96）而不改 mini_batch
- Prefetch capacity/workers 调参
- `sync_weight_interval` 2→8

### 7.3 显存硬限制

- **`mini_batch=4` 在 6p 或 7p 布局下均 OOM**（单卡 ~140GB 用尽）
- 理论 forward 次数减半（48→12）的收益被显存墙阻断

### 7.4 单节点物理上限

在 8× H20 单节点、不损失 RL 质量的前提下，合理上限约 **2.7–3.0 samples/s**。要数量级提升需多节点水平扩展。

---

## 8. 后续建议（按优先级）

| 优先级 | 方向 | 预期收益 | 说明 |
|--------|------|----------|------|
| P0 | `exp_minib2_noreshard` 60 steps 长跑 | 稳定性验证 | 24 steps 已验证 +8.6%，待确认长跑无退化 |
| P1 | Visual encode 缓存 | 可能很大 | rollout 已 encode，policy 可能重复算 ViT |
| P1 | 并行化 `get_policy_input` | 中等 | 48 次串行 data pack 是固定开销 |
| P2 | 7p+1r + 关 GC 试 mb=3 | 未知 | 需重新测 OOM 边界 |
| P3 | 多节点 scale-out | 线性 | `data_dispatch_as_rank_in_mesh` + 多 policy replica |

---

## 9. 复现命令

```bash
# 环境
source /raid/charlie/cc_run/env.sh
cd /raid/charlie/alpamayo/alpamayo-recipes
source recipes/alpamayo1_x_rl/a1x_rl/bin/activate

# 当前最优：mini_batch=2 + reshard=never
bash /raid/charlie/cc_run/train_run.sh exp_minib2_noreshard.toml exp_minib2_noreshard 1 2

# 分析结果
python3 /raid/charlie/cc_run/analyze_run.py /raid/charlie/cc_run/run_exp_minib2_noreshard

# 与 baseline / 单项优化对比
python3 /raid/charlie/cc_run/analyze_run.py /raid/charlie/cc_run/run_exp_base
python3 /raid/charlie/cc_run/analyze_run.py /raid/charlie/cc_run/run_exp_minib2
python3 /raid/charlie/cc_run/analyze_run.py /raid/charlie/cc_run/run_exp_noreshard
```

---

## 10. 实验日志索引

| Tag | 路径 | 状态 |
|-----|------|------|
| motion15 baseline | `/raid/charlie/cc_run/run_motion15/` | ✅ 60 steps |
| joint15 baseline | `/raid/charlie/cc_run/run_joint15/` | ✅ 60 steps |
| exp_base | `/raid/charlie/cc_run/run_exp_base/` | ✅ 24 steps |
| exp_minib2 | `/raid/charlie/cc_run/run_exp_minib2/` | ✅ 24 steps |
| **exp_minib2_noreshard** | `/raid/charlie/cc_run/run_exp_minib2_noreshard/` | ✅ **24 steps（当前最优）** |
| exp_noreshard | `/raid/charlie/cc_run/run_exp_noreshard/` | ✅ 10 steps |
| exp_compile | `/raid/charlie/cc_run/run_exp_compile/` | ✅ 18 steps |
| exp_fp8only | `/raid/charlie/cc_run/run_exp_fp8only/` | ✅ 8 steps（慢） |
| exp_batch96 | `/raid/charlie/cc_run/run_exp_batch96/` | ⚠️ 5 steps（慢） |
| exp_minib4_* | `/raid/charlie/cc_run/run_exp_minib4*/` | ❌ OOM/timeout |

---

*本报告由 Claude Code 辅助生成，基于 `/raid/charlie/cc_run/` 下实际运行日志汇总。*
