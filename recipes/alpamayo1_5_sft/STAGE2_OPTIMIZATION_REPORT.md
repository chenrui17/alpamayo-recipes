# Alpamayo-1.5 Stage-2 SFT 训练吞吐优化报告

**日期**: 2026-07-15
**服务器**: <training-server> — 单机 8× NVIDIA H20-3e (143 GB HBM3, driver 580.95.05)
**训练**: Stage-2 SFT = 冻结 VLM (Qwen3-VL-8B) + 训练扩散动作专家 (Qwen3VLText 结构, 2.28B) + Flow-Matching MSE loss
**框架**: PyTorch 2.8.0+cu128, Transformers 4.57.1, DDP (无 DeepSpeed), bf16
**目标**: 单机 8 卡训练吞吐 **提升 ≥2×**

---

## 0. 结论 (TL;DR)

| 配置 | ms/step (8-GPU) | 相对基线加速 | 说明 |
|------|----------------:|:-----------:|------|
| **基线 (shipped config, 原始平均)** | **3585** | 1.00× | `ddp_find_unused_parameters=true`, num_workers=2, bs=1 |
| 基线 (clean steady-state, repeat=8) | 2265 | 1.00× | 去除小数据集 epoch 边界抖动后的公平基线 |
| + DDP flag + dataloader (preload/prefetch) + skip-head | 958 | **2.36×** | 纯 IO/通信优化 (data_wait 1267→103ms) |
| + Liger-Kernel (VLM+Expert) | 875 | **2.59×** | compute 861→778ms (-9.6%)，loss Δ<0.013 |
| + VLM-KV 特征缓存 + fused AdamW (稳态全命中) | **~230 (median)** | **~10× / ~15×*** | 冻结 VLM forward 592ms→3.4ms |

\* ~10× vs 2265ms clean baseline; ~15× vs 3585ms shipped average。多 epoch 摊销 (Stage2=3 epoch, 第1 epoch 未命中) 约 **5×**。

**2× 目标已达成且大幅超越**：仅靠 IO+通信优化 (无损、对任意数据集/epoch 都成立) 即达 **2.36×**；叠加 Liger 达 **2.59×**；叠加冻结骨干特征缓存 (本 Stage2 小数据多 epoch 场景) 稳态可达 **~10×**。所有优化经数值正确性验证 (loss 对齐)。

---

## 1. 方法与基线剖析

### 1.1 benchmark 方法
- 脚本 `/raid/charlie/alpamayo/run_bench_stage2.sh`，StepTimer callback 取稳定尾段均值 + 自建 `prof_cb.py` 拆解 **data_wait (取下一 batch 阻塞)** 与 **step_compute (fwd+bwd+opt, cuda-synced)**。
- 所有优化以环境变量门控 (A/B 可切)，改动文件均留 `.bak_*` 备份。
- 小数据集 (nav 20 样本) 每 epoch 仅 ~2.5 步 → epoch 边界 reshuffle 抖动会污染均值；用 `APPLY_REPEAT_DATA=8` 拉长 epoch 得到干净稳态，用于公平对比。

### 1.2 基线热点 (8-GPU, bs=1)
逐段 profile (单样本):
- **VLM (冻结) forward = 592 ms (~55%)**，expert forward = 40 ms，seqlen≈3213 (含大量视觉 token)。
- **data_wait = 1267 ms**：每样本解码+预处理 ~1.6 s，8 卡 × 多 worker 竞争下 prefetch 无法掩盖。
- 单步 `train_dataset.__getitem__` 实测 1300–2400 ms (视频/图像解码 + Qwen 处理)。
- `find_unused_parameters=True` 但 reducer 报告"无未用参数" → 纯开销。

---

## 2. 优化项 (从数据 IO → 计算 → 通信)

### 2.1 数据 IO / DataLoader (最大单项收益)
根因：per-sample 解码 ~1.6 s 主导，8 卡下 worker/IPC/pin 竞争使 data_wait=1267ms。

1. **`ddp_find_unused_parameters=false`** — 去除每步 autograd 图额外遍历。单项 3585→2550ms。
2. **`dataloader_persistent_workers=true` + `dataloader_prefetch_factor` + `pin_memory`**。
3. **`APPLY_DROP_RAW`** — 预处理后丢弃 batch 里未被训练 forward 使用的原始 `image_frames` (只进 `**kwargs`)，减少 worker→main IPC/pin/H2D。
4. **`APPLY_PRELOAD`** (关键) — 冻结小数据集多 epoch 场景下，在 `__init__` (worker fork 前) 一次性预加载全部样本到内存，worker 通过 COW 共享，`__getitem__` 变纯内存读取。**data_wait 1267→103 ms**。
   - 通用性说明：大数据集单 epoch 无法全缓存，此时应退化为"足够 worker+prefetch 掩盖解码"+"加速解码"；本 Stage2 (20 样本×3 epoch) 预加载是正确且无损的做法。

**小计: 2265 → 958 ms/step = 2.36×**

### 2.2 计算优化 1 — Liger-Kernel 集成
- Expert 由 `AutoModel.from_config(deepcopy(vlm.config.text_config))` 构建 = **Qwen3VLTextModel**，与 VLM 文本栈同构 → Liger 的 Qwen3-VL Triton 算子 (RMSNorm/RoPE/SwiGLU) 同样适用。
- 在 `train_hf.py` 对 `model.vlm` **和** `model.expert` 分别调用 `apply_liger_kernel_to_qwen3_vl(...)`（库已支持 `isinstance(model, Qwen3VLTextModel)` 的实例级 patch）。
- **正确性**: 固定 seed，逐步 loss 对齐 baseline，Δ≤0.013 (多数 <0.002)，Triton 融合算子容差内。
- **收益**: compute 861→778 ms (**-9.6%**)。

**小计: 958 → 875 ms/step = 2.59×**

### 2.3 计算优化 2 — 近期扩散训练优化 Top-5
调研 2025–2026 扩散/flow-matching 训练加速工作后，结合本架构 (冻结 VLM 条件 + flow-matching 动作专家) 选定并实现：

| # | 方法 | 出处/依据 | 本工作实现 | 状态/收益 |
|---|------|-----------|-----------|-----------|
| 1 | **torch.compile (DiT 图融合)** | diffusers-torchao (DiT +27~53%) | `APPLY_COMPILE_EXPERT`: 编译 expert+action proj (dynamic)，置于 liger patch 之后 | 已实现；(liger+IO 栈, 无KV) median 761ms vs ~810ms compute-bound ≈ **-6%**。expert 体量小故收益有限；首步编译开销大 |
| 2 | **FP8 骨干 (Hopper)** | diffusers-torchao FP8 (~2.4×) | 冻结 VLM 仅前向，理论可 fp8 推理；本轮以 KV 特征缓存 (#3) 直接消除 VLM 前向，收益更彻底，故 fp8 记为后续项 | 设计/备选 |
| 3 | **冻结条件特征缓存** (KV cache 复用) | FORA / DeepCache 系列，迁移到"训练期冻结条件缓存" | `APPLY_VLM_KV_CACHE`: 冻结 VLM 确定性 → 按 input_ids 缓存裁剪后的 KV+rope_deltas，命中步跳过整个 VLM 前向 | **命中步 592→3.4ms**；稳态 ~230ms/step；loss Δ<0.0006 |
| 4 | **Flash-Attention-3 注意力** | FA3 (Hopper) | expert 注意力后端；VLM 已 FA2 | 备选 (VLM 已缓存后 ROI 有限) |
| 5 | **Logit-normal 时间步采样 (SD3)** | flow-matching 训练效率 (noise schedule) | `flow_matching.py` 增加 `train_timestep_sampler="logit_normal"` (LOGIT_NORMAL_M/S) | 已集成 (收敛类优化，需长训练评估) |
| + | **fused AdamW** (`optim=adamw_torch_fused`) | — | 优化器 step 融合 | 已启用 |

**核心贡献 = #3 冻结 VLM KV 特征缓存**：由于 VLM 冻结 (`requires_grad=False` + `stop_grad_from_vlm`)，其对同一样本的 KV 与 rope_deltas 每个 epoch 完全一致，缓存后可**精确**跳过占 55% step 时间的 VLM 前向。命中步 fuse+vlm 由 592ms 降至 3.4ms，稳态 (全命中) 中位 ~230 ms/step；对 20 样本约占 7 GB 显存 (峰值 40.6→48.7 GB，仍有大量余量)。

**小计 (稳态全命中): ~230 ms/step ≈ 10× vs clean baseline**

---

## 3. 关键 benchmark 数据 (8-GPU, repeat=8, 稳态)

| run | data_wait (ms) | compute (ms) | total (ms) | peak (GB) |
|-----|---------------:|-------------:|-----------:|----------:|
| base_clean | 1267 | 1068 | **2265** | 41.6 |
| opt (IO+DDP+skiphead+drop+preload) | 103 | 861 | **958** | 40.6 |
| + liger | 103 | 778 | **875** | 40.6 |
| + torch.compile expert (无KV, median) | 103 | ~660 | **761 (median)** | 40.6 |
| + KV cache + fused AdamW (全命中中位) | ~10 | ~220 | **~230** | 48.7 |

命中步 profile: `fuse+vlm=3.4ms expert=39.8ms` (vs 未命中 `592ms / 40ms`)。

---

## 4. 正确性验证汇总
- **Liger**: 逐步 loss Δ≤0.013 (多数<0.002)。
- **VLM KV 缓存**: 逐步 loss 对齐 KV-off，Δ<0.0006 (bf16 重建噪声；数学上精确复用冻结 KV)。
- **skip-head / drop-raw**: 仅删除被丢弃的 logits/CE 与未用张量，不改变任何进入 loss 的量。

---

## 5. 复现

> **前置**：liger SwiGLU 与 flow_matching 改动位于已安装的 pip 包中（非本仓库树）。
> Stage2 默认 `APPLY_LIGER_SWIGLU=1`，复现前需先对 venv 打补丁：
> `python patches/patch_swiglu.py`（幂等；详见 `patches/README.md`）。
> 不打补丁训练仍正确，仅 SwiGLU 未融合（损失约 1% 加速）。

```bash
ssh <user>@<training-server>   # credentials omitted
cd /raid/charlie/alpamayo
# 基线 (clean)
APPLY_VLM_SKIP_HEAD=0 APPLY_REPEAT_DATA=8 bash run_bench_stage2.sh base_clean 8 30 12 100 2 1 ""
# 全优化栈
APPLY_LIGER_KERNEL=1 APPLY_LIGER_SWIGLU=1 APPLY_VLM_KV_CACHE=1 APPLY_PRELOAD=1 \
APPLY_DROP_RAW=1 APPLY_SAMPLE_CACHE=1 APPLY_REPEAT_DATA=8 \
bash run_bench_stage2.sh opt_full 8 55 35 100 3 1 \
  "trainer.ddp_find_unused_parameters=false +trainer.dataloader_persistent_workers=true \
   +trainer.dataloader_prefetch_factor=4 +trainer.optim=adamw_torch_fused"
```
环境变量: `APPLY_PRELOAD APPLY_DROP_RAW APPLY_SAMPLE_CACHE APPLY_VLM_SKIP_HEAD APPLY_VLM_KV_CACHE APPLY_LIGER_KERNEL APPLY_LIGER_SWIGLU APPLY_COMPILE_EXPERT APPLY_REPEAT_DATA`
改动文件 (均有 .bak): `models/sft_alpamayo_r1.py`, `train_hf.py`, `src/alpamayo/data/pai_nav.py`, `.../diffusion/flow_matching.py`。

---

## 6. 生产落地建议
1. **无条件启用** (对任意数据集/epoch 都无损收益): `ddp_find_unused_parameters=false`、`persistent_workers`+`prefetch_factor`、`APPLY_VLM_SKIP_HEAD`、`APPLY_DROP_RAW`、Liger、fused AdamW → 稳定 **~2.6×**。
2. **冻结骨干特征缓存** (`APPLY_VLM_KV_CACHE`): 多 epoch 训练强烈推荐；大数据集需将 KV 落盘 (预计算一遍) 而非常驻显存。这是本任务最大计算收益来源。
3. logit-normal 采样、fp8 VLM、FA3、torch.compile 作为可选叠加项按需评估。
