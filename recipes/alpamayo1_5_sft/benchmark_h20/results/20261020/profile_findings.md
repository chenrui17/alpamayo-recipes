# Stage2 H20：steps 5–8 profile 归因

审查日期：2026-10-20，Asia/Shanghai。仅在运行 manifest 确认 `status=complete`、`returncode=0` 后导出 SQLite 并分析；后处理没有执行 GPU 工作或修改依赖。最终组合的 manifest 于 09:43:07 UTC 完成，导出于 09:43:13 UTC 完成，严格分析于 09:43:32 UTC 完成。

证据来自 H20-GPU-04：

- `/raid/charlie/alpamayo_training/results/latest_io_profile_r1/nsight.sqlite`
- `/raid/charlie/alpamayo_training/results/latest_combined_profile_r1/nsight.sqlite`
- 两个目录各自的 `analysis.json` / `analysis_csv/`；均为 `capture_validation.valid=true`，8 个 worker process、8 个设备，各 process 的 steps 5、6、7、8 均有带 start/end 的 NVTX 标签和 GPU kernel，共 32 个 process-step。**原 profile 的 `clipped_kernel_count=0`，最终组合为 2；最终 trace 不能描述为完全无边界裁切。**
- 逐步同步复核结果归档为 [profile/comparison.json](profile/comparison.json)。原始 `.nsys-rep` / SQLite 保留在远端运行目录；`.nsys-rep` 另已下载到本地任务产物，不将大文件提交 Git。
- Nsight Systems 2025.3.2；Transformers 4.57.1。以下相对事件时间来自该 trace，单位换算为 ms；它们不是墙上时钟时间。

这些 profile 数字只用于定位问题。吞吐/加速比使用独立非 profile 的 steps 5–20、逐步 rank 最大耗时；不能把跨 8 GPU 的 kernel 总和当作单步 critical path。

## 最终组合的实际变化

这里比较原 `latest_io_profile_r1` 与最终 `latest_combined_profile_r1`。两臂都使用 frozen VLM、BF16、8 GPU、每卡 batch 1、accumulation 1、同一 checkpoint 与 nav 配置、repeat 8、skip-head、preload/drop-raw、fused AdamW，以及相同 TF32/cuDNN 设置。最终组合另外启用 **Liger + prefix trim + 固定轨迹因子缓存 + DDP bucket 256 MiB**。剔除各 run 的输出路径后，`compose_command` 的唯一新增 Hydra override 是 `trainer.ddp_bucket_cap_mb=256`；其余三项由环境开关启用。原 profile 没有显式设置 bucket 大小。

这是四项变化的组合对照，不能把整体 GPU 时间差孤立归于任一新候选。相同配置与种子也不代替每步实际 clip 顺序的逐项记录；完整样本输入等价性与数值正确性由独立检查提供。

下表的每步均值以 **8 进程 × 4 步 = 32** 为分母。GPU 时间是 kernel duration sum 的均值，host 时间是 CUDA API duration sum 的均值；两者都允许跨 stream/进程重叠。

| 实测指标 | 原 IO/skip-head profile | 最终组合 profile |
|---|---:|---:|
| `cudaStreamSynchronize` 总次数 | 18,432 | 13,664 |
| 每进程每步 `cudaStreamSynchronize` 次数 | 576 | 427 |
| scalar crop 区间同步总次数 | 4,608 | **0** |
| 每进程每步 scalar crop 区间同步次数 | **144，32/32 个样本均如此** | **0，32/32 个样本均如此** |
| scalar crop CPU 区间 elapsed 均值（ms） | 3.733 | 0.137 |
| NCCL kernel 总次数 | 3,632 | 656 |
| 每设备每步 NCCL kernel 次数 | 113.5 | 20.5 |
| 每设备每步 NCCL GPU kernel sum（ms） | 118.313 | 70.591 |
| language 前向 GPU kernel sum（ms） | 418.565 | 366.646 |
| visual 前向 GPU kernel sum（ms） | 147.501 | 133.967 |
| expert 前向 GPU kernel sum（ms） | 19.844 | 16.762 |
| trajectory GPU kernel sum（ms） | 0.470 | 0.374 |
| 未归属到前向 phase 的 GPU kernel sum（ms） | 264.467 | 209.196 |

最终组合的两枚边界 kernel 均为 FusedAdamW `multi_tensor_apply_kernel`：PID 31292 / device 1 超出所选 NVTX 窗口 **47,741 ns**，PID 31298 / device 7 超出 **178,357 ns**，合计 **0.226098 ms**，按 32 个 process-step 平均为 **0.007066 ms**。它们不是 NCCL 或 crop kernel；上面的 GPU duration 已由分析器按窗口截断。`strict-window` 的 `valid=true` 验证标签/进程/步骤/GPU 事件覆盖，不能据此推断每枚 kernel 都完整位于窗口内。完整 kernel 名称及起止时间见 [profile/boundary_kernels.json](profile/boundary_kernels.json)。

crop 区间严格取同线程、同一步内 `action_in_proj` range 结束到 `_process_position_ids_qwen2_5_vl` range 开始；仅计完整位于区间内的 runtime `cudaStreamSynchronize`。旧 trace 的每个 process-step 都有 144 次，新 trace 全部为 0，实测支持复用 Python int 边界消除了这一段重复 scalar 转换。区间 elapsed 的差也只是该 CPU 区间的观测，不能直接当成训练净收益。

**同步减少没有带来可直接相加的 host 等待时间下降。** 全部 `cudaStreamSynchronize` host API sum 均值实际从 **259.164 增至 344.508 ms/进程步**。按包含整个调用的最内层 NVTX range 或明确间隙复核：

| 同步位置 | 原次数/进程步 | 最终次数/进程步 | 原 host API sum（ms/进程步） | 最终 host API sum（ms/进程步） |
|---|---:|---:|---:|---:|
| VLM 结束至 future-training 开始 | 1 | 0 | 197.884 | 0 |
| scalar crop 区间 | 144 | 0 | 1.061 | 0 |
| `_process_traj_future_training` 内 | 24 | 15 | 0.159 | 303.091 |
| `vlm_visual` 内 | 311 | 311 | 5.438 | 5.451 |
| `vlm_language_model` 内 | 7 | 7 | 31.822 | 27.554 |

这与等待先前 VLM 工作的同步位置转移相符，也受到组合变化对 CPU 发起/GPU 执行重叠的影响。不能把原先的 197.811 ms 单次等待写成省下的训练时间，也不能因新的 trajectory host 等待增加而把它全部归成轨迹计算变慢。GPU phase 时间通过 process-qualified CUDA correlation ID 关联 CPU launch，再取最内层 phase；当前 hooks 只覆盖前向，expert backward、optimizer 和通信的未归属部分仍在 `unattributed` 中。NCCL kernel 次数也不等于 DDP bucket 数，GPU sum 不是可全部回收的通信 critical path。

最终运行的实际开关有额外落盘证据：`stage2_runtime.json` 记录 Liger 在 VLM 和 expert 各命中 145 个 RMSNorm、36 个 SwiGLU，RoPE/vision RoPE 生效，compile/KV-cache 关闭。`stage2_final.json` 记录 prefix **10/10 次 applied、无 fallback、共裁 1,310 tokens**，最后一次 **3,212 → 3,081**；continuous action-space 的 theta/accel 各为 **miss 1 / hit 9**，2 个 entries、65,792 bytes，fallback/build-failure/eviction 全为 0。20 步独立 `latest_factor_r2` 则各为 **miss 1 / hit 19**。未被当前训练路径调用的 future tokenizer action-space 计数为 0，不将其误报为缓存命中。

`latest_factor_r1` 保留为诊断：其每 role miss 20、hit 0、合计 eviction 32，DDP 广播递增无关 affine buffer 的 `_version` 导致过度失效；最终版本已从 key 删除这些与固定矩阵无关的元数据，归一化仍实时读取原 buffer。不能把这个旧零命中运行作为成功测到缓存复用的证据。

## 原始 profile：StreamSynchronize 的真实位置

以 process PID **8779**（device 0）为具体复核样本。4 步共有 **2304 次** `cudaStreamSynchronize`，即 576 次/步；host API duration 总和 **1075.907 ms / 4 步**。按包含整个 API 调用的最内层 NVTX range 归因：

| 最内 NVTX range | 4 步调用数 | 4 步 host API sum（ms） |
|---|---:|---:|
| `optimizer_step_5..8` 外层 | 588 | 887.928 |
| `vlm_visual` | 1244 | 21.751 |
| `vlm_language_model` | 28 | 128.072 |
| `vlm`（不在上述子阶段内） | 292 | 3.928 |
| `_process_traj_future_training` | 96 | 0.650 |
| `fuse_traj_tokens` | 8 | 0.057 |
| 不在这些 range 内 | 48 | 33.522 |

**147 次/步的外层同步不是 DDP bucket 数量。** Step 5 的准确拆分如下：

| 代码位置/事件区间 | 次数 | host sync sum（ms） | 解释 |
|---|---:|---:|---|
| VLM CPU range 结束 489.047 ms，到 future-training 开始 687.097 ms | 1 | **197.811** | 对应 forward 的 `(input_ids == future_start).nonzero()` 动态输出形状读取；API 时间为 489.137–686.947 ms。 |
| `action_in_proj` 结束 695.385 ms，到 position-ids 开始 698.979 ms | **144** | **1.067** | GPU scalar crop 长度进入 36 层 DynamicLayer；每层条件和两个切片反复转换同一 scalar。这个区段的 CPU elapsed 为 3.595 ms。 |
| expert/action-out 前向结束后，到 step-end | 2 | 34.860 | 发生于后续训练阶段；当前 trace 没有 CUDA API callchain，未进一步宣称具体 Python 调用。 |

`nonzero()` 之后才出现下一次 DeviceSelect kernel，时间位置与源码吻合。**197.811 ms 包含等待先前已排队的 VLM GPU 工作，不能作为“删除 nonzero 后必然节省 197.811 ms”的估计。** 提前准备边界元数据可以减少不必要的主线程等待和重复 D2H，但也可能把等待移到后续其他同步点，收益必须重新测量。

安装版 `cache_utils.py:140–152` 的 DynamicLayer.crop 对 `max_length` 执行两个 Python 条件及 keys/values 两次切片；`:834–837` 将同一值交给每层。36 × 4 = 144 次 scalar 转换与 trace 数量完全对应。

## 视觉 grid 元数据：确有重复同步，但不是 0.27 秒的全部来源

安装版 `modeling_qwen3_vl.py` 的相关位置：

- `603–640 / rot_pos_emb`：`max().item()`、`prod().sum().item()`，循环中的 CUDA grid scalar 进入 arange/expand/repeat 的 Python shape/条件。
- `642–701 / fast_pos_embed_interpolate`：grid scalar 进入 linspace、split、repeat、view；位置索引/插值权重从几何元数据重复构建。
- `205–219 / Qwen3VLVisionAttention`：FA2 分支每层重新计算 `max_seqlen`，并把同一 CUDA scalar 作为 `max_length_q/k` 传给需要 Python int 的 FA 接口。
- `224–227`：非 FA2 分支对 query/key/value 分别调用相同 `lengths.tolist()`，每层可能重复 3 次 D2H。
- `727–735`：cu_seqlens 从 grid 重建；FA2 所需累计长度 tensor 仍须保留在 GPU，不能直接改成 CPU tensor。
- `916–1013 / get_rope_index`：multimodal input `.tolist()`、image/video 计数、各 grid t/h/w `.item()`；`1062` 另有 split-sizes `.tolist()`。

这些分支均来自实际安装源码；不能仅凭 checkpoint 的 attention 字符串断言本次运行命中了哪个分支。该 process 的 visual 同步实际为 **311 次/步，host sum 约 5.438 ms/步**，不能把全部 576 次同步的约 269 ms/步归为视觉几何元数据成本。5.438 ms 也包含依赖等待，并非可承诺的净收益。

可安全探索的方向是保留 CPU grid tuple、一次准备 Python shape/max-length/split-size，并复用几何相关索引；完整 key 至少包含有序 `(t,h,w)`、merge 配置、device 和 dtype，不能只按 tensor shape 缓存。缓存学习到的位置 embedding 值还涉及权重更新/加载失效，优先只缓存几何索引和权重；不缓存视觉特征、KV 或训练激活。RoPE position IDs 与返回的 `rope_deltas` 必须成对保持原语义。本轮尚未实现此候选。

## 原始 profile 导出的候选与证据边界

1. **独立 DDP bucket 128/256 MiB sweep**：确有密集 NCCL kernels，值得测通信次数与重叠变化。安装版 `training_args.py:1338` 提供 `ddp_bucket_cap_mb`；`trainer.py:2203–2209` 把它作为 `bucket_cap_mb` 传入 Accelerate 的 DDP handler。使用 `++trainer.ddp_bucket_cap_mb=128` / `256`，其余优化、batch、数据顺序和 optimizer 保持一致。更大 bucket 可能减少启动开销，也可能推迟通信、减弱 backward overlap；不能预先保证加速或把全部 NCCL duration 当可省时间。147 次同步不作为该候选依据。
2. **prefix trim + 已求得 Python int 边界复用**：prefix helper 已在 VLM 前得到含 `future_start` 的前缀长度，可在其实际 applied 时直接作为后续 crop 长度，避免重复 nonzero 和 144 次 GPU scalar crop 转换。需要报告真实裁剪 tokens、fallback 与 loss/gradient 校验；不预先声称可省上面的 197.811 ms。
3. **固定轨迹 Cholesky factor cache**：仍可作为小范围候选实测，但整个 trajectory CPU range 在 PID 8779 为约 7.324 ms/步、8-process 均值约 7.914 ms/步；对应已归因 GPU kernel sum 均值约 0.470 ms/步。完整阶段时间不是可省时间，求解/输入依赖工作仍然存在。workers=0/preload 不能仅因“存在数据处理”就排在 GPU/通信问题之前。

8 设备的前向 phase GPU kernel duration sum 均值约为 language **418.565 ms/步**、visual **147.501 ms/步**。这是按 launch 关联的 kernel sum，不是阶段 critical path；当前 forward hooks 没有给 expert backward 加阶段 range。最大的 `nvjet_*` kernels 维持 `other` 分类，未凭名称猜测为 GEMM。既有 Liger/compile 是基线/复现选项，不计作此次新优化。

上述 prefix、因子缓存和 DDP bucket 现已完成候选及组合测量；最终 trace 的结果与实际生效计数见前文。视觉 grid 元数据缓存仍未实现，不能与已经验证的候选混在一起。

## 可复核查询

先导出并检查完整采集窗口：

```bash
nsys export --type=sqlite --output=nsight.sqlite nsight.nsys-rep
python -m alpamayo1_5_sft.benchmark_h20.analyze_nsys nsight.sqlite \
  --expected-ranks 8 --first-step 5 --last-step 8 --strict-window \
  --output analysis.json --csv-dir analysis_csv
```

下列 SQL 在该 SQLite 上复现上表。`globalTid` 的 PID 为 `(globalTid >> 24) & 0xFFFFFF`；NVTX 注册字符串通过 `textId → StringIds.id` 关联，literal `text` 优先：

```sql
WITH n AS (
  SELECT e.start, e.end, e.globalTid, COALESCE(e.text,s.value) AS name
  FROM NVTX_EVENTS e LEFT JOIN StringIds s ON e.textId=s.id
), a AS (
  SELECT e.start, e.end, e.globalTid
  FROM CUPTI_ACTIVITY_KIND_RUNTIME e JOIN StringIds s ON e.nameId=s.id
  WHERE ((e.globalTid >> 24) & 16777215)=8779
    AND s.value LIKE 'cudaStreamSynchronize%'
), attributed AS (
  SELECT a.*, COALESCE((
    SELECT n.name FROM n
    WHERE n.globalTid=a.globalTid AND n.start<=a.start AND n.end>=a.end
    ORDER BY n.start DESC LIMIT 1
  ),'outside') AS phase
  FROM a
)
SELECT phase, COUNT(*) AS calls, SUM(end-start)/1e6 AS host_api_sum_ms
FROM attributed GROUP BY phase ORDER BY SUM(end-start) DESC;
```

在上面的 `a` CTE 后用如下 SELECT，可复现 step 5 的 1 / 144 / 2 次拆分；各边界取自相同 process 的相邻 NVTX range：

```sql
SELECT segment, COUNT(*) AS calls, SUM(end-start)/1e6 AS host_api_sum_ms
FROM (
  SELECT a.*, CASE
    WHEN start>=489047192 AND end<=687096787 THEN 'after_vlm_nonzero'
    WHEN start>=695384644 AND end<=698979275 THEN 'scalar_crop'
    WHEN start>=747798514 AND end<=992466520 THEN 'post_expert'
  END AS segment FROM a
) WHERE segment IS NOT NULL GROUP BY segment;
```

安装源码 SHA256（host 路径位于 `/raid/charlie/alpamayo_training/venv/lib/python3.12/site-packages/transformers/`）：

| 相对源码路径 | SHA256 |
|---|---|
| `models/qwen3_vl/modeling_qwen3_vl.py` | `dd63ed3b124232735b3dca1bfa28f9d6b0d3f7182afcb75dde8f3e724b2b22da` |
| `cache_utils.py` | `f4e11caf14c75bdefa5e3003b93ef0e8b97fb6103375705ad944b0b344d7da8a` |
| `modeling_flash_attention_utils.py` | `293fe81c6bd38aac8a3bc6ae086ff21b9695b349e97c9650b1fe61e42a36d710` |
