# Alpamayo-1.5 SFT 性能优化报告

## 结论

本次 benchmark 使用 `recipes/alpamayo1_5_sft` 的 Stage-1 navigation SFT 配置，对 4 组增量优化进行对比。性能指标为跳过初始 warmup 后，step 5-20 的 wall-clock step interval 平均值。

| 组别 | 稳定阶段平均 step 时间 (s) | 样本数 | 最小值 (s) | 最大值 (s) | 相对 baseline 加速比 | 相对上一组加速比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 41.354 | 16 | 40.233 | 42.504 | 1.00x | n/a |
| dataloader_workers | 20.040 | 16 | 19.455 | 21.418 | 2.06x | 2.06x |
| zip_collate_cache | 12.032 | 16 | 11.691 | 12.372 | 3.44x | 1.67x |
| tf32_cudnn_benchmark | 12.021 | 16 | 11.674 | 12.322 | 3.44x | 1.00x |

主要收益来自数据读取链路优化：开启 dataloader workers 后 step 时间从 41.354s 降至 20.040s；继续开启 zip cache 与 collate cache 后降至 12.032s。进一步开启 TF32 与 cuDNN benchmark 后，本次短 benchmark 中 step 时间为 12.021s，收益基本持平。

## Benchmark 配置

| 组别 | 配置 |
| --- | --- |
| `baseline` | 关闭 dataloader workers、zip cache、collate cache、TF32、cuDNN benchmark。 |
| `dataloader_workers` | 设置 `dataloader_num_workers=8`、`dataloader_persistent_workers=true`、`dataloader_prefetch_factor=4`。 |
| `zip_collate_cache` | 在 dataloader workers 基础上设置 `zip_cache=true`、`collate_cache=true`。 |
| `tf32_cudnn_benchmark` | 在数据链路 cache 基础上设置 `tf32=true`、`cudnn_benchmark=true`。 |

## 测试方法

- 报告生成时间: 2026-07-02 13:54:38 CST
- Host: `H20-GPU-03`
- OS: `Linux-5.15.0-1069-nvidia-x86_64-with-glibc2.35`
- GPU: `8 x NVIDIA H20-3e`
- Checkpoint: `/raid/charlie/Alpamayo-1.5-10B-A1-format`
- PAI 数据目录: `/raid/charlie/pai_dataset`
- Nav annotations: `/raid/charlie/alpamayo/alpamayo1.5/notebooks/nav_demo_samples.json`
- 训练入口: `python -m torch.distributed.run -m alpamayo1_5_sft.train_hf`
- Hydra 配置: `sft_stage1_nav`
- 每组运行 step 数: `20`
- 稳定阶段统计窗口: step `5-20`
- 计时方式: `StepTimeCallback` 统计连续两次 `on_step_end` 之间的间隔，并在计时前做 CUDA synchronize，因此包含数据读取、collate、forward/backward/update 的整体 step interval。
- benchmark 运行时关闭 checkpoint saving、evaluation、W&B 和外部 reporting。

## 结果文件

- `baseline`: `/raid/charlie/alpamayo/git-0611/alpamayo-recipes/recipes/alpamayo1_5_sft/benchmark/results/baseline.json`, `/raid/charlie/alpamayo/git-0611/alpamayo-recipes/recipes/alpamayo1_5_sft/benchmark/results/baseline.log`
- `dataloader_workers`: `/raid/charlie/alpamayo/git-0611/alpamayo-recipes/recipes/alpamayo1_5_sft/benchmark/results/dataloader_workers.json`, `/raid/charlie/alpamayo/git-0611/alpamayo-recipes/recipes/alpamayo1_5_sft/benchmark/results/dataloader_workers.log`
- `zip_collate_cache`: `/raid/charlie/alpamayo/git-0611/alpamayo-recipes/recipes/alpamayo1_5_sft/benchmark/results/zip_collate_cache.json`, `/raid/charlie/alpamayo/git-0611/alpamayo-recipes/recipes/alpamayo1_5_sft/benchmark/results/zip_collate_cache.log`
- `tf32_cudnn_benchmark`: `/raid/charlie/alpamayo/git-0611/alpamayo-recipes/recipes/alpamayo1_5_sft/benchmark/results/tf32_cudnn_benchmark.json`, `/raid/charlie/alpamayo/git-0611/alpamayo-recipes/recipes/alpamayo1_5_sft/benchmark/results/tf32_cudnn_benchmark.log`
