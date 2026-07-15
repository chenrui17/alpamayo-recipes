# Alpamayo-1 SFT Stage-1 Throughput Benchmark

Updated: 2026-06-11T09:25:44.193107+00:00
Hardware: 8x H20, single node
Method: 50 optimizer steps, average of last 40 stable steps
Tasks: CoC disabled (`sft_bench_no_coc`) / CoC enabled (`sft_bench_coc`)

| Run | Task | Avg Step (s) | vs Task Baseline | workers | Optimizations |
|-----|------|-------------:|-----------------:|--------:|---------------|
| no_coc_baseline | no_coc | 19.018 | +0.0% | 2 | - |
| no_coc_dl_workers8 | no_coc | 10.295 | +84.7% | 8 | - |
| no_coc_production_opt | no_coc | 11.923 | +59.5% | 8 | zip,collate,tf32,cudnn |
| coc_baseline | coc | 15.995 | +0.0% | 2 | - |
| coc_dl_workers8 | coc | 15.429 | +3.7% | 8 | - |
| coc_production_opt | coc | 10.152 | +57.6% | 8 | zip,collate,tf32,cudnn |
| coc_production_opt_workers2 | coc | 10.085 | +58.6% | 2 | zip,collate,tf32,cudnn |
| coc_production_opt_preprocess | coc | 8.859 | +80.6% | 8 | zip,collate,preprocess,tf32,cudnn |
| coc_production_opt_loadcache | coc | 5.492 | +191.2% | 8 | collate,load,tf32,cudnn |
| coc_dl_workers8_preprocess | coc | 14.540 | +10.0% | 8 | preprocess |
| coc_dl_workers8_zip | coc | 16.562 | -3.4% | 8 | zip |
| coc_dl_workers8_collate | coc | 10.245 | +56.1% | 8 | collate |
| coc_dl_workers8_tf32 | coc | 16.800 | -4.8% | 8 | tf32 |
| coc_dl_workers8_cudnn | coc | 17.032 | -6.1% | 8 | cudnn |

## CoC workers=8 single-optimization ablation

Reference: `coc_dl_workers8` = **15.429 s** (workers=8, no zip/collate/tf32/cudnn)

| Run | Avg Step (s) | vs coc_dl_workers8 | Enabled |
|-----|-------------:|-------------------:|---------|
| coc_dl_workers8_zip | 16.562 | -6.8% | zip |
| coc_dl_workers8_collate | 10.245 | +50.6% | collate |
| coc_dl_workers8_tf32 | 16.800 | -8.2% | tf32 |
| coc_dl_workers8_cudnn | 17.032 | -9.4% | cudnn |

---

详细分析见 [`benchmark/PROFILE_ANALYSIS.md`](PROFILE_ANALYSIS.md)。
