#!/usr/bin/env bash
set -eu
source /raid/charlie/cc_run/env.sh

WORKDIR=/raid/charlie/alpamayo/alpamayo-recipes/recipes/alpamayo1_5_sft
VENV=$WORKDIR/a1_5_sft
PAI_DIR=/raid/charlie/pai_dataset
NAV_ANN=/raid/charlie/alpamayo/alpamayo1.5/notebooks/nav_demo_samples.json
CKPT_A1=/raid/charlie/Alpamayo-1.5-10B-A1-format

RUN_NAME=${1:-baseline}
NGPU=${2:-8}
MAXSTEPS=${3:-20}
WARMUP=${4:-8}
LOGSTEPS=${5:-100}
NWORKERS=${6:-2}
BS=${7:-1}
EXTRA=${8:-}

RESULT_PATH=$WORKDIR/benchmark/results/${RUN_NAME}_stage2_${NGPU}gpu.json
mkdir -p $WORKDIR/benchmark/results
cd $WORKDIR
echo "=== stage2 bench run=$RUN_NAME ngpu=$NGPU steps=$MAXSTEPS workers=$NWORKERS extra=[$EXTRA] ==="

DEV=$(python -c "print(\",\".join(str(i) for i in range($NGPU)))")
CUDA_VISIBLE_DEVICES=$DEV $VENV/bin/torchrun --nproc_per_node $NGPU \
    -m alpamayo1_5_sft.train_hf \
    --config-path pkg://alpamayo1_5_sft/configs \
    --config-name sft_stage2_nav \
    ~model.checkpoint_path model.pretrained_model_name_or_path="$CKPT_A1" \
    model.stage1_vlm_checkpoint_path=null \
    data.train_dataset.local_dir="$PAI_DIR" \
    data.train_dataset.annotations_path="$NAV_ANN" \
    data.val_dataset.local_dir="$PAI_DIR" \
    data.val_dataset.annotations_path="$NAV_ANN" \
    trainer.per_device_train_batch_size=$BS \
    trainer.gradient_accumulation_steps=1 \
    trainer.dataloader_num_workers=$NWORKERS \
    trainer.logging_steps=$LOGSTEPS \
    trainer.report_to=none \
    +trainer.max_steps=$MAXSTEPS \
    +trainer.save_strategy=no \
    paths.output_dir="$WORKDIR/benchmark/runs/${RUN_NAME}_stage2_${NGPU}gpu" \
    "+callbacks.timer._target_=alpamayo1_5_sft.step_timer.StepTimerCallback" \
    "+callbacks.timer.warmup_steps=$WARMUP" \
    "+callbacks.timer.result_path=$RESULT_PATH" \
    "+callbacks.timer.run_name=$RUN_NAME" \
    "+callbacks.timer.stable_tail=$((MAXSTEPS-WARMUP))" \
    $EXTRA
echo "=== done: $RESULT_PATH ==="
