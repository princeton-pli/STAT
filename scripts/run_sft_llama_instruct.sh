#!/bin/bash

MODEL_PATH="Meta-Llama/Meta-Llama-3.2-3B-Instruct"
MODEL_NAME=$(basename "$MODEL_PATH")

DATA_NAME="STAT-Sel" # or "STAT-Syn"

IDX=$SLURM_ARRAY_TASK_ID
LR="${LRS[$IDX]}"


OUTPUT_DIR="./output/finetuned_adapters/${MODEL_NAME}/sft_${DATA_NAME}_${MODEL_NAME}"


echo "Training $MODEL_NAME on $DATA_NAME data"

python sft/finetune_instruct_model.py \
    --model_id ${MODEL_PATH} \
    --dataset_path "./STAT_data/${DATA_NAME}/${DATA_NAME}_${MODEL_NAME}_9.5k.json" \
    --output_dir ${OUTPUT_DIR} \
    --learning_rate 5e-6 \
    --num_train_epochs 3