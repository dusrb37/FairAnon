#!/bin/bash
# train_fairanon_OSG.sh

# Set GPU
export CUDA_VISIBLE_DEVICES=0

# Training command
accelerate launch train_fairanon_OSG.py \
  --model_id "stabilityai/stable-diffusion-2-inpainting" \
  --data_dir "./data/asian_faces" \
  --output_dir "./outputs/fairanon_stage1" \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 15000 \
  --learning_rate 1e-5 \
  --lambda_orth 0.1 \
  --lambda_norm 0.01 \
  --epsilon 0.05 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --mixed_precision "fp16" \
  --logging_steps 50 \
  --save_steps 500 \
  --num_workers 4 \
  --seed 42