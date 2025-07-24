#!/bin/bash
# Example script to run LoRA fine-tuning on FLUX.1-schnell

# Make sure you have your training data in inputs/lora-fine-tuning directory
# The script will automatically generate captions based on filenames

# Basic training run
python scripts/fine_tune.py \
    --data_dir "inputs/lora-fine-tuning" \
    --output_dir "./lora_outputs" \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --resolution 1024 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --num_train_epochs 100 \
    --save_steps 500 \
    --lora_rank 64 \
    --lora_alpha 64 \
    --mixed_precision "bf16" \
    --gradient_checkpointing \
    --seed 42

# Advanced training run with more configuration
# python scripts/fine_tune.py \
#     --data_dir "inputs/lora-fine-tuning" \
#     --output_dir "./lora_outputs_advanced" \
#     --model_name "black-forest-labs/FLUX.1-schnell" \
#     --resolution 1024 \
#     --batch_size 2 \
#     --gradient_accumulation_steps 2 \
#     --learning_rate 5e-5 \
#     --lr_scheduler "cosine" \
#     --lr_warmup_steps 1000 \
#     --num_train_epochs 200 \
#     --max_train_steps 10000 \
#     --save_steps 250 \
#     --lora_rank 128 \
#     --lora_alpha 128 \
#     --mixed_precision "bf16" \
#     --gradient_checkpointing \
#     --push_to_hub \
#     --hub_model_id "your-username/flux-schnell-pixel-art-lora" \
#     --seed 42
