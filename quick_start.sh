#!/bin/bash
# Quick start script for LoRA fine-tuning with your current setup

echo "🚀 Starting LoRA fine-tuning on FLUX.1-schnell"
echo "📁 Using training data from: inputs/lora-fine-tuning"
echo "🎯 Output will be saved to: lora_outputs"
echo "💾 You have 94GB VRAM - using optimized settings"

# Run training with optimized settings for your RTX PRO 6000
poetry run python scripts/fine_tune.py \
    --data_dir "inputs/lora-fine-tuning" \
    --output_dir "./lora_outputs" \
    --model_name "black-forest-labs/FLUX.1-schnell" \
    --resolution 1024 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --num_train_epochs 50 \
    --save_steps 250 \
    --lora_rank 128 \
    --lora_alpha 128 \
    --mixed_precision "bf16" \
    --gradient_checkpointing \
    --seed 42

echo "✅ Training complete! Check lora_outputs/ for your trained LoRA weights"
