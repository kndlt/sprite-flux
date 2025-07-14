# FLUX.1-schnell LoRA Fine-tuning

This directory contains scripts for fine-tuning FLUX.1-schnell with LoRA (Low-Rank Adaptation) for pixel art generation.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements_training.txt
```

2. Prepare your training data:
   - Create a directory with your pixel art images (JPG, PNG, WebP supported)
   - Images should be high quality pixel art
   - The script will automatically generate captions based on filenames
   - Recommended: At least 100-500 images for good results

## Training

### Basic Usage

```bash
python scripts/fine_tune.py \
    --data_dir "/path/to/your/pixel_art_images" \
    --output_dir "./lora_outputs" \
    --num_train_epochs 100 \
    --batch_size 1 \
    --learning_rate 1e-4
```

### Advanced Usage

See `train_example.sh` for more configuration options:

```bash
./train_example.sh
```

### Key Parameters

- `--data_dir`: Directory containing your training images
- `--output_dir`: Where to save the trained LoRA weights
- `--model_name`: Base model (default: "black-forest-labs/FLUX.1-schnell")
- `--resolution`: Training resolution (default: 1024)
- `--batch_size`: Batch size (start with 1 for 24GB VRAM)
- `--gradient_accumulation_steps`: Effective batch size multiplier
- `--learning_rate`: Learning rate (1e-4 to 5e-5 recommended)
- `--num_train_epochs`: Number of training epochs
- `--lora_rank`: LoRA rank (64-128 recommended)
- `--lora_alpha`: LoRA alpha (usually same as rank)

### Memory Requirements

- **Minimum**: 16GB VRAM (batch_size=1, gradient_checkpointing=True)
- **Recommended**: 24GB+ VRAM for better performance
- Use `--gradient_checkpointing` to reduce memory usage
- Reduce `--batch_size` if you run out of memory

## Testing the Trained LoRA

After training, test your LoRA with:

```bash
python test_lora.py \
    --lora_path "./lora_outputs/final_lora" \
    --prompt "retro pixel art character sprite" \
    --output_path "test_output.png"
```

## Training Tips

1. **Data Quality**: Use high-quality, consistent pixel art images
2. **Captions**: The script auto-generates captions, but you can modify the `generate_caption` method for custom captions
3. **Learning Rate**: Start with 1e-4, reduce if training is unstable
4. **LoRA Rank**: Higher rank (128-256) for more detailed learning, but requires more VRAM
5. **Training Steps**: Monitor loss - stop when it plateaus
6. **Validation**: Periodically test generation quality during training

## File Structure After Training

```
lora_outputs/
├── final_lora/           # Final LoRA weights
├── checkpoint-500/       # Periodic checkpoints
├── checkpoint-1000/
└── training_info.json    # Training metadata
```

## Troubleshooting

### Out of Memory
- Reduce `--batch_size` to 1
- Enable `--gradient_checkpointing`
- Reduce `--resolution` to 512
- Reduce `--lora_rank`

### Poor Quality Results
- Increase training data quantity/quality
- Adjust learning rate (try 5e-5 or 2e-4)
- Increase `--lora_rank` and `--lora_alpha`
- Train for more epochs

### Training Too Slow
- Increase `--batch_size` if you have VRAM
- Use multiple GPUs with `accelerate`
- Reduce `--resolution` for faster training

## Using with Hugging Face Hub

To automatically upload your trained LoRA:

```bash
python scripts/fine_tune.py \
    --data_dir "/path/to/images" \
    --push_to_hub \
    --hub_model_id "your-username/flux-schnell-pixel-art-lora"
```

Make sure you're logged in to Hugging Face:
```bash
huggingface-cli login
```
