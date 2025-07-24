#!/usr/bin/env python3
"""
Test script to generate images using a fine-tuned LoRA with FLUX.1-schnell
"""

import torch
from diffusers import FluxPipeline
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Test LoRA fine-tuned FLUX.1-schnell")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the trained LoRA weights")
    parser.add_argument("--prompt", type=str, default="retro pixel art character sprite", help="Generation prompt")
    parser.add_argument("--output_path", type=str, default="test_output.png", help="Output image path")
    parser.add_argument("--num_inference_steps", type=int, default=4, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=0.0, help="Guidance scale (0.0 for schnell)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lora_scale", type=float, default=1.0, help="LoRA scale")
    
    args = parser.parse_args()
    
    # Load the base model
    print("Loading FLUX.1-schnell...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Load the LoRA weights
    print(f"Loading LoRA weights from {args.lora_path}...")
    pipe.load_lora_weights(args.lora_path, adapter_name="pixel_art")
    pipe.set_adapters(["pixel_art"], adapter_weights=[args.lora_scale])
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Generate image
    print(f"Generating image with prompt: '{args.prompt}'")
    image = pipe(
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=1024,
        width=1024,
        generator=torch.Generator().manual_seed(args.seed),
    ).images[0]
    
    # Save image
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    main()
