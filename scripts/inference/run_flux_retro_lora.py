"""
FLUX Retro LoRA Inference Script

This script generates images using FLUX models with multiple prompts support.

Features:
- Multiple prompts: Define a list of prompts and generate images for all of them
- Multiple models: Test different FLUX model variants
- Smart file naming: SEO-friendly filenames with prompt hash for uniqueness
- Skip existing: Automatically skips generation if output files already exist
- Seed management: Each prompt gets a unique seed (base_seed + index)
- Metadata saving: Saves generation parameters as JSON alongside images

Usage:
1. Edit the 'prompts' list to add your desired prompts
2. Adjust 'base_seed' if needed (each prompt gets base_seed + index)
3. Modify 'generation_params' for custom parameters
4. Run the script: python run_flux_retro_lora.py
"""

from diffusers import DiffusionPipeline
import torch
import random
import numpy as np
import re
import json
import hashlib
from pathlib import Path
from datetime import datetime

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def seoify_text(text, max_length=100):
    """Convert text to SEO-friendly format: lowercase, alphanumeric + hyphens, truncated"""
    # Convert to lowercase and replace spaces/special chars with hyphens
    seo_text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    seo_text = re.sub(r'\s+', '-', seo_text.strip())
    # Remove multiple consecutive hyphens
    seo_text = re.sub(r'-+', '-', seo_text)
    # Remove leading/trailing hyphens
    seo_text = seo_text.strip('-')
    # Truncate to max length
    if len(seo_text) > max_length:
        seo_text = seo_text[:max_length].rstrip('-')
    return seo_text

def generate_filename(model_id, prompt, seed, **kwargs):
    """Generate SEO-friendly filename with model, prompt, and parameters"""
    # Get short model name
    model_name = get_short_model_name(model_id)
    
    # SEO-ify the prompt (limit to 25 chars to leave room for 4-digit hash)
    seo_prompt = seoify_text(prompt, 25)
    
    # Add 4-digit hash of the original prompt for uniqueness
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:4]
    seo_prompt_with_hash = f"{seo_prompt}-{prompt_hash}"  # Total: max 30 chars
    
    # Create parameter string - if too long, use hash instead
    param_parts = [f"seed-{seed}"]
    for key, value in kwargs.items():
        if value is not None:
            param_parts.append(f"{key}-{value}")
    
    seo_params = "_".join(param_parts)
    
    # If params too long (>30 chars), use hash instead
    if len(seo_params) > 30:
        param_dict = {"seed": seed, **kwargs}
        param_string = json.dumps(param_dict, sort_keys=True)
        param_hash = hashlib.sha256(param_string.encode()).hexdigest()[:7]
        seo_params = f"params-{param_hash}"  # Total: 14 chars (well under 30)
    
    # Combine all parts
    filename = f"{model_name}_{seo_prompt_with_hash}_{seo_params}.png"
    
    return filename

def get_short_model_name(model_id):
    """Generate a short, SEO-friendly model name with hash"""
    # Common model short names mapping
    model_short_names = {
        "black-forest-labs/FLUX.1-dev": "flux1-dev",
        "black-forest-labs/FLUX.1-schnell": "flux1-schnell", 
        "stabilityai/stable-diffusion-xl-base-1.0": "sdxl-base",
        "stabilityai/stable-diffusion-2-1": "sd2-1",
        "runwayml/stable-diffusion-v1-5": "sd1-5",
        "CompVis/stable-diffusion-v1-4": "sd1-4",
    }
    
    # If we have a predefined short name, use it
    if model_id in model_short_names:
        return model_short_names[model_id]
    
    # Otherwise, create a short hash-based name
    model_hash = hashlib.sha256(model_id.encode()).hexdigest()[:6]
    # Extract meaningful parts from model ID
    parts = model_id.replace('/', '-').replace('_', '-').split('-')
    # Take first meaningful part and combine with hash
    if len(parts) > 0:
        base_name = parts[0][:8]  # Max 8 chars from first part
        return f"{base_name}-{model_hash}"
    else:
        return f"model-{model_hash}"

def generate_and_save_batch(model_id, prompts, base_seed, output_dir="outputs", **generation_params):
    """Generate and save images for multiple prompts efficiently"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check which files already exist
    prompts_to_generate = []
    seeds_to_generate = []
    filenames_to_generate = []
    
    print(f"Checking existing files for {len(prompts)} prompts...")
    for i, prompt in enumerate(prompts):
        seed = base_seed + i
        base_filename = generate_filename(model_id, prompt, seed, **generation_params).replace('.png', '')
        image_path = Path(output_dir) / f"{base_filename}.png"
        json_path = Path(output_dir) / f"{base_filename}.json"
        
        if image_path.exists() and json_path.exists():
            print(f"Skipping (exists): {base_filename}.png")
        else:
            prompts_to_generate.append(prompt)
            seeds_to_generate.append(seed)
            filenames_to_generate.append(base_filename)
    
    # If no new images to generate, return early
    if not prompts_to_generate:
        print("All images already exist, skipping generation")
        return
    
    print(f"Loading model: {model_id}")
    pipe = DiffusionPipeline.from_pretrained(model_id)
    pipe.to("cuda")
    
    print(f"Generating {len(prompts_to_generate)} new images...")
    
    try:
        # Generate images one by one (more memory efficient than true batching)
        for i, (prompt, seed, base_filename) in enumerate(zip(prompts_to_generate, seeds_to_generate, filenames_to_generate)):
            print(f"\n[{i+1}/{len(prompts_to_generate)}] Generating: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
            print(f"Seed: {seed}")
            
            set_seed(seed)
            
            # Generate image
            image = pipe(prompt, **generation_params).images[0]
            
            # Save image
            image_path = Path(output_dir) / f"{base_filename}.png"
            image.save(image_path)
            print(f"Saved: {image_path}")
            
            # Save parameters as JSON
            json_path = Path(output_dir) / f"{base_filename}.json"
            params_data = {
                "model_id": model_id,
                "prompt": prompt,
                "seed": seed,
                "generation_timestamp": datetime.now().isoformat(),
                "generation_parameters": generation_params,
                "output_image": f"{base_filename}.png"
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(params_data, f, indent=2, ensure_ascii=False)
            print(f"Saved metadata: {json_path}")
        
        print(f"Completed batch generation for model: {model_id}")
    
    finally:
        # Aggressive GPU memory cleanup for large models
        pipe.to("cpu")  # Move to CPU first
        del pipe
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for all operations to complete

# ---- Config ----
models = [
    "black-forest-labs/FLUX.1-dev",
    "black-forest-labs/FLUX.1-schnell",
    # Add more variants here if needed
]

# Multiple prompts - each will get a different seed (base_seed + index)
prompts = [
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    # "A magical forest with glowing mushrooms, fantasy style, vibrant colors",
    # "Cyberpunk city at night, neon lights, futuristic, detailed architecture",
    # "Medieval castle on a hill, dramatic lighting, sunset, photorealistic",
    # Add more prompts here as needed
]

base_seed = 1337

# Optional generation parameters (add any pipeline-specific parameters here)
generation_params = {
    # "num_inference_steps": 10,
    # "guidance_scale": 7.5,
    "width": 256,
    "height": 256,
}

# Generate images for each model and all prompts
for model in models:
    print(f"\n=== Processing model: {model} ===")
    generate_and_save_batch(model, prompts, base_seed, **generation_params)
