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
    # Get model name part
    model_name = model_id.replace('/', '_').replace('-', '_')
    
    # SEO-ify the prompt (limit to ~50 chars to leave room for other parts)
    seo_prompt = seoify_text(prompt, 50)
    
    # Create parameter string
    param_parts = [f"seed-{seed}"]
    for key, value in kwargs.items():
        if value is not None:
            param_parts.append(f"{key}-{value}")
    
    seo_params = "_".join(param_parts)
    
    # Combine all parts
    filename = f"{model_name}_{seo_prompt}_{seo_params}.png"
    
    # If filename is too long, use hash for parameters
    if len(filename) > 200:
        # Create a hash of all parameters (including seed)
        param_dict = {"seed": seed, **kwargs}
        param_string = json.dumps(param_dict, sort_keys=True)
        param_hash = hashlib.sha256(param_string.encode()).hexdigest()[:7]
        
        # Use longer prompt since we're saving space on parameters
        seo_prompt = seoify_text(prompt, 80)
        filename = f"{model_name}_{seo_prompt}_{param_hash}.png"
        
        # Final check - if still too long, truncate prompt further
        if len(filename) > 200:
            available_chars = 200 - len(model_name) - len(param_hash) - 10  # 10 for separators and extension
            seo_prompt = seoify_text(prompt, available_chars)
            filename = f"{model_name}_{seo_prompt}_{param_hash}.png"
    
    return filename

def generate_and_save_image(model_id, prompt, seed, output_dir="outputs", **generation_params):
    print(f"Loading model: {model_id}")
    pipe = DiffusionPipeline.from_pretrained(model_id)
    pipe.to("cuda")

    set_seed(seed)

    print(f"Generating image with seed {seed}")
    # Pass any additional generation parameters to the pipeline
    image = pipe(prompt, **generation_params).images[0]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate SEO-friendly filename (without extension)
    base_filename = generate_filename(model_id, prompt, seed, **generation_params).replace('.png', '')
    
    # Save image
    image_path = Path(output_dir) / f"{base_filename}.png"
    image.save(image_path)
    print(f"Saved image to {image_path}")
    
    # Save parameters as JSON
    params_data = {
        "model_id": model_id,
        "prompt": prompt,
        "seed": seed,
        "generation_timestamp": datetime.now().isoformat(),
        "generation_parameters": generation_params,
        "output_image": f"{base_filename}.png"
    }
    
    json_path = Path(output_dir) / f"{base_filename}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(params_data, f, indent=2, ensure_ascii=False)
    print(f"Saved parameters to {json_path}")

# ---- Config ----
models = [
    "black-forest-labs/FLUX.1-dev",
    # "black-forest-labs/FLUX.1-schnell",
    # Add more variants here if needed
]
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
seed = 1337

# Optional generation parameters (add any pipeline-specific parameters here)
generation_params = {
    # "num_inference_steps": 20,
    # "guidance_scale": 7.5,
    # "width": 512,
    # "height": 512,
}

for model in models:
    generate_and_save_image(model, prompt, seed, **generation_params)
