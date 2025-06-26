from diffusers import DiffusionPipeline
import gc
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

def generate_filename(model_id, prompt, seed, quantization=None, **kwargs):
    """Generate SEO-friendly filename with model, prompt, and parameters"""
    # Get short model name
    model_name = get_short_model_name(model_id)
    if quantization:                       # e.g. fp16, int8
        model_name = f"{model_name}-{quantization}"

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

def load_pipe(model_id: str, quant: str = "fp16") -> DiffusionPipeline:
    """
    quant ∈ {"fp32", "fp16", "int8", "int4"}
    Streams weights straight to the best GPU (device_map="auto") with
    negligible host-RAM thanks to low_cpu_mem_usage=True.
    """
    quant = quant.lower()

    # args common to every flavour
    kwargs = dict(device_map="balanced", low_cpu_mem_usage=True)

    if quant == "fp16":
        kwargs["torch_dtype"] = torch.float16
    elif quant == "fp32":
        kwargs["torch_dtype"] = torch.float32          # explicit but optional
    elif quant == "int8":
        kwargs["load_in_8bit"] = True                  # bitsandbytes
    elif quant == "int4":
        kwargs["load_in_4bit"] = True                  # bitsandbytes
    else:
        raise ValueError(f"unknown quant: {quant}")

    # ONE call, regardless of precision
    return DiffusionPipeline.from_pretrained(model_id, **kwargs)

def generate_and_save_image(model_id, prompt, seed, output_dir="outputs", quantization="fp16", **generation_params):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate SEO-friendly filename (without extension)
    base_filename = generate_filename(model_id, prompt, seed, quantization=quantization, **generation_params).replace('.png', '')
    
    # Check if files already exist
    image_path = Path(output_dir) / f"{base_filename}.png"
    json_path = Path(output_dir) / f"{base_filename}.json"
    
    if image_path.exists() and json_path.exists():
        print(f"Skipping generation - files already exist:")
        print(f"  Image: {image_path}")
        print(f"  Parameters: {json_path}")
        return
    
    print(f"↳ Loading model: {model_id}  [{quantization}]")
    pipe = load_pipe(model_id, quant=quantization)
    set_seed(seed)

    print(f"Generating image with seed {seed}")
    # Pass any additional generation parameters to the pipeline
    with torch.inference_mode():
        image = pipe(prompt, **generation_params).images[0]

    # Save image
    image.save(image_path)
    print(f"Saved image to {image_path}")
    
    # Save parameters as JSON
    params_data = {
        "model_id": model_id,
        "prompt": prompt,
        "seed": seed,
        "generation_timestamp": datetime.now().isoformat(),
        "generation_parameters": generation_params,
        "quantization": quantization,
        "output_image": f"{base_filename}.png"
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(params_data, f, indent=2, ensure_ascii=False)
    print(f"Saved parameters to {json_path}")

    # -------- VRAM cleanup --------
    del image          # drop PIL image reference
    pipe.reset_device_map() 
    pipe.to("meta")    # move weights off GPU first
    del pipe           # release DiffusionPipeline
    torch.cuda.empty_cache()
    gc.collect()
    print("🧹 VRAM cleared.\n")

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

# ---- Config ----
models = [
    "black-forest-labs/FLUX.1-dev",
    "black-forest-labs/FLUX.1-schnell",
    # Add more variants here if needed
]
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
seed = 1337

# Optional generation parameters (add any pipeline-specific parameters here)
generation_params = {
    # "num_inference_steps": 10,
    # "guidance_scale": 7.5,
    "width": 256,
    "height": 256,
}

for model in models:
    generate_and_save_image(model, prompt, seed, **generation_params)
