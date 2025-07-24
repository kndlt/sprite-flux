from diffusers import (
    FluxPipeline,
    FluxControlNetPipeline,
    FluxControlNetModel,
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
    DiffusionPipeline
)
from diffusers.models import FluxMultiControlNetModel
from diffusers.utils import load_image
from PIL import Image, ImageDraw
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

def generate_filename(model_id, prompt, seed, quantization=None, lora_repo=None, lora_scale=None, controlnet_repo=None, control_scale=None, **kwargs):
    """Generate SEO-friendly filename with model, prompt, and parameters"""
    # Get short model name
    model_name = get_short_model_name(model_id)
    # if quantization:
    #     model_name = f"{model_name}-{quantization}"
    if lora_repo:
        short_lora = f"l{round(lora_scale*100)}"
        model_name = f"{model_name}_{short_lora}"
    if controlnet_repo:
        short_controlnet = f"c{round(control_scale*100)}"
        model_name = f"{model_name}_{short_controlnet}"

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
    filename = f"{model_name}___{seo_prompt_with_hash}___{seo_params}.jpg"
    
    return filename

def load_pipe(model_id: str, quant: str = "bf16", controlnet_repo: str | None = None) -> DiffusionPipeline:
    """
    quant ∈ {"bf16", "fp16", "fp32", "int8", "int4"}
    Streams weights straight to the best GPU (device_map="auto") with
    negligible host-RAM thanks to low_cpu_mem_usage=True.
    """
    quant = quant.lower()
    # args common to every flavour
    kwargs = dict(
        # device_map="balanced", 
        # low_cpu_mem_usage=True,
    )
    if quant == "fp16":
        kwargs["torch_dtype"] = torch.float16
    if quant == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif quant == "fp32":
        kwargs["torch_dtype"] = torch.float32          # explicit but optional
    elif quant == "int8":
        kwargs["load_in_8bit"] = True                  # bitsandbytes
    elif quant == "int4":
        kwargs["load_in_4bit"] = True                  # bitsandbytes
    else:
        raise ValueError(f"unknown quant: {quant}")
    if controlnet_repo:
        controlnet_union = FluxControlNetModel.from_pretrained(controlnet_repo, torch_dtype=kwargs["torch_dtype"])
        controlnet = FluxMultiControlNetModel([controlnet_union]) # we always recommend loading via FluxMultiControlNetModel
        # controlnet.to("cuda")
        pipe = FluxControlNetPipeline.from_pretrained(
            model_id, controlnet=controlnet, **kwargs
        )
        pipe.to("cuda")
    else:
        pipe = FluxPipeline.from_pretrained(model_id, **kwargs)
    # scheduler swap (do it AFTER instantiation)
    # pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    return pipe

def generate_and_save_image(
    model_id, 
    prompt, 
    seed, 
    output_dir="outputs", 
    quantization="bf16", 
    lora_repo=None,
    lora_scale=1.0,
    controlnet_repo=None,
    control_scale=0.5,
    **generation_params):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate SEO-friendly filename (without extension)
    base_filename = generate_filename(
        model_id, prompt, seed,
        quantization=quantization,
        lora_repo=lora_repo, lora_scale=lora_scale,
        controlnet_repo=controlnet_repo, control_scale=control_scale,
        **generation_params
    ).replace('.jpg', '')

    # Check if files already exist
    image_path = Path(output_dir) / f"{base_filename}.jpg"
    json_path = Path(output_dir) / f"{base_filename}.json"
    
    if image_path.exists() and json_path.exists():
        print(f"Skipping generation - files already exist:")
        print(f"  Image: {image_path}")
        print(f"  Parameters: {json_path}")
        return
    
    print(f"↳ Loading model: {model_id}  [{quantization}]")
    pipe = load_pipe(model_id, quant=quantization, controlnet_repo=controlnet_repo)

    # ----------  🎛  LoRA magic  ----------
    if lora_repo:
        pipe.load_lora_weights(lora_repo, adapter_name="retro")
        pipe.set_adapters(["retro"], adapter_weights=[lora_scale])
        print(f"✓ Retro LoRA attached (scale={lora_scale})")

    # Grid mask only if ControlNet present
    cond = build_grid_mask() if controlnet_repo else None
    gen_kwargs = dict(**generation_params)
    if cond is not None:
        # cond_tensor = (
        #     torch.from_numpy(np.array(cond)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        # ).to(pipe.device)  # 👈 move to same device
        gen_kwargs.update(
            # https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro
            dict(
                control_image=[load_image("./templates/sheet-of-2x2-chibi-characters-1024.png")],
                control_mode=[1],
                controlnet_conditioning_scale=[control_scale],
            )
        )

    set_seed(seed)

    print(f"Generating image with seed {seed}")
    # Pass any additional generation parameters to the pipeline
    with torch.inference_mode():
        image = pipe(prompt, **gen_kwargs).images[0]

    # Save image
    image.save(image_path, format='JPEG', quality=90)
    print(f"Saved image to {image_path}")
    
    # Save parameters as JSON
    params_data = {
        "model_id": model_id,
        "prompt": prompt,
        "seed": seed,
        "generation_timestamp": datetime.now().isoformat(),
        "generation_parameters": generation_params,
        "quantization": quantization,
        "lora_repo": lora_repo,
        "lora_scale": lora_scale,
        "controlnet_repo": controlnet_repo, 
        "control_scale":control_scale,
        "output_image": f"{base_filename}.jpg"
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
        "black-forest-labs/FLUX.1-dev": "f1d",
        "black-forest-labs/FLUX.1-schnell": "f1s"
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

def get_short_lora_name(lora_repo):
    """Generate a short SEO-friendly LoRA name with hash"""
    if not lora_repo:
        return None
    
    lora_short_names = {
        "prithivMLmods/Retro-Pixel-Flux-LoRA": "retro"
    }
    
    # Use repo name part
    base = lora_repo.split('/')[-1]
    seo_base = seoify_text(base, max_length=20)
    # 4-digit hash for uniqueness
    lora_hash = hashlib.sha256(lora_repo.encode()).hexdigest()[:4]
    return f"{seo_base}-{lora_hash}"

def get_short_controlnet_name(controlnet_repo):
    """Generate a short SEO-friendly ControlNet name with hash"""
    if not controlnet_repo:
        return None
    base = controlnet_repo.split('/')[-1]
    seo_base = seoify_text(base, max_length=20)
    controlnet_hash = hashlib.sha256(controlnet_repo.encode()).hexdigest()[:4]
    return f"{seo_base}-{controlnet_hash}"

def build_grid_mask(res=32, cells=8, pad=1):
    """
    White squares where sprites go, black elsewhere.
    For FLUX 512x512 base resolution.
    """
    m = Image.new("RGB", (res, res), (255, 255, 255))
    draw, cell =  ImageDraw.Draw(m), res // cells
    for y in range(cells):
        for x in range(cells):
            draw.rectangle(
                [x*cell+pad, y*cell+pad, (x+1)*cell-pad, (y+1)*cell-pad],
                fill=(128, 128, 128)
            )
    return m

# ---- Config ----
models = [
    # {
    #     "model_id": "black-forest-labs/FLUX.1-dev",
    #     "quantization": "bf16",
    #     "lora_repo": "prithivMLmods/Retro-Pixel-Flux-LoRA",
    #     "lora_scale": 1.0
    # },
    # {
    #     "model_id": "black-forest-labs/FLUX.1-schnell",
    #     "quantization": "bf16",
    #     "lora_repo": "prithivMLmods/Retro-Pixel-Flux-LoRA",
    #     "lora_scale": 1.0,
    #     # "controlnet_repo": None,
    #     "controlnet_repo": "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro",  # v1 (has tile)
    #     "control_scale": 1.0,
    # },
    {
        "model_id": "black-forest-labs/FLUX.1-dev",
        "quantization": "bf16",
        # "guidance_scale": 10,
        # "max_sequence_length": 50,
        # "generator": torch.Generator("cpu").manual_seed(0)
        "lora_repo": "prithivMLmods/Retro-Pixel-Flux-LoRA",
        "lora_scale": 0.8,
        # "lora_repo": None,
        # "lora_scale": 0.8,
        "controlnet_repo": "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro",
        # "controlnet_repo": None,
        "control_scale": 0.30,
    },
    # Add more variants here if needed
]

prompt = """
2 by 2 Retro Pixel Art Character Sheet of NPC characters for pixel art game called Machi.
"""

seed = 12

# Optional generation parameters (add any pipeline-specific parameters here)
generation_params = {
    # "num_inference_steps": 40,
    # "guidance_scale": 7.5,
    # "width": 1024,
    # "height": 1024,
}

for model in models:
    generate_and_save_image(
        model["model_id"], 
        prompt, 
        seed=seed,
        output_dir="outputs",
        quantization=model["quantization"],
        lora_repo=model["lora_repo"],
        lora_scale=model["lora_scale"],
        controlnet_repo=model["controlnet_repo"], 
        control_scale=model["control_scale"],
        **generation_params
    )
