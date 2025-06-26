from diffusers import DiffusionPipeline
import torch
import random
import numpy as np
from pathlib import Path

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def generate_and_save_image(model_id, prompt, seed, output_dir="outputs"):
    print(f"Loading model: {model_id}")
    pipe = DiffusionPipeline.from_pretrained(model_id)
    pipe.to("cuda")

    set_seed(seed)

    print(f"Generating image with seed {seed}")
    image = pipe(prompt).images[0]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    image_path = Path(output_dir) / f"{model_id.replace('/', '_')}.png"
    image.save(image_path)
    print(f"Saved to {image_path}")

# ---- Config ----
models = [
    # "black-forest-labs/FLUX.1-dev",
    "black-forest-labs/FLUX.1-schnell",
    # Add more variants here if needed
]
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
seed = 1337

for model in models:
    generate_and_save_image(model, prompt, seed)
