from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")

pipe.to("cuda")  # 👈 THIS is what makes it actually use your GPU

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt).images[0]
image.save("output.png")