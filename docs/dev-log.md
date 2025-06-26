# Dev Log

## 6/26/25 - Installing flux model into ubuntu server.

SSH to server.
```
# Gin
mkdir sprite-flux
cd sprite-flux
poetry init --name sprite-flux --python ^3.12 --no-interaction
poetry env use python3.12
eval $(poetry env activate)
poetry source add --priority=explicit torch-cu128 https://download.pytorch.org/whl/cu128
poetry add --source torch-cu128 torch torchvision torchaudio
poetry add protobuf
poetry run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())" # -> 2.7.1+cu128 12.8 True
poetry add diffusers transformers accelerate xformers
```

If you want to access files from your client:
```
code --folder-uri "vscode-remote://ssh-remote+gin/home/gin/dev/sprite-flux"
```
But assume all the commands below will be ran in server.

Install git LFS
```
sudo apt update
sudo apt install git-lfs
git lfs install
```

Log into huggingface
```
huggingface-cli login
```

Go to website to agree to Terms.
https://huggingface.co/black-forest-labs/FLUX.1-dev

Create a test python script
```
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt).images[0]
```

Try running it.
```
poetry run python scripts/inference/run_flux_retro_lora.py
```
Download takes a while.


