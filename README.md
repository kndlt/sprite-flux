# Benchmarking Flux

Let's benchmark flux. Results can be found in docs/results.md


## Prerequisite
- `poetry`
- CUDA capable GPU
- Hugging face account.
- Accept terms for flux1-dev model in hugging face.

## Setup

```
poetry install --no-root
huggingface-cli login
```

## Run

```
poetry run python scripts/inference/run_flux_retro_lora.py
```