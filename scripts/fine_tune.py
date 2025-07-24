import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import FluxPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
import json
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

logger = get_logger(__name__, log_level="INFO")

def set_seed(seed):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class PixelArtDataset(Dataset):
    """Dataset class for pixel art training data"""
    
    def __init__(self, data_dir, tokenizer, text_encoder_2_tokenizer, size=1024, center_crop=True):
        self.data_dir = Path(data_dir)
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.text_encoder_2_tokenizer = text_encoder_2_tokenizer
        
        # Find all image files in the directory
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            self.image_paths.extend(self.data_dir.glob(f"**/{ext}"))
            self.image_paths.extend(self.data_dir.glob(f"**/{ext.upper()}"))
        
        print(f"Found {len(self.image_paths)} images in {data_dir}")
        
        # Create transforms
        self.transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        image = self.transforms(image)
        
        # Generate caption based on filename or use a default pixel art caption
        caption = self.generate_caption(image_path)
        
        # Tokenize caption for both text encoders
        text_inputs = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        text_inputs_2 = self.text_encoder_2_tokenizer(
            caption,
            padding="max_length",
            max_length=self.text_encoder_2_tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": image,
            "input_ids": text_inputs.input_ids.flatten(),
            "attention_mask": text_inputs.attention_mask.flatten(),
            "input_ids_2": text_inputs_2.input_ids.flatten(),
            "attention_mask_2": text_inputs_2.attention_mask.flatten(),
            "caption": caption
        }
    
    def generate_caption(self, image_path):
        """Generate caption based on image filename or return default"""
        stem = image_path.stem.lower()
        
        # Check for specific keywords in filename
        if any(word in stem for word in ['pixel', 'retro', '8bit', '16bit']):
            base_caption = "retro pixel art"
        elif any(word in stem for word in ['character', 'sprite', 'npc']):
            base_caption = "pixel art character sprite"
        elif any(word in stem for word in ['sheet', 'atlas']):
            base_caption = "pixel art sprite sheet"
        else:
            base_caption = "pixel art"
            
        # Add variations
        variations = [
            f"{base_caption} in retro game style",
            f"detailed {base_caption} with clean lines",
            f"{base_caption} for indie game",
            f"high quality {base_caption}",
            f"{base_caption} with vibrant colors"
        ]
        
        return random.choice(variations)

def collate_fn(examples):
    """Collate function for DataLoader"""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.stack([example["input_ids"] for example in examples])
    attention_masks = torch.stack([example["attention_mask"] for example in examples])
    input_ids_2 = torch.stack([example["input_ids_2"] for example in examples])
    attention_masks_2 = torch.stack([example["attention_mask_2"] for example in examples])
    
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "input_ids_2": input_ids_2,
        "attention_mask_2": attention_masks_2,
    }

def create_lora_config(rank=64, alpha=64, target_modules=None):
    """Create LoRA configuration for FLUX model"""
    if target_modules is None:
        # Target the transformer blocks in FLUX
        target_modules = [
            "to_k", "to_q", "to_v", "to_out.0",
            "ff.net.0.proj", "ff.net.2",
            "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0"
        ]
    
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none"
    )

def main():
    parser = argparse.ArgumentParser(description="Fine-tune FLUX.1-schnell with LoRA")
    parser.add_argument("--data_dir", type=str, default="inputs/lora-fine-tuning", help="Directory containing training images")
    parser.add_argument("--output_dir", type=str, default="./lora_outputs", help="Output directory for LoRA weights")
    parser.add_argument("--model_name", type=str, default="black-forest-labs/FLUX.1-schnell", help="Model to fine-tune")
    parser.add_argument("--resolution", type=int, default=1024, help="Training resolution")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help="Learning rate scheduler")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Learning rate warmup steps")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Maximum training steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lora_rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Hub model ID")
    
    args = parser.parse_args()
    
    # Initialize accelerator
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
    
    # Set up logging
    if accelerator.is_local_main_process:
        logger.info(f"Starting LoRA fine-tuning of {args.model_name}")
        logger.info(f"Training data directory: {args.data_dir}")
        logger.info(f"Output directory: {args.output_dir}")
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizers
    logger.info("Loading FLUX.1-schnell model...")
    pipeline = FluxPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16,
        device_map="balanced" if torch.cuda.device_count() > 1 else None,
    )

    # override the scheduler for training
    pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    
    # now your scheduler.add_noise will exist
    scheduler = pipeline.scheduler

    pipeline.to(accelerator.device)
    
    # Get the transformer (UNet-like) model
    transformer = pipeline.transformer
    text_encoder = pipeline.text_encoder
    text_encoder_2 = pipeline.text_encoder_2
    tokenizer = pipeline.tokenizer
    tokenizer_2 = pipeline.tokenizer_2
    vae = pipeline.vae
    scheduler = pipeline.scheduler
    
    # Freeze all parameters except those we'll train with LoRA
    transformer.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    vae.requires_grad_(False)
    
    # Apply LoRA to the transformer
    lora_config = create_lora_config(rank=args.lora_rank, alpha=args.lora_alpha)
    transformer = get_peft_model(transformer, lora_config)
    
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    
    # Create dataset and dataloader
    logger.info("Creating dataset...")
    dataset = PixelArtDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        text_encoder_2_tokenizer=tokenizer_2,
        size=args.resolution,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )
    
    # Calculate total training steps
    num_update_steps_per_epoch = len(dataloader)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = args.max_train_steps // num_update_steps_per_epoch
    
    # Create learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    
    # Prepare everything with accelerator
    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = len(dataloader)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    for epoch in range(args.num_train_epochs):
        transformer.train()
        train_loss = 0.0
        
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(transformer):
                # Convert images to latent space
                with torch.no_grad():
                    # latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                    pixel_values = batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                
                # Sample noise to add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample a random timestep for each image
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                
                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings
                with torch.no_grad():
                    prompt_embeds = text_encoder(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        return_dict=False,
                    )[0]
                    
                    pooled_prompt_embeds = text_encoder_2(
                        batch["input_ids_2"],
                        attention_mask=batch["attention_mask_2"],
                        return_dict=False,
                    )[0]
                
                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
                
                # Calculate loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # Gather the losses across all processes for logging (if we use distributed training)
                avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
                # Backpropagate
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                logs = {
                    "step_loss": train_loss,
                    "lr": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch,
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                train_loss = 0.0
                
                # Save checkpoint
                if global_step % args.save_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        
                        # Save LoRA weights
                        transformer.save_pretrained(os.path.join(save_path, "lora"))
                        logger.info(f"Saved checkpoint to {save_path}")
            
            if global_step >= args.max_train_steps:
                break
    
    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = accelerator.unwrap_model(transformer)
        transformer.save_pretrained(os.path.join(args.output_dir, "final_lora"))
        
        # Save training info
        training_info = {
            "model_name": args.model_name,
            "data_dir": args.data_dir,
            "resolution": args.resolution,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "max_train_steps": args.max_train_steps,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "seed": args.seed,
            "training_completed": datetime.now().isoformat(),
        }
        
        with open(os.path.join(args.output_dir, "training_info.json"), "w") as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"Training completed! LoRA weights saved to {args.output_dir}")
        
        # Push to hub if requested
        if args.push_to_hub:
            if args.hub_model_id is None:
                args.hub_model_id = f"flux-schnell-pixel-art-lora-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            transformer.push_to_hub(
                args.hub_model_id,
                commit_message=f"LoRA fine-tuned FLUX.1-schnell for pixel art generation",
            )
            logger.info(f"Model pushed to hub: {args.hub_model_id}")

if __name__ == "__main__":
    main()
