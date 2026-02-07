#!/usr/bin/env python
# train_fairanon_OSG.py
"""
FairAnon Stage 1: Orthogonal Semantic Guidance with Inpainting
"""

import os
import math
import random
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from torchvision import transforms

from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler

logger = get_logger(__name__)


def worker_init_fn(worker_id):
    """Initialize worker seed for reproducibility"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def generate_random_mask(image_size, min_ratio=0.1, max_ratio=0.5):
    """Generate random mask for inpainting"""
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    
    # Random size
    ratio = random.uniform(min_ratio, max_ratio)
    mask_width = int(image_size[0] * ratio)
    mask_height = int(image_size[1] * ratio)
    
    # Random position
    x1 = random.randint(0, image_size[0] - mask_width)
    y1 = random.randint(0, image_size[1] - mask_height)
    x2 = x1 + mask_width
    y2 = y1 + mask_height
    
    # Random shape (rectangle or ellipse)
    if random.random() > 0.5:
        draw.rectangle([x1, y1, x2, y2], fill=255)
    else:
        draw.ellipse([x1, y1, x2, y2], fill=255)
    
    return mask


class FairAnonDataset(Dataset):
    """Dataset for FairAnon Stage 1 training"""
    
    def __init__(self, data_root, tokenizer, size=512):
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.size = size
        
        # Get all image files
        self.image_paths = sorted(list(self.data_root.glob("*.jpg")) + 
                                 list(self.data_root.glob("*.png")))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_root}")
        
        print(f"Found {len(self.image_paths)} images")
        
        # Demographic prompts
        self.demographic_prompts = [
            "An Asian man",
            "An Asian woman", 
            "A White man",
            "A White woman",
            "A Black man", 
            "A Black woman",
            "A Latino man",
            "A Latino woman",
            "A Middle Eastern man",
            "A Middle Eastern woman"
        ]
        
        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        
        # Generate random mask
        mask = generate_random_mask((self.size, self.size))
        
        # Prepare for model
        image_tensor = self.transform(image)
        
        # Convert mask to tensor
        mask_np = np.array(mask.resize((self.size, self.size))).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
        
        # Create masked image
        masked_image = image_tensor * (1 - mask_tensor)
        
        # Random demographic prompt
        prompt = random.choice(self.demographic_prompts)
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": image_tensor,
            "masks": mask_tensor,
            "masked_images": masked_image,
            "input_ids": tokens.input_ids[0],
            "attention_mask": tokens.attention_mask[0]
        }


def collate_fn(examples):
    """Collate function for DataLoader"""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    masks = torch.stack([example["masks"] for example in examples])
    masked_images = torch.stack([example["masked_images"] for example in examples])
    input_ids = torch.stack([example["input_ids"] for example in examples])
    attention_mask = torch.stack([example["attention_mask"] for example in examples])
    
    return {
        "pixel_values": pixel_values,
        "masks": masks,
        "masked_images": masked_images,
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }


class OrthogonalSemanticGuidance:
    """OSG implementation"""
    
    def __init__(self, lambda_orth=0.1, lambda_norm=0.01, epsilon=0.05, use_dynamic=True):
            self.lambda_orth = lambda_orth
            self.lambda_norm = lambda_norm 
            self.epsilon = epsilon
            self.use_dynamic = use_dynamic
            self.demographic_pairs = []
            
            if use_dynamic:
                self._init_dynamic_pairs()
            else:
                self._init_hardcoded_pairs()
    
    def _init_dynamic_pairs(self):
        """Generate all pairs with dynamically determined baselines"""
        all_demographics = [
            "An Asian man", "An Asian woman", 
            "A White man", "A White woman",
            "A Black man", "A Black woman",
            "A Latino man", "A Latino woman",
            "A Middle Eastern man", "A Middle Eastern woman"
        ]
        
        all_demographics.sort()
        
        for i in range(len(all_demographics)):
            for j in range(i + 1, len(all_demographics)):
                p_i = all_demographics[i]
                p_j = all_demographics[j]
                baseline = self._extract_shared_concept(p_i, p_j)
                
                self.demographic_pairs.append({
                    "p_i": p_i,
                    "p_j": p_j,
                    "baseline": baseline
                })      
                
    def _init_hardcoded_pairs(self):
        """Hardcoded pairs for comparing specific demographics only"""
        self.demographic_pairs = []

        # Baseline: "man"
        self.demographic_pairs.extend([
            {"p_i": "An Asian man",           "p_j": "A White man", "baseline": "a man"},
            {"p_i": "A Black man",            "p_j": "A White man", "baseline": "a man"},
            {"p_i": "A Latino man",           "p_j": "A White man", "baseline": "a man"},
            {"p_i": "A Middle Eastern man",   "p_j": "A White man", "baseline": "a man"},
        ])

        # Baseline: "woman"
        self.demographic_pairs.extend([
            {"p_i": "An Asian woman",         "p_j": "A White woman", "baseline": "a woman"},
            {"p_i": "A Black woman",          "p_j": "A White woman", "baseline": "a woman"},
            {"p_i": "A Latino woman",         "p_j": "A White woman", "baseline": "a woman"},
            {"p_i": "A Middle Eastern woman", "p_j": "A White woman", "baseline": "a woman"},
        ])

        # Cross-Gender Comparisons
        self.demographic_pairs.extend([
            {"p_i": "An Asian man", "p_j": "An Asian woman", "baseline": "Asian"},
            {"p_i": "A White man",  "p_j": "A White woman",  "baseline": "White"},
            {"p_i": "A Black man",  "p_j": "A Black woman",  "baseline": "Black"},
            {"p_i": "A Latino man", "p_j": "A Latino woman", "baseline": "Latino"},
            {"p_i": "A Middle Eastern man", "p_j": "A Middle Eastern woman", "baseline": "Middle Eastern"},
        ])
        
    def _extract_shared_concept(self, text1, text2):
            """
            Extract shared baseline via word intersection.
            """
            t1_lower = text1.lower()
            t2_lower = text2.lower()
            
            if "middle eastern" in t1_lower and "middle eastern" in t2_lower:
                return "Middle Eastern"
                
            words1 = set(t1_lower.replace(",", "").replace(".", "").split())
            words2 = set(t2_lower.replace(",", "").replace(".", "").split())
            
            stopwords = {'a', 'an', 'the', 'with', 'of'}
            words1 -= stopwords
            words2 -= stopwords
            
            common = words1 & words2
            
            if not common:
                return "a person"
            
            priority_terms = {'man', 'woman', 'boy', 'girl', 'male', 'female',
                            'asian', 'white', 'black', 'latino', 'person'}
            
            priority_common = common & priority_terms
            
            if priority_common:
                if 'man' in priority_common: return "a man"
                if 'woman' in priority_common: return "a woman"
                if 'person' in priority_common: return "a person"
                return sorted(list(priority_common))[0]
            
            return sorted(list(common))[0]
    
    def _encode_text(self, text, text_encoder, tokenizer, device):
        """Encode text to embedding"""
        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        )
        
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)
        
        output = text_encoder(input_ids, attention_mask=attention_mask)
        
        if hasattr(output, 'pooler_output') and output.pooler_output is not None:
            return output.pooler_output
        else:
            hidden = output[0]
            mask = attention_mask.unsqueeze(-1)
            masked = hidden * mask
            return masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    
    def compute_loss(self, text_encoder, tokenizer, device):
        """
        Compute OSG loss.
        """
        orth_loss = torch.tensor(0.0, device=device)
        norm_loss = torch.tensor(0.0, device=device)
        
        unique_vectors = {}
        
        all_texts = set()
        for pair in self.demographic_pairs:
            all_texts.add(pair["p_i"])
            all_texts.add(pair["p_j"])
            all_texts.add(pair["baseline"])
            
        for text in all_texts:
            unique_vectors[text] = self._encode_text(text, text_encoder, tokenizer, device)
            
        penalized_deltas = set() 
        
        for pair in self.demographic_pairs:
            p_i = pair["p_i"]
            p_j = pair["p_j"]
            base = pair["baseline"]
            
            e_i = unique_vectors[p_i]
            e_j = unique_vectors[p_j]
            e_base = unique_vectors[base]
            
            delta_i = e_i - e_base
            delta_j = e_j - e_base
            
            cos_sim = F.cosine_similarity(delta_i, delta_j, dim=-1)
            orth_loss += F.relu(cos_sim.abs() - self.epsilon) ** 2
                        
            key_i = f"{p_i}_{base}"
            if key_i not in penalized_deltas:
                norm_loss += (delta_i.norm(dim=-1) - 1) ** 2
                penalized_deltas.add(key_i)
                
            key_j = f"{p_j}_{base}"
            if key_j not in penalized_deltas:
                norm_loss += (delta_j.norm(dim=-1) - 1) ** 2
                penalized_deltas.add(key_j)
                
        if orth_loss.dim() > 0:
            orth_loss = orth_loss.mean()
        if norm_loss.dim() > 0:
            norm_loss = norm_loss.mean()
        
        total_loss = self.lambda_orth * orth_loss + self.lambda_norm * norm_loss
        
        return total_loss, orth_loss.item(), norm_loss.item()

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Model
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-2-inpainting")
    
    # Data
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with Asian face images")
    parser.add_argument("--resolution", type=int, default=512)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="fairanon_stage1_output")
    
    # Training
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=15000)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    
    # OSG hyperparameters
    parser.add_argument("--lambda_orth", type=float, default=0.1)
    parser.add_argument("--lambda_norm", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=0.05)
    
    # Optimization
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    
    # Logging
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=500)

    # OSG mode
    parser.add_argument("--use_dynamic_baseline", action="store_true", help="Use dynamic baseline. If not set, uses hardcoded pairs for efficiency.")

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup accelerator
    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs")
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=project_config
    )
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    set_seed(args.seed)
    
    # Load models
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet")
    scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    
    # Freeze VAE and set to eval mode
    vae.requires_grad_(False)
    vae.eval()
    
    # Text encoder: trainable
    text_encoder.requires_grad_(True)
    
    # UNet: only cross-attention layers trainable
    unet.requires_grad_(False)
    for name, param in unet.named_parameters():
        if "attn2" in name and any(k in name for k in ["to_k", "to_v", "to_q", "to_out"]):
            param.requires_grad = True
    
    if args.gradient_checkpointing:
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()
    
    # Initialize OSG
    osg = OrthogonalSemanticGuidance(
        lambda_orth=args.lambda_orth,
        lambda_norm=args.lambda_norm,
        epsilon=args.epsilon,
        use_dynamic=args.use_dynamic_baseline  

    )
    
    # Optimizers
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("Please install bitsandbytes: pip install bitsandbytes")
    else:
        optimizer_class = torch.optim.AdamW
    
    text_encoder_optimizer = optimizer_class(
        text_encoder.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8
    )
    
    unet_params = [p for p in unet.parameters() if p.requires_grad]
    unet_optimizer = optimizer_class(
        unet_params,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8
    )
    
    # Dataset and dataloader with generator for reproducibility
    dataset = FairAnonDataset(args.data_dir, tokenizer, args.resolution)
    
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        generator=generator,
        pin_memory=True
    )
    
    # Prepare for training
    text_encoder, unet = accelerator.prepare(text_encoder, unet)
    text_encoder_optimizer, unet_optimizer = accelerator.prepare(text_encoder_optimizer, unet_optimizer)
    dataloader = accelerator.prepare(dataloader)
    
    # Move VAE to device
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    vae.to(accelerator.device, dtype=weight_dtype)
    
    # Training loop
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    if args.max_train_steps is not None:
        max_train_steps = min(max_train_steps, args.max_train_steps)
    
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Training")
    
    global_step = 0
    
    for epoch in range(args.num_train_epochs):
        text_encoder.train()
        unet.train()
        
        for step, batch in enumerate(dataloader):
            # Move batch tensors to device with proper dtype
            device = accelerator.device
            pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
            masked_images = batch["masked_images"].to(device, dtype=weight_dtype)
            masks = batch["masks"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Update Text Encoder with OSG
            with accelerator.accumulate(text_encoder):
                osg_loss, orth_val, norm_val = osg.compute_loss(
                    text_encoder, tokenizer, device
                )
                accelerator.backward(osg_loss)
                text_encoder_optimizer.step()
                text_encoder_optimizer.zero_grad()
            
            # Update U-Net
            with accelerator.accumulate(unet):
                # Encode images to latent space
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    
                    masked_latents = vae.encode(masked_images).latent_dist.sample()
                    masked_latents = masked_latents * vae.config.scaling_factor
                
                # Prepare mask for latent space with proper dtype
                masks_latent = F.interpolate(masks, size=(64, 64), mode="nearest")
                masks_latent = masks_latent.to(dtype=latents.dtype)
                
                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=device
                ).long()
                
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                
                # Concatenate for inpainting model (9 channels)
                latent_model_input = torch.cat([noisy_latents, masks_latent, masked_latents], dim=1)
                
                # Get text embeddings
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(
                        input_ids,
                        attention_mask=attention_mask
                    )[0]
                
                # Predict noise
                noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample
                
                # Loss
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                
                accelerator.backward(loss)
                unet_optimizer.step()
                unet_optimizer.zero_grad()
            
            # Logging
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if global_step % args.logging_steps == 0:
                    logs = {
                        "osg_loss": osg_loss.item(),
                        "orthogonality": orth_val,
                        "normalization": norm_val,
                        "diffusion_loss": loss.item(),
                    }
                    accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(**logs)
                
                if global_step % args.save_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        
                        pipeline = StableDiffusionInpaintPipeline(
                            vae=vae,
                            text_encoder=accelerator.unwrap_model(text_encoder),
                            tokenizer=tokenizer,
                            unet=accelerator.unwrap_model(unet),
                            scheduler=scheduler,
                            safety_checker=None,
                            feature_extractor=None,
                            requires_safety_checker=False
                        )
                        pipeline.save_pretrained(save_path)
                        logger.info(f"Saved checkpoint to {save_path}")
                
                if global_step >= max_train_steps:
                    break
        
        if global_step >= max_train_steps:
            break
    
    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline = StableDiffusionInpaintPipeline(
            vae=vae,
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            unet=accelerator.unwrap_model(unet),
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False
        )
        pipeline.save_pretrained(args.output_dir)
        print(f"Training complete! Model saved to {args.output_dir}")
    
    accelerator.end_training()


if __name__ == "__main__":
    main()