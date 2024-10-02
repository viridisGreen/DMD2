import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
repo_name = "tianweiy/DMD2"
ckpt_name = "dmd2_sdxl_1step_unet_fp16.bin"

# Load model.
unet = UNet2DConditionModel.from_config(base_model_id, subfolder="unet").to("cuda:7", torch.float16)
unet.load_state_dict(torch.load(hf_hub_download(repo_name, ckpt_name), map_location="cuda:7"))
pipe = DiffusionPipeline.from_pretrained(base_model_id, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda:7")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

prompt="Westlake"
image=pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0, timesteps=[399]).images[0]
image.save(f"{prompt}.png")