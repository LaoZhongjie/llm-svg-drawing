"""Stable Diffusion + LoRA model loading."""

from __future__ import annotations

from typing import Optional

import kagglehub
import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from safetensors.torch import load_file

from .config import default_device

_pipeline = None
_pipeline_device: Optional[str] = None


def load_generation_pipeline(
    device: Optional[str] = None,
    force_reload: bool = False,
    verbose: bool = True,
):
    global _pipeline, _pipeline_device

    target_device = device or default_device()
    if _pipeline is not None and _pipeline_device == target_device and not force_reload:
        return _pipeline

    base = kagglehub.model_download("stabilityai/stable-diffusion-xl/PyTorch/base-1-0/1")
    unet_state = kagglehub.model_download(
        "arnavkohli2005/sdxl-lightning/PyTorch/sdxl_lightning_4step_unet/1"
    )
    doctor_diffusion_vector_path = kagglehub.model_download(
        "crischir/doctor-diffusions-controllable-vector/Other/default/2"
    )

    scheduler = EulerDiscreteScheduler.from_pretrained(base, subfolder="scheduler")
    if verbose:
        print("Scheduler loaded successfully!")
        print(scheduler.config)

    unet_config = UNet2DConditionModel.load_config(base, subfolder="unet")
    unet_dtype = torch.float16 if target_device.startswith("cuda") else torch.float32
    unet = UNet2DConditionModel.from_config(unet_config).to(target_device, unet_dtype)
    state_dict = load_file(
        f"{unet_state}/sdxl_lightning_4step_unet.safetensors",
        device="cpu",
    )
    unet.load_state_dict(state_dict)

    pipe_kwargs = {
        "unet": unet,
        "scheduler": scheduler,
        "use_safetensors": True,
        "safety_checker": None,
    }
    if target_device.startswith("cuda"):
        pipe_kwargs["torch_dtype"] = torch.float16
        pipe_kwargs["variant"] = "fp16"
    else:
        pipe_kwargs["torch_dtype"] = torch.float32

    pipe = StableDiffusionXLPipeline.from_pretrained(base, **pipe_kwargs)
    pipe.to(target_device)
    pipe.load_lora_weights(
        doctor_diffusion_vector_path,
        weight_name="DD-vector-v2.safetensors",
    )

    _pipeline = pipe
    _pipeline_device = target_device
    return _pipeline


def get_generation_pipeline(device: Optional[str] = None):
    return load_generation_pipeline(device=device, verbose=False)
