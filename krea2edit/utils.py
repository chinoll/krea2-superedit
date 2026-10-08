"""Checkpoint loading and adapter export utilities."""

import json
import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from peft import get_peft_model_state_dict
from safetensors.torch import load_file, save_file

from krea2edit.weight_names import to_original_module


def torch_dtype(name: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def checkpoint_path(name_or_path: str, filename: str | None):
    path = Path(name_or_path)
    if path.is_file():
        return path
    if path.is_dir():
        return path / filename if filename else next(path.glob("*.safetensors"))
    filename = (
        filename or f"{name_or_path.split('/')[-1].split('-')[-1].lower()}.safetensors"
    )
    return Path(
        hf_hub_download(name_or_path, filename=filename, token=os.getenv("HF_TOKEN"))
    )


def save_diffusers_lora(model, output_dir: Path):
    from diffusers import Krea2Pipeline

    state = {
        key.removeprefix("base_model.model."): value
        for key, value in get_peft_model_state_dict(model).items()
    }
    virtual_weight = state.pop("virtual_embedding.weight")
    metadata = model.peft_config["default"].to_dict()
    metadata["modules_to_save"] = None
    Krea2Pipeline.save_lora_weights(
        output_dir,
        transformer_lora_layers=state,
        transformer_lora_adapter_metadata=metadata,
    )
    save_file(
        {"weight": virtual_weight.contiguous()},
        str(output_dir / "virtual_embedding.safetensors"),
    )


def export_comfyui_lora(adapter_file: Path, output_file: Path):
    """Export original-name weights; virtual tokens require a matching custom node."""
    adapter_config = json.loads(
        adapter_file.with_name("adapter_config.json").read_text(encoding="utf-8")
    )
    alpha = int(adapter_config["lora_alpha"])
    peft_state = load_file(str(adapter_file), device="cpu")
    comfy_state = {}
    lora_modules = []
    for key, value in peft_state.items():
        key = "diffusion_model." + to_original_module(
            key.removeprefix("base_model.model.")
        )
        comfy_state[key] = value.contiguous()
        if key.endswith(".lora_A.weight"):
            lora_modules.append(key.removesuffix(".lora_A.weight"))

    for module_name in lora_modules:
        comfy_state[f"{module_name}.alpha"] = torch.tensor(float(alpha))

    save_file(
        comfy_state,
        str(output_file),
        metadata={
            "format": "pt",
            "model": "krea2",
            "architecture": (
                "krea2edit_virtual_tokens"
                if "diffusion_model.virtual_embedding.weight" in comfy_state
                else "krea2edit"
            ),
            "lora_alpha": str(alpha),
        },
    )
