import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file, save_file

from krea2edit.latents import latent_tokens
from krea2edit.weight_names import to_original_module


LORA_INITIALIZATIONS = {"gaussian", "nora_init"}


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


def quantize_component(model: nn.Module, config: dict):
    backend = config["backend"]
    if backend == "none":
        model.requires_grad_(False)
        return model
    if backend != "quanto":
        raise ValueError(f"unsupported component quantization backend: {backend}")
    from optimum.quanto import freeze, qfloat8, qint4, qint8, quantize

    qtype = {"qint4": qint4, "qint8": qint8, "qfloat8": qfloat8}[config["weights"]]
    # Quanto kernels accept 2D/3D inputs; upstream text fusion feeds its
    # tiny layer-axis projector a 4D tensor, so leave that projector unquantized.
    quantize(model, weights=qtype, exclude=["text_fusion.projector", "virtual_embedding"])
    freeze(model)
    return model


@torch.no_grad()
def apply_nora_init(model: nn.Module, epsilon: float = 1e-12) -> int:
    """Normalize every LoRA A column along its rank dimension once."""
    initialized = 0
    for name, parameter in model.named_parameters():
        if ".lora_A." not in name or not name.endswith(".weight"):
            continue
        if parameter.ndim != 2:
            raise RuntimeError(
                f"NoRA-init expected a matrix for {name}, got {tuple(parameter.shape)}"
            )
        value = parameter.detach().float()
        column_norms = torch.linalg.vector_norm(
            value, ord=2, dim=0, keepdim=True
        ).clamp_min(epsilon)
        parameter.copy_((value / column_norms).to(dtype=parameter.dtype))
        initialized += 1
    if initialized == 0:
        raise RuntimeError("NoRA-init found no PEFT LoRA A matrices")
    return initialized


def add_lora(model: nn.Module, config: dict):
    initialization = str(config.get("initialization", "gaussian")).lower()
    initialization = initialization.replace("-", "_")
    if initialization not in LORA_INITIALIZATIONS:
        choices = ", ".join(sorted(LORA_INITIALIZATIONS))
        raise ValueError(f"lora.initialization must be one of: {choices}")
    target_modules = config.get("target_modules", "all-linear")
    if target_modules == "all-linear":
        target_modules = [
            name
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        ]
    lora = LoraConfig(
        r=int(config["rank"]),
        lora_alpha=int(config["alpha"]),
        lora_dropout=float(config.get("dropout", 0.0)),
        target_modules=target_modules,
        bias="none",
        init_lora_weights="gaussian",
        modules_to_save=["virtual_embedding"],
    )
    peft_model = get_peft_model(model, lora)
    if initialization == "nora_init":
        apply_nora_init(peft_model)
    return peft_model


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


class RaggedEditModel(nn.Module):
    def __init__(
        self,
        dit,
        representation_projection: str = "none",
        projection_hidden_dim: int = 1024,
        patch: int = 2,
    ):
        super().__init__()
        self.dit = dit
        self.patch = patch
        self.depth = dit.config.num_layers
        self.features = dit.config.attention_head_dim * dit.config.num_attention_heads
        if representation_projection == "none":
            self.representation_projector = None
        elif representation_projection == "identity":
            self.representation_projector = nn.Identity()
        elif representation_projection == "mlp":
            if projection_hidden_dim <= 0:
                raise ValueError("projection_hidden_dim must be > 0")
            anchor = next(
                parameter for parameter in dit.parameters() if parameter.requires_grad
            )
            self.representation_projector = nn.Sequential(
                nn.Linear(self.features, projection_hidden_dim),
                nn.SiLU(),
                nn.Linear(projection_hidden_dim, self.features),
            ).to(device=anchor.device, dtype=anchor.dtype)
        else:
            raise ValueError("representation_projection must be none, identity, or mlp")

    def forward(
        self,
        noisy_targets: list[torch.Tensor],
        references: list[list[torch.Tensor]],
        contexts: list[torch.Tensor],
        timesteps: torch.Tensor,
        alternate_timesteps: torch.Tensor | None = None,
        alternate_target_masks: list[torch.Tensor] | None = None,
        representation_layer: int | None = None,
        project_representation: bool = False,
        return_velocity: bool = True,
        kv_cache: dict | None = None,
        drop_reference_attention: list[bool] | None = None,
    ):
        patch = self.patch
        cached = bool(kv_cache)
        image_tokens, positions, reference_counts = [], [], []
        for target, sample_refs in zip(noisy_targets, references):
            ref_tokens, ref_positions = [], []
            if not cached:
                for frame, latent in enumerate(sample_refs, start=1):
                    tokens, pos = latent_tokens(latent, patch, frame)
                    ref_tokens.append(tokens)
                    ref_positions.append(pos)
            target_tokens, target_positions = latent_tokens(target, patch, 0)
            image_tokens.append(torch.cat(ref_tokens + [target_tokens]))
            positions.append(torch.cat(ref_positions + [target_positions]))
            reference_counts.append(
                sum(
                    (ref.shape[-2] // patch) * (ref.shape[-1] // patch)
                    for ref in sample_refs
                )
            )

        result = self.dit(
            hidden_states=image_tokens,
            encoder_hidden_states=contexts,
            timestep=timesteps,
            position_ids=positions,
            reference_token_counts=reference_counts,
            drop_reference_attention=drop_reference_attention,
            alternate_timestep=alternate_timesteps,
            alternate_target_masks=alternate_target_masks,
            representation_layer=representation_layer,
            return_velocity=return_velocity,
            kv_cache=kv_cache,
        )
        if representation_layer is None:
            return result.sample
        predictions, representations = result.sample, result.representations

        if project_representation:
            if self.representation_projector is None:
                raise RuntimeError("no representation projector is configured")
            representations = [
                self.representation_projector(value) for value in representations
            ]
        return predictions, representations


def target_velocity_tokens(clean: torch.Tensor, noise: torch.Tensor, patch: int):
    return latent_tokens(noise - clean, patch, 0)[0]
