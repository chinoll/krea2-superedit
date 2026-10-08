import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from krea2edit.latents import latent_tokens


LORA_INITIALIZATIONS = {"gaussian", "nora_init"}


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
