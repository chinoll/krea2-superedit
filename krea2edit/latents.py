"""Shared patch layout for training and inference."""

import torch
from einops import rearrange


def latent_tokens(latent: torch.Tensor, patch: int, frame: int):
    _, height, width = latent.shape
    grid_h, grid_w = height // patch, width // patch
    tokens = rearrange(latent, "c (h ph) (w pw) -> (h w) (c ph pw)", ph=patch, pw=patch)
    y, x = torch.meshgrid(
        torch.arange(grid_h, device=latent.device, dtype=torch.float32),
        torch.arange(grid_w, device=latent.device, dtype=torch.float32),
        indexing="ij",
    )
    positions = torch.stack((torch.full_like(y, frame), y, x), dim=-1).reshape(
        grid_h * grid_w, 3
    )
    return tokens, positions


def velocity_tokens_to_latent(
    tokens: torch.Tensor,
    latent_height: int,
    latent_width: int,
    patch: int,
):
    channels = tokens.shape[-1] // (patch * patch)
    return rearrange(
        tokens,
        "(h w) (c ph pw) -> c (h ph) (w pw)",
        h=latent_height // patch,
        w=latent_width // patch,
        c=channels,
        ph=patch,
        pw=patch,
    )
