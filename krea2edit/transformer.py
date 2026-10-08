"""Editing extensions for Diffusers' Krea2Transformer2DModel.

Diffusers supplies the model layers. Editing adds learned virtual embeddings,
reference KV caching, per-sample attention dropout and token flow times.
"""

from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F
from diffusers import Krea2Transformer2DModel
from diffusers.configuration_utils import register_to_config

# Diffusers discovers custom component base classes through module exports.
from diffusers import ModelMixin as ModelMixin
from diffusers.utils import BaseOutput
from safetensors.torch import load_file

from krea2edit.attn_backend import (
    PrefixKrea2AttnProcessor,
    build_prefix_attention,
)
from krea2edit.weight_names import convert_transformer_state_dict


@dataclass
class Krea2EditTransformerOutput(BaseOutput):
    sample: list[torch.Tensor] | None
    representations: list[torch.Tensor] | None = None


def _edit_block(block, hidden, time_mod, time_ids, rotary, attention, kv_cache=None):
    # Keep modulation compact (one vector per flow time), expanding one term
    # at a time instead of storing a (tokens, 6 * hidden_size) time embedding.
    values = (time_mod.unflatten(-1, (6, -1)) + block.scale_shift_table).unbind(-2)
    pre = (1 + values[0][time_ids]) * block.norm1(hidden) + values[1][time_ids]
    update = block.attn(
        pre, attention_mask=attention, image_rotary_emb=rotary, kv_cache=kv_cache
    )
    hidden = hidden + values[2][time_ids] * update
    post = (1 + values[3][time_ids]) * block.norm2(hidden) + values[4][time_ids]
    return hidden + values[5][time_ids] * block.ff(post)


class Krea2EditTransformer2DModel(Krea2Transformer2DModel):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 64,
        num_layers: int = 28,
        attention_head_dim: int = 128,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 12,
        intermediate_size: int = 16384,
        timestep_embed_dim: int = 256,
        text_hidden_dim: int = 2560,
        num_text_layers: int = 12,
        text_num_attention_heads: int = 20,
        text_num_key_value_heads: int = 20,
        text_intermediate_size: int = 6912,
        num_layerwise_text_blocks: int = 2,
        num_refiner_text_blocks: int = 2,
        axes_dims_rope: tuple[int, int, int] = (32, 48, 48),
        rope_theta: float = 1000.0,
        norm_eps: float = 1e-5,
        num_virtual_tokens: int = 16,
    ):
        super().__init__(
            in_channels=in_channels,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            timestep_embed_dim=timestep_embed_dim,
            text_hidden_dim=text_hidden_dim,
            num_text_layers=num_text_layers,
            text_num_attention_heads=text_num_attention_heads,
            text_num_key_value_heads=text_num_key_value_heads,
            text_intermediate_size=text_intermediate_size,
            num_layerwise_text_blocks=num_layerwise_text_blocks,
            num_refiner_text_blocks=num_refiner_text_blocks,
            axes_dims_rope=axes_dims_rope,
            rope_theta=rope_theta,
            norm_eps=norm_eps,
        )
        self.virtual_embedding = torch.nn.Embedding(
            num_virtual_tokens, self.hidden_size
        )
        torch.nn.init.normal_(self.virtual_embedding.weight, std=0.02)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return_info = kwargs.pop("output_loading_info", False)
        model, info = super().from_pretrained(*args, output_loading_info=True, **kwargs)
        if "virtual_embedding.weight" in info["missing_keys"]:
            # Original Diffusers checkpoints do not contain the new embeddings.
            model.virtual_embedding = torch.nn.Embedding(
                model.config.num_virtual_tokens,
                model.hidden_size,
                device=model.device,
                dtype=model.dtype,
            )
            torch.nn.init.normal_(model.virtual_embedding.weight, std=0.02)
        return (model, info) if return_info else model

    @classmethod
    def from_original_checkpoint(
        cls,
        checkpoint_file,
        torch_dtype=torch.bfloat16,
        num_virtual_tokens=16,
    ):
        """Load raw.safetensors directly into Diffusers layers, without disk conversion."""
        state = convert_transformer_state_dict(
            load_file(str(checkpoint_file), device="cpu")
        )
        for name, value in state.items():
            keep_fp32 = any(
                part in cls._keep_in_fp32_modules for part in name.split(".")
            )
            state[name] = value.to(dtype=torch.float32 if keep_fp32 else torch_dtype)
        with torch.device("meta"):
            model = cls(num_virtual_tokens=num_virtual_tokens)
        state["virtual_embedding.weight"] = torch.empty(
            (num_virtual_tokens, model.hidden_size),
            dtype=torch_dtype,
        ).normal_(std=0.02)
        model.load_state_dict(state, strict=True, assign=True)
        return model.eval()

    # Processors carry no parameters and are installed lazily after loading.
    def _enable_prefix_attention(self):
        for index, block in enumerate(self.transformer_blocks):
            if not isinstance(block.attn.processor, PrefixKrea2AttnProcessor):
                block.attn.set_processor(PrefixKrea2AttnProcessor(index))

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        position_ids,
        encoder_attention_mask=None,
        attention_kwargs=None,
        return_dict=True,
        reference_token_counts=None,
        drop_reference_attention=None,
        alternate_timestep=None,
        alternate_target_masks=None,
        representation_layer=None,
        return_velocity=True,
        kv_cache=None,
    ):
        # Retain the standard Diffusers tensor API for text-to-image callers.
        if isinstance(hidden_states, torch.Tensor):
            return super().forward(
                hidden_states,
                encoder_hidden_states,
                timestep,
                position_ids,
                encoder_attention_mask,
                attention_kwargs,
                return_dict,
            )
        batch_size = len(hidden_states)
        if reference_token_counts is None:
            reference_token_counts = [0] * batch_size
        if drop_reference_attention is None:
            drop_reference_attention = [False] * batch_size

        self._enable_prefix_attention()
        cached = bool(kv_cache)
        device = hidden_states[0].device
        sequences, time_ids, target_indices = [], [], []
        full_positions, virtual_lengths, target_lengths = [], [], []
        offset = 0
        for index, (image, context, ref_count) in enumerate(
            zip(
                hidden_states,
                encoder_hidden_states,
                reference_token_counts,
            )
        ):
            target_length = image.shape[0] if cached else image.shape[0] - ref_count
            text_length = context.shape[0]
            virtual_length = self.config.num_virtual_tokens if ref_count else 0
            active_refs = 0 if cached else ref_count
            target_start = active_refs + virtual_length + text_length
            length = target_start + target_length
            image = self.img_in(image).unsqueeze(0)
            text = self.txt_in(self.text_fusion(context.unsqueeze(0)))
            virtual = (
                self.virtual_embedding(torch.arange(virtual_length, device=device))
                .to(image.dtype)
                .unsqueeze(0)
            )
            sequences.append(
                torch.cat(
                    (
                        image[:, :active_refs],
                        virtual,
                        text,
                        image[:, active_refs:],
                    ),
                    dim=1,
                )
            )
            positions = position_ids[index]
            full_positions.append(
                torch.cat(
                    (
                        positions[:active_refs],
                        positions.new_zeros((virtual_length + text_length, 3)),
                        positions[active_refs:],
                    )
                )
            )
            ids = torch.full((length,), index, device=device, dtype=torch.long)
            # Virtual tokens and text follow the target's sampled time.
            # Only raw references use t=0 and are cached during inference.
            ids[:active_refs] = batch_size + index
            if alternate_timestep is not None:
                mask = alternate_target_masks[index].to(device=device, dtype=torch.bool)
                ids[target_start:] = torch.where(
                    mask,
                    2 * batch_size + index,
                    index,
                )
            time_ids.append(ids)
            target_indices.append(
                torch.arange(
                    offset + target_start,
                    offset + length,
                    device=device,
                )
            )
            virtual_lengths.append(virtual_length)
            target_lengths.append(target_length)
            offset += length

        hidden = torch.cat(sequences, dim=1)
        ids = torch.cat(time_ids)
        targets = torch.cat(target_indices)
        times = [timestep, torch.zeros_like(timestep)]
        if alternate_timestep is not None:
            times.append(alternate_timestep)
        time_embed = self.time_embed(torch.cat(times), dtype=hidden.dtype).squeeze(1)
        time_mod = self.time_mod_proj(F.gelu(time_embed, approximate="tanh"))
        rotary = self.rotary_emb(torch.cat(full_positions))
        attention = build_prefix_attention(
            reference_token_counts,
            virtual_lengths,
            [context.shape[0] for context in encoder_hidden_states],
            target_lengths,
            drop_reference_attention,
            device,
            cached,
        )

        representations = None
        for index, block in enumerate(self.transformer_blocks, start=1):
            block_forward = partial(_edit_block, block, kv_cache=kv_cache)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden = self._gradient_checkpointing_func(
                    block_forward,
                    hidden,
                    time_mod,
                    ids,
                    rotary,
                    attention,
                )
            else:
                hidden = block_forward(hidden, time_mod, ids, rotary, attention)
            if index == representation_layer:
                representations = list(hidden[0, targets].split(target_lengths))
                if not return_velocity:
                    break

        predictions = None
        if return_velocity:
            # Reuse the upstream final layer, treating tokens as a batch so its
            # two modulation slots can broadcast independently per target time.
            output = self.final_layer(
                hidden[0, targets].unsqueeze(1),
                time_embed[ids[targets]].unsqueeze(1),
            ).squeeze(1)
            predictions = list(output.split(target_lengths))
        if not return_dict:
            return predictions, representations
        return Krea2EditTransformerOutput(predictions, representations)
