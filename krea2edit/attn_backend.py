"""Attention backends and masks for reference-conditioned Krea2."""

from dataclasses import dataclass
from functools import cache

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_krea2 import Krea2AttnProcessor
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)


_create_block_mask = torch.compile(create_block_mask, dynamic=True)
_flex_attention = torch.compile(flex_attention, dynamic=True)


@dataclass
class PrefixAttention:
    # Full: [references | virtual | text | target]. Cached queries omit references.
    reference_lengths: list[int]
    key_lengths: list[int]
    query_lengths: list[int]
    segments: list[list[tuple[int, int, int]]]
    cached: bool
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    key_starts: torch.Tensor
    key_ends: torch.Tensor
    flex_mask: BlockMask | None = None


def build_prefix_attention(
    reference_lengths,
    virtual_lengths,
    text_lengths,
    target_lengths,
    drop_reference_attention,
    device,
    cached=False,
):
    """Store each query's visible key interval [start, end) in O(tokens)."""
    key_lengths, query_lengths, sample_segments = [], [], []
    key_starts, key_ends = [], []
    for reference, virtual, text, target, drop in zip(
        reference_lengths,
        virtual_lengths,
        text_lengths,
        target_lengths,
        drop_reference_attention,
    ):
        total = reference + virtual + text + target
        key_lengths.append(total)
        query_lengths.append(total - reference if cached else total)
        segments = []
        if not cached:
            segments.append((reference, 0, reference))
        segments.extend(
            [
                (virtual, 0, reference + virtual),
                # Cutting references keeps virtual/text/target attention intact.
                (text + target, reference if drop else 0, total),
            ]
        )
        segments = [segment for segment in segments if segment[0]]
        sample_segments.append(segments)
        for length, start, end in segments:
            key_starts.append(
                torch.full((length,), start, device=device, dtype=torch.int32)
            )
            key_ends.append(
                torch.full((length,), end, device=device, dtype=torch.int32)
            )
    return PrefixAttention(
        reference_lengths,
        key_lengths,
        query_lengths,
        sample_segments,
        cached,
        torch.tensor([0] + query_lengths, device=device, dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        ),
        torch.tensor([0] + key_lengths, device=device, dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        ),
        torch.cat(key_starts),
        torch.cat(key_ends),
    )


def sdpa_prefix_attention(q, k, v, layout):
    """Exact segmented SDPA without a dense full-sequence attention mask."""
    outputs = []
    query_offset = key_offset = 0
    for segments, key_length in zip(layout.segments, layout.key_lengths):
        for query_length, start, end in segments:
            query_end = query_offset + query_length
            key_start, key_end = key_offset + start, key_offset + end
            outputs.append(
                F.scaled_dot_product_attention(
                    q[query_offset:query_end].transpose(0, 1).unsqueeze(0),
                    k[key_start:key_end].transpose(0, 1).unsqueeze(0),
                    v[key_start:key_end].transpose(0, 1).unsqueeze(0),
                    enable_gqa=q.shape[1] != k.shape[1],
                )
                .squeeze(0)
                .transpose(0, 1)
            )
            query_offset = query_end
        key_offset += key_length
    return torch.cat(outputs)


def flex_prefix_attention(q, k, v, layout):
    if layout.flex_mask is None:
        # One mask per model forward, reused across all transformer layers.
        starts, offset = [], 0
        for query_length, key_length in zip(layout.query_lengths, layout.key_lengths):
            starts.append(
                torch.full((query_length,), offset, device=q.device, dtype=torch.int32)
            )
            offset += key_length
        sample_starts = torch.cat(starts)
        starts = sample_starts + layout.key_starts
        ends = sample_starts + layout.key_ends

        def mask_mod(batch, head, q_idx, kv_idx):
            # BlockMask construction can evaluate padded tile positions.
            query = q_idx.clamp(max=starts.shape[0] - 1)
            return (kv_idx >= starts[query]) & (kv_idx < ends[query])

        layout.flex_mask = _create_block_mask(
            mask_mod, None, None, q.shape[0], k.shape[0], device=q.device
        )
    return (
        _flex_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            block_mask=layout.flex_mask,
            enable_gqa=q.shape[1] != k.shape[1],
        )
        .squeeze(0)
        .transpose(0, 1)
    )


@cache
def _get_fa4_mask():
    # Load CuTe only when the FA4 backend is used.
    import cutlass
    import cutlass.cute as cute

    @cute.jit
    def prefix_mask(batch, head, q_idx, kv_idx, seqlen_info, aux_tensors):
        # FA4 uses sample-local positions; aux data uses packed global offsets.
        # Clamp tile-tail queries; FA4 masks padded lanes.
        query = cute.make_rmem_tensor(1, cutlass.Int32)
        query.store(q_idx)
        key_start = cute.make_rmem_tensor(1, cutlass.Int32)
        key_end = cute.make_rmem_tensor(1, cutlass.Int32)
        packed_query = seqlen_info.offset_q + cutlass.min(
            query[0], seqlen_info.seqlen_q - 1
        )
        key_start[0] = aux_tensors[0][packed_query]
        key_end[0] = aux_tensors[1][packed_query]
        return (kv_idx >= key_start.load()) & (kv_idx < key_end.load())

    return prefix_mask


def prefix_attention(q, k, v, layout):
    # Current FA4 supports mask_mod backward on SM90/100/110, but not SM120.
    if (
        q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and torch.cuda.get_device_capability(q.device)[0] in (9, 10, 11)
    ):
        try:
            from flash_attn.cute import flash_attn_varlen_func
        except ImportError:
            pass
        else:
            output = flash_attn_varlen_func(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                cu_seqlens_q=layout.cu_seqlens_q,
                cu_seqlens_k=layout.cu_seqlens_k,
                max_seqlen_q=max(layout.query_lengths),
                max_seqlen_k=max(layout.key_lengths),
                mask_mod=_get_fa4_mask(),
                aux_tensors=[layout.key_starts, layout.key_ends],
                causal=False,
            )
            return output[0] if isinstance(output, tuple) else output
    if (
        q.is_cuda
        and torch.cuda.get_device_capability(q.device)[0] >= 8
        and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return flex_prefix_attention(q, k, v, layout)
    return sdpa_prefix_attention(q, k, v, layout)


class PrefixKrea2AttnProcessor(Krea2AttnProcessor):
    def __init__(self, layer_index):
        self.layer_index = layer_index

    def __call__(
        self,
        attn,
        hidden_states,
        attention_mask=None,
        image_rotary_emb=None,
        kv_cache=None,
    ):
        if not isinstance(attention_mask, PrefixAttention):
            return super().__call__(
                attn, hidden_states, attention_mask, image_rotary_emb
            )
        query = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        value = attn.to_v(hidden_states).unflatten(
            -1, (attn.num_kv_heads, attn.head_dim)
        )
        query, key = attn.norm_q(query), attn.norm_k(key)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        query, key, value = query.squeeze(0), key.squeeze(0), value.squeeze(0)
        if kv_cache is not None:
            if attention_mask.cached:
                # Virtual tokens, text and target are freshly computed.
                # Restore only the raw reference K/V before them.
                merged = []
                for tensor, references in zip((key, value), kv_cache[self.layer_index]):
                    chunks = tensor.split(attention_mask.query_lengths)
                    merged.append(
                        torch.cat(
                            [
                                torch.cat((reference, chunk))
                                for chunk, reference in zip(chunks, references)
                            ]
                        )
                    )
                key, value = merged
            else:
                # References read only references at t=0, so their K/V do not
                # depend on the changing text, target or denoising time.
                kv_cache[self.layer_index] = tuple(
                    [
                        chunk[:reference].clone()
                        for chunk, reference in zip(
                            tensor.split(attention_mask.key_lengths),
                            attention_mask.reference_lengths,
                        )
                    ]
                    for tensor in (key, value)
                )
        output = (
            prefix_attention(query, key, value, attention_mask).flatten(1).unsqueeze(0)
        )
        return attn.to_out[0](output * torch.sigmoid(attn.to_gate(hidden_states)))
