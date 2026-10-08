"""Weight names shared by original Krea checkpoints, Diffusers and LoRA exports."""

_PREFIXES = (
    ("txt_in.linear_1", "txtmlp.1"),
    ("txt_in.linear_2", "txtmlp.3"),
    ("txt_in.norm", "txtmlp.0"),
    ("time_embed.linear_1", "tmlp.0"),
    ("time_embed.linear_2", "tmlp.2"),
    ("time_mod_proj", "tproj.1"),
    ("img_in", "first"),
    ("transformer_blocks", "blocks"),
    ("text_fusion", "txtfusion"),
    ("final_layer", "last"),
)
_PARTS = (
    (".attn.to_out.0", ".attn.wo"),
    (".attn.to_q", ".attn.wq"),
    (".attn.to_k", ".attn.wk"),
    (".attn.to_v", ".attn.wv"),
    (".attn.to_gate", ".attn.gate"),
    (".ff.", ".mlp."),
)


def _map_module(name, reverse):
    for pair in _PREFIXES:
        source, target = pair[::-1] if reverse else pair
        if name == source or name.startswith(source + "."):
            name = target + name[len(source) :]
            break
    for pair in _PARTS:
        source, target = pair[::-1] if reverse else pair
        name = name.replace(source, target)
    return name


def to_original_module(name):
    return _map_module(name, reverse=False)


def to_diffusers_module(name):
    return _map_module(name, reverse=True)


def convert_transformer_state_dict(checkpoint):
    """Rename original Krea 2 weights in memory and reshape modulation tables."""
    renames = {
        ".attn.qknorm.qnorm.scale": ".attn.norm_q.weight",
        ".attn.qknorm.knorm.scale": ".attn.norm_k.weight",
        ".prenorm.scale": ".norm1.weight",
        ".postnorm.scale": ".norm2.weight",
        ".norm.scale": ".norm.weight",
        ".mod.lin": ".scale_shift_table",
        ".modulation.lin": ".scale_shift_table",
    }
    converted = {}
    for key, value in checkpoint.items():
        key = to_diffusers_module(
            key.removeprefix("model.").removeprefix("diffusion_model.")
        )
        for old, new in renames.items():
            key = key.replace(old, new)
        if key.startswith("transformer_blocks.") and key.endswith(".scale_shift_table"):
            value = value.reshape(6, -1)
        converted[key] = value
    return converted
