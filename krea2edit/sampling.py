import math
import textwrap
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.transforms.functional import to_pil_image


class PreviewDatasetView:
    """ID-addressable view that only indexes the configured preview samples."""

    def __init__(self, dataset, sample_ids: list[str]):
        self.dataset = dataset
        wanted = set(sample_ids)
        self.indices_by_id = {}
        if not wanted:
            return
        for index, sample in enumerate(dataset.samples):
            if sample["id"] in wanted:
                self.indices_by_id[sample["id"]] = index
                if len(self.indices_by_id) == len(wanted):
                    break

    def get_by_id(self, sample_id: str):
        return self.dataset[self.indices_by_id[sample_id]]


@torch.inference_mode()
def sample_edit(
    model,
    conditioning,
    prompt,
    references,
    width,
    height,
    steps,
    guidance_scale,
    seed,
    negative_prompt="",
    schedule_mu=None,
    drop_reference_attention=False,
):
    return conditioning(
        prompt=prompt,
        references=references,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator(device=conditioning.device).manual_seed(seed),
        negative_prompt=negative_prompt,
        mu=schedule_mu,
        edit_model=model,
        drop_reference_attention=drop_reference_attention,
    ).images[0]


def render_preview_sheet(
    references: list[torch.Tensor],
    output: Image.Image,
    target: torch.Tensor,
    prompt: str,
):
    """Render the prompt above ``[reference montage | generated | ground truth]``."""
    output = output.convert("RGB")
    panel_width, panel_height = output.size
    input_panel = Image.new("RGB", output.size, (24, 24, 24))
    columns = max(1, math.ceil(math.sqrt(len(references))))
    rows = math.ceil(len(references) / columns)
    cell_width = panel_width // columns
    cell_height = panel_height // rows

    for index, reference in enumerate(references):
        image = to_pil_image(reference.detach().cpu().clamp(0, 1)).convert("RGB")
        image = ImageOps.contain(image, (cell_width, cell_height))
        x = (index % columns) * cell_width + (cell_width - image.width) // 2
        y = (index // columns) * cell_height + (cell_height - image.height) // 2
        input_panel.paste(image, (x, y))

    target_image = to_pil_image(target.detach().cpu().clamp(0, 1)).convert("RGB")
    target_image = ImageOps.contain(target_image, (panel_width, panel_height))
    target_panel = Image.new("RGB", output.size, (24, 24, 24))
    target_panel.paste(
        target_image,
        (
            (panel_width - target_image.width) // 2,
            (panel_height - target_image.height) // 2,
        ),
    )

    body = Image.new("RGB", (panel_width * 3, panel_height), "black")
    body.paste(input_panel, (0, 0))
    body.paste(output, (panel_width, 0))
    body.paste(target_panel, (panel_width * 2, 0))

    font_size = max(18, min(32, body.width // 48))
    font = ImageFont.load_default(size=font_size)
    lines = textwrap.wrap(
        str(prompt), width=max(24, int(body.width / (font_size * 0.6)))
    ) or [""]
    header_height = 20 + (font_size + 6) * len(lines)
    sheet = Image.new("RGB", (body.width, header_height + body.height), "white")
    ImageDraw.Draw(sheet).multiline_text(
        (12, 10), "\n".join(lines), fill="black", font=font, spacing=4
    )
    sheet.paste(body, (0, header_height))
    return sheet


def render_preview_grid(sheets: list[Image.Image]):
    columns = math.ceil(math.sqrt(len(sheets)))
    rows = math.ceil(len(sheets) / columns)
    cell_width = max(sheet.width for sheet in sheets)
    cell_height = max(sheet.height for sheet in sheets)
    grid = Image.new("RGB", (columns * cell_width, rows * cell_height), (16, 16, 16))
    for index, sheet in enumerate(sheets):
        x = (index % columns) * cell_width + (cell_width - sheet.width) // 2
        y = (index // columns) * cell_height + (cell_height - sheet.height) // 2
        grid.paste(sheet, (x, y))
    return grid, rows, columns


def generate_previews(
    model,
    conditioning,
    dataset,
    config: dict,
    output_dir: Path,
    step: int,
):
    step_dir = output_dir / "samples" / f"step-{step:08d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = []
    sheets = []
    was_training = model.training
    model.eval()
    try:
        for specification in config["samples"]:
            sample = dataset.get_by_id(specification["id"])
            height, width = sample["target"].shape[-2:]
            seed = int(specification.get("seed", 42))
            output = sample_edit(
                model=model,
                conditioning=conditioning,
                prompt=sample["prompt"],
                references=sample["references"],
                width=width,
                height=height,
                steps=int(config["steps"]),
                guidance_scale=float(config["guidance_scale"]),
                seed=seed,
                negative_prompt=str(config.get("negative_prompt", "")),
                schedule_mu=config.get("schedule_mu"),
                drop_reference_attention=specification.get(
                    "drop_reference_attention", config.get("drop_reference_attention", False)
                ),
            )
            sheet = render_preview_sheet(
                sample["references"], output, sample["target"], sample["prompt"]
            )
            sample_ids.append(sample["id"])
            sheets.append(sheet)
    finally:
        model.train(was_training)

    grid, rows, columns = render_preview_grid(sheets)
    path = step_dir / "preview-grid.webp"
    grid.save(path, format="WEBP", quality=80)
    return {
        "path": path,
        "rows": rows,
        "columns": columns,
        "sample_ids": sample_ids,
    }
