"""Krea 2 image editing with Diffusers loading, components and scheduling."""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    Krea2Pipeline,
)
from diffusers.pipelines.krea2.pipeline_krea2 import calculate_shift
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms.functional import to_pil_image, to_tensor
from transformers import AutoProcessor, AutoTokenizer, ProcessorMixin, Qwen3VLModel

from krea2edit.latents import velocity_tokens_to_latent
from krea2edit.transformer import Krea2EditTransformer2DModel


class Krea2EditPipeline(Krea2Pipeline):
    """One editing path shared by inference and training previews.

    The upstream pipeline supplies component registration, model loading, LoRA
    loading, offloading and the Krea prompt template. This class adds grounded
    prompts and clean reference latents; denoising uses the loaded scheduler.
    """

    _optional_components = ["processor"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLQwenImage,
        text_encoder: Qwen3VLModel,
        tokenizer: AutoTokenizer,
        transformer: Krea2EditTransformer2DModel,
        processor: ProcessorMixin = None,
        text_encoder_select_layers: tuple[int, ...] | list[int] | None = None,
        is_distilled: bool = False,
        patch_size: int = 2,
        grounding_max_px: int = 768,
        grounding_jitter_min: int = 384,
        max_prompt_tokens: int = 512,
    ):
        super().__init__(
            scheduler,
            vae,
            text_encoder,
            tokenizer,
            transformer,
            text_encoder_select_layers,
            is_distilled,
            patch_size,
        )
        self.register_modules(processor=processor)
        self.register_to_config(
            grounding_max_px=grounding_max_px,
            grounding_jitter_min=grounding_jitter_min,
            max_prompt_tokens=max_prompt_tokens,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load a Diffusers repository or convert a local original checkpoint in memory."""
        hub_kwargs = {
            key: kwargs[key]
            for key in (
                "token",
                "cache_dir",
                "revision",
                "local_files_only",
                "force_download",
            )
            if key in kwargs
        }
        processor_path = kwargs.pop("processor_name_or_path", None)
        text_encoder_path = kwargs.pop("text_encoder_name_or_path", None)
        vae_path = kwargs.pop("vae_name_or_path", None)
        num_virtual_tokens = kwargs.pop("num_virtual_tokens", None)
        virtual_kwargs = (
            {"num_virtual_tokens": num_virtual_tokens}
            if num_virtual_tokens is not None else {}
        )
        if Path(pretrained_model_name_or_path).is_file():
            dtype = kwargs.pop("torch_dtype", torch.bfloat16)
            for key in hub_kwargs:
                kwargs.pop(key)
            text_encoder_path = text_encoder_path or "Qwen/Qwen3-VL-4B-Instruct"
            return cls(
                transformer=Krea2EditTransformer2DModel.from_original_checkpoint(
                    pretrained_model_name_or_path, torch_dtype=dtype, **virtual_kwargs
                ),
                text_encoder=Qwen3VLModel.from_pretrained(
                    text_encoder_path, torch_dtype=dtype, **hub_kwargs
                ),
                tokenizer=AutoTokenizer.from_pretrained(
                    text_encoder_path, **hub_kwargs
                ),
                processor=AutoProcessor.from_pretrained(
                    processor_path or text_encoder_path, **hub_kwargs
                ),
                vae=AutoencoderKLQwenImage.from_pretrained(
                    vae_path or "Qwen/Qwen-Image",
                    subfolder="vae",
                    torch_dtype=dtype,
                    **hub_kwargs,
                ),
                scheduler=FlowMatchEulerDiscreteScheduler(
                    use_dynamic_shifting=True,
                    base_image_seq_len=256,
                    max_image_seq_len=6400,
                    base_shift=0.5,
                    max_shift=1.15,
                ),
                **kwargs,
            )
        config = cls.load_config(pretrained_model_name_or_path, **hub_kwargs)
        if "transformer" not in kwargs:
            model_kwargs = {
                key: kwargs[key]
                for key in (
                    "torch_dtype",
                    "variant",
                    "use_safetensors",
                    "low_cpu_mem_usage",
                )
                if key in kwargs
            }
            kwargs["transformer"] = Krea2EditTransformer2DModel.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="transformer",
                **hub_kwargs,
                **model_kwargs,
                **virtual_kwargs,
            )
        if "processor" not in kwargs and (
            processor_path is not None or config.get("_class_name") != cls.__name__
        ):
            # The text-only upstream tokenizer folder lacks Qwen's image
            # processor. Load that small component from the encoder's source.
            processor_kwargs = {
                key: value for key, value in hub_kwargs.items() if key != "revision"
            }
            kwargs["processor"] = AutoProcessor.from_pretrained(
                processor_path or "Qwen/Qwen3-VL-4B-Instruct",
                **processor_kwargs,
            )
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def load_edit_adapter(self, directory):
        """Load a local lora/ export, including its learned virtual embeddings."""
        directory = Path(directory)
        self.load_lora_weights(directory)
        weight = load_file(str(directory / "virtual_embedding.safetensors"))["weight"]
        self.transformer.virtual_embedding = torch.nn.Embedding.from_pretrained(
            weight.to(device=self.transformer.device, dtype=self.transformer.dtype),
            freeze=False,
        )
        self.transformer.register_to_config(num_virtual_tokens=weight.shape[0])

    def _grounding_image(self, image, jitter):
        cap = self.config.grounding_max_px
        if jitter and 0 < self.config.grounding_jitter_min < cap:
            cap = random.randint(self.config.grounding_jitter_min, cap)
        height, width = image.shape[-2:]
        if cap > 0 and max(height, width) > cap:
            scale = cap / max(height, width)
            image = F.interpolate(
                image.unsqueeze(0),
                size=(max(1, round(height * scale)), max(1, round(width * scale))),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return to_pil_image(image.cpu())

    @torch.no_grad()
    def encode_edit_prompt(self, prompt, references, grounding_jitter=True):
        """Ground language in reference images, then drop image-pad tokens."""
        if self.config.max_prompt_tokens > 0:
            ids = self.tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=self.config.max_prompt_tokens,
                padding=False,
            )["input_ids"]
            prompt = self.tokenizer.decode(
                ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        images = [
            self._grounding_image(image, grounding_jitter) for image in references
        ]
        vision = "<|vision_start|><|image_pad|><|vision_end|>" * len(images)
        text = (
            self.prompt_template_encode_prefix
            + vision
            + prompt
            + self.prompt_template_encode_suffix
        )
        inputs = self.processor(
            text=[text], images=images or None, return_tensors="pt"
        ).to(self._execution_device)
        outputs = self.text_encoder(
            **inputs, output_hidden_states=True, use_cache=False
        )
        hidden = torch.stack(
            [outputs.hidden_states[i] for i in self.text_encoder_select_layers], dim=2
        )[0]
        start = self.prompt_template_encode_start_idx
        input_ids = inputs["input_ids"][0, start:]
        return hidden[start:][input_ids != self.text_encoder.config.image_token_id].to(
            self.transformer.dtype
        )

    def _latent_stats(self, latent):
        mean = latent.new_tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = latent.new_tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1)
        return mean, std

    @torch.no_grad()
    def encode_image(self, image, sample_posterior=True):
        pixels = (
            image.to(self._execution_device, self.vae.dtype)
            .mul(2)
            .sub(1)[None, :, None]
        )
        posterior = self.vae.encode(pixels).latent_dist
        latent = posterior.sample() if sample_posterior else posterior.mode()
        mean, std = self._latent_stats(latent)
        return ((latent - mean) / std)[0, :, 0]

    def decode_latent_to_pixels(self, latent):
        """Decode to [-1,1], retaining latent gradients for perceptual training."""
        latent = latent.to(self._execution_device, self.vae.dtype)[None, :, None]
        mean, std = self._latent_stats(latent)
        return self.vae.decode(latent * std + mean).sample[0, :, 0].float().clamp(-1, 1)

    @torch.no_grad()
    def decode_image(self, latent):
        return self.decode_latent_to_pixels(latent).add(1).div(2)

    def _prepare_references(self, references):
        multiple = self.vae_scale_factor * self.patch_size
        prepared = []
        for image in references:
            if isinstance(image, Image.Image):
                image = to_tensor(image.convert("RGB"))
            height, width = image.shape[-2:]
            size = (
                max(multiple, round(height / multiple) * multiple),
                max(multiple, round(width / multiple) * multiple),
            )
            if size != (height, width):
                image = F.interpolate(
                    image[None].float(), size=size, mode="bilinear", align_corners=False
                )[0]
            prepared.append(image)
        return prepared

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        references: list,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 4.5,
        negative_prompt: str = "",
        generator: torch.Generator | None = None,
        mu: float | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        edit_model=None,
        use_kv_cache: bool = True,
        drop_reference_attention: bool = False,
    ):
        from krea2edit.modeling import RaggedEditModel

        references = self._prepare_references(references)
        context = self.encode_edit_prompt(prompt, references, grounding_jitter=False)
        negative = (
            self.encode_edit_prompt(negative_prompt, references, grounding_jitter=False)
            if guidance_scale > 0
            else None
        )
        reference_latents = [
            self.encode_image(image, sample_posterior=False) for image in references
        ]
        model = (
            edit_model
            if edit_model is not None
            else RaggedEditModel(self.transformer, patch=self.patch_size)
        )
        latent_height, latent_width = (
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        channels = self.transformer.config.in_channels // self.patch_size**2
        latent = randn_tensor(
            (channels, latent_height, latent_width),
            generator=generator,
            device=self._execution_device,
            dtype=torch.float32,
        )
        token_count = (latent_height // self.patch_size) * (
            latent_width // self.patch_size
        )
        if mu is None:
            mu = (
                1.15
                if self.config.is_distilled
                else calculate_shift(
                    token_count,
                    self.scheduler.config.get("base_image_seq_len", 256),
                    self.scheduler.config.get("max_image_seq_len", 6400),
                    self.scheduler.config.get("base_shift", 0.5),
                    self.scheduler.config.get("max_shift", 1.15),
                )
            )
        self.scheduler.set_timesteps(
            sigmas=np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps),
            device=self._execution_device,
            mu=mu,
        )
        self.scheduler.set_begin_index(0)
        # Cache reference K/V per invocation and CFG branch. References read
        # only references at t=0; virtual tokens, text and target are recomputed.
        positive_cache = {} if use_kv_cache and reference_latents else None
        negative_cache = {} if use_kv_cache and reference_latents else None
        for time in self.progress_bar(self.scheduler.timesteps):
            timestep = (
                (time / self.scheduler.config.num_train_timesteps)
                .reshape(1)
                .to(self.transformer.dtype)
            )

            def predict(embedding, cache):
                tokens = model(
                    [latent.to(self.transformer.dtype)],
                    [reference_latents],
                    [embedding],
                    timestep,
                    kv_cache=cache,
                    drop_reference_attention=[drop_reference_attention],
                )[0]
                return velocity_tokens_to_latent(
                    tokens, latent_height, latent_width, self.patch_size
                )

            velocity = predict(context, positive_cache)
            if negative is not None:
                velocity = velocity + guidance_scale * (
                    velocity - predict(negative, negative_cache)
                )
            latent = self.scheduler.step(
                velocity.float(), time, latent, return_dict=False
            )[0]

        if output_type == "latent":
            images = latent.unsqueeze(0)
        else:
            images = self.decode_image(latent).unsqueeze(0)
            if output_type == "pil":
                images = [to_pil_image(images[0].cpu())]
            elif output_type == "np":
                images = images.permute(0, 2, 3, 1).cpu().numpy()
        self.maybe_free_model_hooks()
        return ImagePipelineOutput(images=images) if return_dict else (images,)
