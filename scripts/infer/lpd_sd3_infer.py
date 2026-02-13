#!/usr/bin/env python
# coding=utf-8

import argparse
import os
import sys
from typing import Optional

# BOOTSTRAP_PATHS: allow running scripts directly without installing the package.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import torch
from safetensors.torch import load_file
from diffusers import StableDiffusion3Pipeline

from lpd.models import PromptResampler
from lpd.pipeline_lpd_sd3 import PromptDecomposePipeline


class TextDecomposer(torch.nn.Module):
    def __init__(
        self,
        num_components=4,
        num_tokens=128,
        width=None,
        heads=32,
        layers=6,
        text_hidden_dim=None,
        unet_hidden_dim=None,
    ):
        super().__init__()
        self.num_components = num_components
        self.mask_head = PromptResampler(
            width=width,
            heads=heads,
            layers=layers,
            num_tokens=num_tokens,
            num_components=num_components,
            input_dim=text_hidden_dim,
            output_dim=unet_hidden_dim,
        )

    def forward(self, encoder_hidden_state_t5):
        return self.mask_head(encoder_hidden_state_t5)


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt,
    device=None,
    num_images_per_prompt=1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


def clip_encode_prompt(
    text_encoders,
    tokenizers,
    prompt,
    target_hidden_size,
    device=None,
    num_images_per_prompt=1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[0]
    clip_text_encoders = text_encoders[0]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    if clip_prompt_embeds.shape[-1] < target_hidden_size:
        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, target_hidden_size - clip_prompt_embeds.shape[-1])
        )
    elif clip_prompt_embeds.shape[-1] > target_hidden_size:
        clip_prompt_embeds = clip_prompt_embeds[..., :target_hidden_size]

    return clip_prompt_embeds, pooled_prompt_embeds


def parse_dtype(dtype: str):
    dtype = dtype.lower()
    if dtype == "fp16":
        return torch.float16
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def resolve_checkpoint_path(ckpt: str) -> str:
    if os.path.isdir(ckpt):
        ckpt = os.path.join(ckpt, "model.safetensors")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


def parse_args():
    parser = argparse.ArgumentParser(description="Run SD3.5 + PromptDecomposer inference.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--decomposer_ckpt", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="output/infer")
    parser.add_argument("--compose_filename", type=str, default="compose.png")
    parser.add_argument("--save_decompose", action="store_true", help="Save per-component images.")

    parser.add_argument("--num_components", type=int, default=4)
    parser.add_argument("--num_tokens", type=int, default=128)
    parser.add_argument("--decomposer_width", type=int, default=None)
    parser.add_argument("--decomposer_heads", type=int, default=32)
    parser.add_argument("--decomposer_layers", type=int, default=6)

    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", type=str, default=None, help="Default: cuda if available else cpu.")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dtype = parse_dtype(args.dtype)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32

    ckpt_path = resolve_checkpoint_path(args.decomposer_ckpt)

    base_pipe = StableDiffusion3Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=dtype,
    )
    base_pipe = base_pipe.to(device)

    pipe = PromptDecomposePipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=base_pipe.text_encoder,
        text_encoder_2=base_pipe.text_encoder_2,
        text_encoder_3=base_pipe.text_encoder_3,
        tokenizer=base_pipe.tokenizer,
        tokenizer_2=base_pipe.tokenizer_2,
        tokenizer_3=base_pipe.tokenizer_3,
        transformer=base_pipe.transformer,
        vae=base_pipe.vae,
        safety_checker=None,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    t5_hidden_size = pipe.text_encoder_3.config.d_model
    decomposer_width = args.decomposer_width or t5_hidden_size
    if decomposer_width != t5_hidden_size:
        raise ValueError(
            f"--decomposer_width must match text_encoder_3 hidden size {t5_hidden_size}, got {decomposer_width}."
        )
    if decomposer_width % args.decomposer_heads != 0:
        raise ValueError("--decomposer_width must be divisible by --decomposer_heads.")

    decomposer = TextDecomposer(
        width=decomposer_width,
        heads=args.decomposer_heads,
        layers=args.decomposer_layers,
        num_tokens=args.num_tokens,
        num_components=args.num_components,
    ).to(device=device, dtype=dtype)
    decomposer.load_state_dict(load_file(ckpt_path, device="cpu"))
    decomposer.eval()

    use_cfg = args.guidance_scale > 1.0
    captions = [args.negative_prompt, args.prompt] if use_cfg else [args.prompt]
    text_encoders = ((pipe.text_encoder, pipe.text_encoder_2), pipe.text_encoder_3)
    tokenizers = ((pipe.tokenizer, pipe.tokenizer_2), pipe.tokenizer_3)

    with torch.no_grad():
        encoder_hidden_states_t5 = _encode_prompt_with_t5(
            text_encoder=pipe.text_encoder_3,
            tokenizer=pipe.tokenizer_3,
            max_sequence_length=args.max_sequence_length,
            prompt=captions,
            device=device,
        )
        clip_prompt_embeds, pooled_prompt_embeds = clip_encode_prompt(
            text_encoders=text_encoders,
            tokenizers=tokenizers,
            prompt=captions,
            target_hidden_size=t5_hidden_size,
            device=device,
        )

    generator: Optional[torch.Generator] = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    with torch.no_grad():
        composed = pipe(
            decomposer,
            encoder_hidden_states_t5,
            clip_prompt_embeds,
            pooled_prompt_embeds,
            prompt="",
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]

    compose_path = os.path.join(args.output_dir, args.compose_filename)
    composed.save(compose_path)
    print(f"Saved composed image: {compose_path}")

    if args.save_decompose:
        with torch.no_grad():
            component_images = pipe.decompose(
                decomposer,
                encoder_hidden_states_t5,
                clip_prompt_embeds,
                pooled_prompt_embeds,
                prompt="",
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images

        for idx, image in enumerate(component_images):
            path = os.path.join(args.output_dir, f"component_{idx}.png")
            image.save(path)
        print(f"Saved {len(component_images)} component images under: {args.output_dir}")


if __name__ == "__main__":
    main()
