#!/usr/bin/env python
# coding=utf-8

import argparse
import os
import sys

# BOOTSTRAP_PATHS: allow running scripts directly without installing the package.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import torch
from diffusers import StableDiffusion3Pipeline


def parse_dtype(dtype: str):
    dtype = dtype.lower()
    if dtype == "fp16":
        return torch.float16
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def parse_args():
    parser = argparse.ArgumentParser(description="Vanilla SD3 inference")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--output", type=str, default="output/infer/sd3.png")
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    dtype = parse_dtype(args.dtype)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=dtype,
    ).to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None

    kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": args.num_inference_steps,
        "max_sequence_length": args.max_sequence_length,
        "guidance_scale": args.guidance_scale,
        "generator": generator,
    }
    if args.height is not None:
        kwargs["height"] = args.height
    if args.width is not None:
        kwargs["width"] = args.width

    image = pipe(**kwargs).images[0]
    image.save(args.output)
    print(f"Saved image: {args.output}")


if __name__ == "__main__":
    main()
