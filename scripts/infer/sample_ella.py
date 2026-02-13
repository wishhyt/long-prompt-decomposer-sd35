#!/usr/bin/env python
# coding=utf-8

import os
import sys

# BOOTSTRAP_PATHS: allow running scripts directly without installing the package.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

# system
import argparse
import logging
import math
import os
import json
import random
import shutil
import warnings
import time

# utilities
import numpy as np
import PIL
import safetensors
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from accelerate import Accelerator
from safetensors.torch import save_file, load_file
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from lpd.tools import caption2embed, ids2embed

# data
from PIL import Image
from packaging import version
from torchvision import transforms
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    AutoTokenizer,
    T5EncoderModel,
)
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

# models
import diffusers as diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
    StableDiffusionPipeline
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.models.attention_processor import AttnProcessor
from diffusers.image_processor import IPAdapterMaskProcessor

from lpd.pipeline_prompt_decomposition import PromptDecomposePipeline
from lpd.models import MLP, PromptResampler


if is_wandb_available():
    import wandb

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Optional, Union

import safetensors.torch
from safetensors.torch import load_file
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from torchvision.utils import save_image

from lpd.model import ELLA, T5TextEmbedder


class ELLAProxyUNet(torch.nn.Module):
    def __init__(self, ella, unet):
        super().__init__()
        # In order to still use the diffusers pipeline, including various workaround

        self.ella = ella
        self.unet = unet
        self.config = unet.config
        self.dtype = unet.dtype
        self.device = unet.device

        self.flexible_max_length_workaround = None

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[dict[str, Any]] = None,
        added_cond_kwargs: Optional[dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        if self.flexible_max_length_workaround is not None:
            time_aware_encoder_hidden_state_list = []
            for i, max_length in enumerate(self.flexible_max_length_workaround):
                time_aware_encoder_hidden_state_list.append(
                    self.ella(encoder_hidden_states[i : i + 1, :max_length], timestep)
                )
            # No matter how many tokens are text features, the ella output must be 64 tokens.
            time_aware_encoder_hidden_states = torch.cat(
                time_aware_encoder_hidden_state_list, dim=0
            )
        else:
            time_aware_encoder_hidden_states = self.ella(
                encoder_hidden_states, timestep
            )

        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=time_aware_encoder_hidden_states,
            class_labels=class_labels,
            timestep_cond=timestep_cond,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            down_intrablock_additional_residuals=down_intrablock_additional_residuals,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=return_dict,
        )


def generate_image_with_flexible_max_length(
    pipe, t5_encoder, prompt, fixed_negative=False, output_type="pt", **pipe_kwargs
):
    device = pipe.device
    dtype = pipe.dtype
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    prompt_embeds = t5_encoder(prompt, max_length=None).to(device, dtype)
    negative_prompt_embeds = t5_encoder(
        [""] * batch_size, max_length=128 if fixed_negative else None
    ).to(device, dtype)

    # diffusers pipeline concatenate `prompt_embeds` too early...
    # https://github.com/huggingface/diffusers/blob/b6d7e31d10df675d86c6fe7838044712c6dca4e9/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L913
    pipe.unet.flexible_max_length_workaround = [
        negative_prompt_embeds.size(1)
    ] * batch_size + [prompt_embeds.size(1)] * batch_size

    max_length = max([prompt_embeds.size(1), negative_prompt_embeds.size(1)])
    b, _, d = prompt_embeds.shape
    prompt_embeds = torch.cat(
        [
            prompt_embeds,
            torch.zeros(
                (b, max_length - prompt_embeds.size(1), d), device=device, dtype=dtype
            ),
        ],
        dim=1,
    )
    negative_prompt_embeds = torch.cat(
        [
            negative_prompt_embeds,
            torch.zeros(
                (b, max_length - negative_prompt_embeds.size(1), d),
                device=device,
                dtype=dtype,
            ),
        ],
        dim=1,
    )

    images = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        **pipe_kwargs,
    ).images[0]
    pipe.unet.flexible_max_length_workaround = None
    return images


def load_ella(filename, device, dtype):
    ella = ELLA()
    # safetensors.torch.load_model(ella, filename, strict=True)
    test_sd = load_file(
        f"/scratch4/mdredze1/jhuan236/zoo/ella/ella-sd1.5-tsc-t5xl.safetensors",
    device="cpu")
    ella.load_state_dict(test_sd)
    ella.to(device, dtype=dtype)
    return ella


def load_ella_for_pipe(pipe, ella):
    pipe.unet = ELLAProxyUNet(ella, pipe.unet)


def offload_ella_for_pipe(pipe):
    pipe.unet = pipe.unet.unet


def generate_image_with_fixed_max_length(
    pipe, t5_encoder, prompt, output_type="pt", **pipe_kwargs
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    prompt_embeds = t5_encoder(prompt, max_length=128).to(pipe.device, pipe.dtype)
    negative_prompt_embeds = t5_encoder([""] * len(prompt), max_length=128).to(
        pipe.device, pipe.dtype
    )

    return pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        **pipe_kwargs,
    ).images[0]


def sample(pipe, t5_encoder, prompt):

    _batch_size = 1
    size = 512
    seed = 1001
    prompt = [prompt] * _batch_size

    return generate_image_with_flexible_max_length(
        pipe,
        t5_encoder,
        prompt,
        guidance_scale=12,
        num_inference_steps=50,
        height=size,
        width=size,
        generator=[
            torch.Generator(device="cuda").manual_seed(seed + i)
            for i in range(_batch_size)
        ],
    )
    return generate_image_with_fixed_max_length(
        pipe,
        t5_encoder,
        prompt,
        guidance_scale=12,
        num_inference_steps=50,
        height=size,
        width=size,
        generator=[
            torch.Generator(device="cuda").manual_seed(seed + i)
            for i in range(_batch_size)
        ],
    )


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.32.0.dev0")

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--num_components",
        type=int,
        default=4,
        help="How many textual inversion vectors shall be used to learn the concept.",
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=1,
        help="How many learnable tokens in resampler.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument("--token_length", type=int, default=512)
    parser.add_argument(
        "--decomposer_width",
        type=int,
        default=1024,
        help="The rank of the LoRA projection matrix.",
    )
    parser.add_argument(
        "--decomposer_heads",
        type=int,
        default=8,
        help="The rank of the LoRA projection matrix.",
    )
    parser.add_argument(
        "--decomposer_layers",
        type=int,
        default=6,
        help="The rank of the LoRA projection matrix.",
    )
    parser.add_argument(
        "--regu_weight",
        type=float,
        default=0.0,
        help="The dropout probability for the dropout layer added before applying the LoRA to each layer input.",
    )
    parser.add_argument(
        "--l2_norm_coeff",
        type=float,
        default=0.0,
        help="The dropout probability for the dropout layer added before applying the LoRA to each layer input.",
    )
    parser.add_argument(
        "--component_dropout",
        type=float,
        default=0.1,
        help="The dropout probability for the dropout layer added before applying the LoRA to each layer input.",
    )
    parser.add_argument(
        "--dropout_v1",
        action="store_true",
        help="Whether or not to use dropout v1: drop some components in a batch.",
    )
    parser.add_argument(
        "--tune_unet",
        action="store_true",
        help="Whether or not to tune unet LoRA.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The rank of the LoRA projection matrix.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="The dropout probability for the dropout layer added before applying the LoRA to each layer input.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default=None,
        help=(
            "A comma-separated string of target module keys to add LoRA to. If not set, a default list of modules will"
            " be used. By default, LoRA will be applied to all conv and linear layers."
        ),
    )
    parser.add_argument(
        "--learn_temperature",
        action="store_true",
        help="Whether or not to learn tempreture in CWA.",
    )
    parser.add_argument(
        "--compositional_entropy",
        action="store_true",
        help=(
            "Whether or not to use compositional entropy regulariser"
        ),
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=None, help="A folder containing the training data."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_size",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--interpolation",
        type=str,
        default="bilinear",
        help=(
            'train data processing interpolation mode'
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and Nvidia Ampere GPU or Intel Gen 4 Xeon (and later) ."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default='''
        The image showcases a box of Advocate flea and heartworm treatment for dogs. The box is (1) predominantly
        orange in color, standing upright against a white background.
        The front of the box is (2) adorned with a photograph of a
        black and white dog, who appears to be (3) standing on a
        grassy field. The dog's gaze is directed towards the camera,
        adding a sense of engagement to the image. Overall, the box is
        designed to provide essential information about the product in
        a clear and concise manner, while also emphasizing the
        importance of safety.''',
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        help="If specified save the checkpoint not in `safetensors` format, but in original PyTorch format instead.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        split_batches=True
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.save_dir, exist_ok=True)

    text_encoder_t5 = T5TextEmbedder("/scratch4/mdredze1/jhuan236/zoo/ella/models--google--flan-t5-xl--text_encoder",)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device and cast to weight_dtype
    # teacher_unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder_t5.to(accelerator.device, weight_dtype)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset and DataLoaders creation:
    if args.debug:
        from datasets import Dataset
        def dummy_data_generator():
            for i in range(100):
                yield {
                    "res": args.train_size,
                    "caption": "",
                }
        train_dataset = Dataset.from_generator(dummy_data_generator)
    if "JourneyDB" in args.train_data_dir:
        # base_url = "https://huggingface.co/datasets/JourneyDB/JourneyDB/resolve/main/data/train/imgs/{i:03d}.tgz"
        # urls = [base_url.format(i=i) for i in range(200)]
        # train_dataset_jdb = load_dataset("webdataset", data_files={"train": urls}, split="train", streaming=True)
        train_dataset_jdb = load_dataset(
            '/scratch4/mdredze1/jhuan236/data/JourneyDB',
            cache_dir="/scratch4/mdredze1/jhuan236/data/JourneyDB/.cache",
            split="train",
            trust_remote_code=True,
            streaming=True,
        ).shuffle(seed=args.seed, buffer_size=10)
    if "coco" in args.train_data_dir:
        # train_dataset_coco = load_dataset(
        #     "webdataset",
        #     data_files={"train": '/scratch4/mdredze1/jhuan236/data/coco2017/train2017.zip'},
        #     split="train"
        # ).to_iterable_dataset(num_shards=200)
        train_dataset_coco = load_dataset(
            '/scratch4/mdredze1/jhuan236/data/coco2017',
            cache_dir=f"/scratch4/mdredze1/jhuan236/data/coco2017/.cache",
            split="train",
            trust_remote_code=True,
        ).to_iterable_dataset(num_shards=200).shuffle(seed=args.seed, buffer_size=10)
    if "llava" in args.train_data_dir:
        # train_dataset_lcs = load_dataset(
        #     "webdataset",
        #     data_files={"train": '/scratch4/mdredze1/jhuan236/data/LLaVA-Pretrain/images.zip'},
        #     split="train"
        # ).to_iterable_dataset(num_shards=200)
        train_dataset_lcs = load_dataset(
            '/scratch4/mdredze1/jhuan236/data/LLaVA-Pretrain',
            cache_dir=f"/scratch4/mdredze1/jhuan236/data/LLaVA-Pretrain/.cache",
            split="train",
            trust_remote_code=True,
        ).to_iterable_dataset(num_shards=200).shuffle(seed=args.seed, buffer_size=10)
    if "sam" in args.train_data_dir:
        # train_dataset_sam = load_dataset(
        #     '/scratch4/mdredze1/jhuan236/data/sa-1b',
        #     split="train",
        #     trust_remote_code=True,
        #     streaming=True,
        # ).shuffle(seed=args.seed, buffer_size=10)
        train_dataset_sam = load_dataset(
            '/scratch4/mdredze1/jhuan236/data/sa-1b',
            cache_dir=f"/scratch4/mdredze1/jhuan236/data/sa-1b/.cache",
            split="train",
            trust_remote_code=True,
        ).to_iterable_dataset(num_shards=200).shuffle(seed=args.seed, buffer_size=10)

    # Preprocessing the datasets.
    image_column, caption_column = 'pixel_values', 'captions'

    # with open('journeydb_caption.json', 'r') as file:
    #     longcaption = json.load(file)

    # Preprocessing the datasets.
    args.interpolation = {
        "linear": PIL_INTERPOLATION["linear"],
        "bilinear": PIL_INTERPOLATION["bilinear"],
        "bicubic": PIL_INTERPOLATION["bicubic"],
        "lanczos": PIL_INTERPOLATION["lanczos"],
    }[args.interpolation]
    train_transforms = transforms.Compose(
        [
            transforms.CenterCrop(args.train_size) if args.center_crop else transforms.RandomCrop(args.train_size),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # def collate_fn(examples):
    #     pixel_values = torch.stack([example["pixel_values"] for example in examples])
    #     pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    #     results = {
    #         "pixel_values": pixel_values,
    #         "captions": [example["captions"] for example in examples],
    #         "path": [example["url"] for example in examples],
    #     }
    #     # tokens_t5 = tokenizer_t5(captions, max_length=tokenizer_t5.model_max_length,
    #     #                          padding="max_length", truncation=True, return_tensors="pt")
    #     # results["input_ids_t5"], results["attention_mask_t5"] = tokens_t5.input_ids, tokens_t5.attention_mask
    #     return results

    # # DataLoaders creation:
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     shuffle=False,
    #     collate_fn=collate_fn,
    #     batch_size=args.train_batch_size,
    #     num_workers=args.dataloader_num_workers,
    # )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    if args.max_train_steps is None:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = args.max_train_steps
    if overrode_max_train_steps:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    pipe = StableDiffusionPipeline.from_pretrained(
        "/scratch4/mdredze1/jhuan236/zoo/runwayml/stable-diffusion-v1-5",
        torch_dtype=weight_dtype,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(accelerator.device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    ella = load_ella("/scratch4/mdredze1/jhuan236/zoo/ella", pipe.device, pipe.dtype)
    load_ella_for_pipe(pipe, ella)
    t5_encoder = text_encoder_t5

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    # logger.info(f"  Total optimization steps = {args.max_train_samples}")
    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Training loop
    # train_dataloader_iter = iter(train_dataloader)
    step = initial_global_step

    tic = time.time()
    out_meta = []
    with open('/scratch4/mdredze1/jhuan236/data-juicer/DetailMaster_Dataset/DetailMaster_Dataset.json', 'r') as file:
        meta = json.load(file)
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(meta):
            caption = batch['polished_prompt']
            out_meta.append({"output_image_name": f"{step}.png", "image_id": f"{batch['dataset_target']}_{batch['image_id']}"})
            with torch.no_grad():
                # run inference
                image = sample(pipe, t5_encoder, caption)
                image.save(f"{args.save_dir}/{step}.png")
            toc = time.time()
            logs = {"time": toc-tic}
            tic = toc

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()
    with open(f"{args.save_dir}.json", "w") as f:
        json.dump(out_meta, f, indent=4)


if __name__ == "__main__":
    main()
