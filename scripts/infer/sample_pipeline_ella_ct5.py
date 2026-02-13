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
from datasets import load_dataset, interleave_datasets
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

from lpd.pipeline_lpd_ct5 import PromptDecomposePipeline
from scripts.train.lpd_ella_ct5 import TextDecomposer


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

UG_IDS = torch.tensor(
    [[49406, 49407,     0,     0,     0,     0,     0,     0,     0,     0,
        0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        0,     0,     0,     0,     0,     0,     0]]
)


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.32.0.dev0")

logger = get_logger(__name__)


def get_activation(mem, name):
    def get_output_hook(module, input, output):
        mem[name] = output[1]

    return get_output_hook

def add_hook(net, mem, mapping_layers):
    for n, m in net.named_modules():
        if n in mapping_layers:
            m.register_forward_hook(get_activation(mem, n))


class DecomposingAttnProcessor_pool(nn.Module):

    def __init__(self, num_components, return_entropy = True, learn_temperature = False, dtype = torch.float32):
        super().__init__()
        self.num_components = num_components
        self.return_entropy = return_entropy
        self.learn_temperature = learn_temperature
        if learn_temperature:
            self.temperature = nn.Parameter(torch.ones(1, dtype=dtype))

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states = None,
        attention_mask = None,
        temb = None,
        decompose_mask: nn.Module = None,
    ):
        assert encoder_hidden_states is not None

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_times_comp, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_times_comp, channel, height * width).transpose(1, 2)

        batch_times_comp, sequence_length, hidden_dim = hidden_states.shape
        encoder_sequence_length = encoder_hidden_states.size(1)
        dim_head = attn.out_dim // attn.heads
        # pooled_hidden_states = encoder_hidden_states.sum(dim=1, keepdim=True)   # TODO: mean

        # Derive actual batch size
        batch_size = batch_times_comp // self.num_components

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            # pooled_hidden_states = attn.norm_encoder_hidden_states(pooled_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        # key_pool = attn.to_k(pooled_hidden_states)

        query = query.view(batch_times_comp, sequence_length, attn.heads, dim_head).transpose(1, 2)
        key = key.view(batch_times_comp, encoder_sequence_length, attn.heads, dim_head).transpose(1, 2)
        value = value.view(batch_times_comp, encoder_sequence_length, attn.heads, dim_head).transpose(1, 2)
        # key_pool = key_pool.view(batch_times_comp, 1, attn.heads, dim_head).transpose(1, 2)

        # # TODO: replace this with torch.badmm() if attention_mask is needed
        # attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        # query = query.reshape(batch_times_comp * attn.heads, encoder_sequence_length, dim_head)
        # key = key.reshape(batch_times_comp * attn.heads, encoder_sequence_length, dim_head)
        # value = value.reshape(batch_times_comp * attn.heads, encoder_sequence_length, dim_head)

        if self.learn_temperature:
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.temperature + 1e-8)
        else:
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) * attn.scale
            # attention_scores_pool = torch.matmul(query, key_pool.transpose(-2, -1)) * attn.scale

        attention_scores = attention_scores.view(self.num_components, batch_size, attn.heads, sequence_length, encoder_sequence_length)
        # attention_scores_pool = attention_scores_pool.view(self.num_components, batch_size, attn.heads, sequence_length, 1)
        attention_scores_pool = attention_scores.mean(dim=-1, keepdim=True)

        # apply soft-max
        attention_weights_pooled = F.softmax(attention_scores_pool, dim=0)
        attention_weights = F.softmax(attention_scores, dim=-1) * attention_weights_pooled

        if self.return_entropy:
            compositional_entropy = (attention_weights_pooled*(attention_weights_pooled).log()).sum(dim=0)

        attention_weights = attention_weights.view(batch_times_comp, attn.heads, sequence_length, encoder_sequence_length)

        # Apply attention to values
        hidden_states = torch.matmul(attention_weights, value)
        hidden_states = hidden_states.transpose(1, 2).contiguous().view(
            batch_times_comp, sequence_length, hidden_dim
        )

        if decompose_mask is not None:
            mask_downsample = IPAdapterMaskProcessor.downsample(
                decompose_mask,
                batch_times_comp,
                num_queries=hidden_states.shape[1],
                value_embed_dim=hidden_states.shape[2],
            ).to(device=hidden_states.device)
            hidden_states = hidden_states * mask_downsample
            hidden_states = hidden_states.to(residual.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_times_comp, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        if self.return_entropy:
            return hidden_states, compositional_entropy
        return hidden_states


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

    # Load tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Load scheduler and models
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    tokenizer_t5 = AutoTokenizer.from_pretrained("//zoo/ella/models--google--flan-t5-xl--text_encoder", model_max_length=512)
    text_encoder_t5 = T5EncoderModel.from_pretrained("//zoo/ella/models--google--flan-t5-xl--text_encoder")
    decomposer = TextDecomposer(
        width=args.decomposer_width,
        heads=args.decomposer_heads,
        layers=args.decomposer_layers,
        num_components=args.num_components,
        num_tokens=args.num_tokens,
        text_hidden_dim=text_encoder_t5.config.d_model,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    # Freeze vae and unet
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    decomposer.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_t5.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device and cast to weight_dtype
    # teacher_unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    decomposer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_t5.to(accelerator.device, weight_dtype)

    # Add LoRA to the U-Net, only the LoRA projection matrix will be updated by the optimizer.
    if args.tune_unet:
        if args.lora_target_modules is not None:
            lora_target_modules = [module_key.strip() for module_key in args.lora_target_modules.split(",")]
        else:
            lora_target_modules = [
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                # "proj_in",
                # "proj_out",
                # "ff.net.0.proj",
                # "ff.net.2",
                # "conv1",
                # "conv2",
                # "conv_shortcut",
                # "downsamplers.0.conv",
                # "upsamplers.0.conv",
                # "time_emb_proj",
            ]
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=lora_target_modules,
            lora_dropout=args.lora_dropout,
        )
        unet.add_adapter(lora_config)
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters into fp32
        cast_training_params(unet, dtype=torch.float32)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            # teacher_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

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
            '//data/JourneyDB',
            cache_dir="//data/JourneyDB/.cache",
            split="train",
            trust_remote_code=True,
            streaming=True,
        ).shuffle(seed=args.seed, buffer_size=10)
    if "coco" in args.train_data_dir:
        # train_dataset_coco = load_dataset(
        #     "webdataset",
        #     data_files={"train": '//data/coco2017/train2017.zip'},
        #     split="train"
        # ).to_iterable_dataset(num_shards=200)
        train_dataset_coco = load_dataset(
            '//data/coco2017',
            cache_dir=f"//data/coco2017/.cache",
            split="train",
            trust_remote_code=True,
        ).to_iterable_dataset(num_shards=200).shuffle(seed=args.seed, buffer_size=10)
    if "llava" in args.train_data_dir:
        # train_dataset_lcs = load_dataset(
        #     "webdataset",
        #     data_files={"train": '//data/LLaVA-Pretrain/images.zip'},
        #     split="train"
        # ).to_iterable_dataset(num_shards=200)
        train_dataset_lcs = load_dataset(
            '//data/LLaVA-Pretrain',
            cache_dir=f"//data/LLaVA-Pretrain/.cache",
            split="train",
            trust_remote_code=True,
        ).to_iterable_dataset(num_shards=200).shuffle(seed=args.seed, buffer_size=10)
    if "sam" in args.train_data_dir:
        # train_dataset_sam = load_dataset(
        #     '//data/sa-1b',
        #     split="train",
        #     trust_remote_code=True,
        #     streaming=True,
        # ).shuffle(seed=args.seed, buffer_size=10)
        train_dataset_sam = load_dataset(
            '//data/sa-1b',
            cache_dir=f"//data/sa-1b/.cache",
            split="train",
            trust_remote_code=True,
        ).to_iterable_dataset(num_shards=200).shuffle(seed=args.seed, buffer_size=10)


    # # Preprocessing the datasets.
    # with open('longcaption.json', 'r') as file:
    #     longcaption = json.load(file)

    caption2embed_simple = lambda captions: caption2embed(captions, [tokenizer, tokenizer_t5], [text_encoder, text_encoder_t5],
                                                          accelerator.device, weight_dtype, args=args, token_length=512)

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

    def preprocess_train(examples: dict):
        # image key is jpg
        if len(examples['image_path']) > 0:
            examples['pixel_values'] = Image.open(examples['image_path'])
        examples["pixel_values"] = examples['pixel_values'].convert("RGB").resize(
            (args.train_size, args.train_size), resample=args.interpolation
        )
        # results["pixel_values"] = train_transforms(
        #     examples['pixel_values'].convert("RGB").resize(
        #         (args.train_size, args.train_size), resample=args.interpolation
        #     )
        # )

        # # captions
        # if examples['__key__'].endswith('.jpg'):
        #     # sam and journeydb
        #     if examples['__key__'].startswith('sa'):
        #         path = 'sam-570k/images/'+examples['__key__']
        #         if path not in longcaption['sam']: raise KeyError
        #         results['captions'] = longcaption['sam'][path]
        #     else:
        #         path = "journeydb/images/"+examples['__key__']
        #         if path not in longcaption['journeydb']: raise KeyError
        #         results['captions'] = longcaption['journeydb'][path]
        # else:
        #     # coco and lcs
        #     if 'train2017' in examples['__key__']:
        #         path = f"coco2017/{examples['__key__']}.jpg"
        #         if path not in longcaption['coco']: raise KeyError
        #         results['captions'] = longcaption['coco'][path]
        #     else:
        #         path = f"lcs-558k/images/{examples['__key__']}.jpg"
        #         if path not in longcaption['llava']: raise KeyError
        #         results['captions'] = longcaption['llava'][path]
        return examples
  
    probs = [0.5, 0.1, 0.2, 0.2]
    class MixedIterableDataset(torch.utils.data.IterableDataset):
        def __init__(self, prob = [1.0, 0.0, 0.0, 0.0]):
            self.probs = prob
            self.jdb = train_dataset_jdb
            self.coco = train_dataset_coco
            self.lcs = train_dataset_lcs
            self.sam = train_dataset_sam
            self.captions = longcaption
            self.jdb_done=False
            self.coco_done=False
            self.lcs_done=False
            self.sam_done=False
            self.iter_jdb = iter(self.jdb)
            self.iter_coco = iter(self.coco)
            self.iter_lcs = iter(self.lcs)
            self.iter_sam = iter(self.sam)

        def preprocess_train(self, examples: dict):
            examples["pixel_values"] = train_transforms(
                examples["pixel_values"].convert("RGB").resize(
                    (args.train_size, args.train_size), resample=args.interpolation
                )
            )
            return examples

        def preprocess_captions(self, examples: dict):
            if 'image_path' in examples:
                examples['jpg'] = Image.open(examples['image_path'])
            # elif 'LLaVA' in examples['__url__']:
            #     path = f"lcs-558k/images/{examples['__key__']}.jpg"
            #     examples['captions'] = longcaption['llava'][path] if path in longcaption['llava'] else ""
            # elif 'coco' in examples['__url__']:
            #     path = f"coco2017/{examples['__key__']}.jpg"
            #     examples['captions'] = longcaption['coco'][path] if path in longcaption['coco'] else ""
            # elif 'JourneyDB' in examples['__url__']:
            #     path = f"journeydb/images/{examples['__key__']}.jpg"
            #     examples['captions'] = longcaption['journeydb'][path] if path in longcaption['journeydb'] else ""
            if 'pixel_values' not in examples:
                examples["pixel_values"] = train_transforms(
                    examples['jpg'].convert("RGB").resize(
                        (args.train_size, args.train_size), resample=args.interpolation
                    )
                )
            else:
                examples["pixel_values"] = train_transforms(
                    examples["pixel_values"].convert("RGB").resize(
                        (args.train_size, args.train_size), resample=args.interpolation
                    )
                )
            return examples

        def get_jdb(self):
            for _ in range(5):
                data = next(self.iter_jdb)
                path = f"journeydb/images/{data['url']}"
                if path in longcaption['journeydb']:
                    data['captions'] = longcaption['journeydb'][path]
                    return data
            raise StopIteration

        def get_coco(self):
            for _ in range(5):
                data = next(self.iter_coco)
                path = f"coco2017/{data['__key__']}.jpg"
                if path in longcaption['coco']:
                    data['captions'] = longcaption['coco'][path]
                    return data
            raise StopIteration

        def get_lcs(self):
            for _ in range(5):
                data = next(self.iter_lcs)
                path = f"lcs-558k/images/{data['__key__']}.jpg"
                if path in longcaption['llava']:
                    data['captions'] = longcaption['llava'][path]
                    return data
            raise StopIteration

        def __iter__(self):
            # Create iterators for both datasets
            while not (self.jdb_done and self.coco_done and self.lcs_done and self.sam_done):
                rnd = random.uniform(0, 1)

                try:
                    if rnd < self.probs[0]:
                        if self.jdb_done: continue
                        yield self.preprocess_captions(self.get_jdb())
                    elif rnd < self.probs[1]:
                        if self.coco_done: continue
                        yield self.preprocess_captions(self.get_coco())
                    elif rnd < self.probs[2]:
                        if self.lcs_done: continue
                        yield self.preprocess_captions(self.get_lcs())
                    else:
                        if self.sam_done: continue
                        yield self.preprocess_captions(next(self.iter_sam))
                except StopIteration as e:
                    # If one dataset is exhausted, try another
                    if rnd < self.probs[0]:
                        self.jdb_done = True
                        if not self.coco_done:
                            yield self.preprocess_captions(self.get_coco())
                        elif not self.lcs_done:
                            yield self.preprocess_captions(self.get_lcs())
                        elif not self.sam_done:
                            yield self.preprocess_captions(next(self.iter_sam))
                        else:
                            break
                    elif rnd < self.probs[1]:
                        self.coco_done = True
                        if not self.jdb_done:
                            yield self.preprocess_captions(self.get_jdb())
                        elif not self.lcs_done:
                            yield self.preprocess_captions(self.get_lcs())
                        elif not self.sam_done:
                            yield self.preprocess_captions(next(self.iter_sam))
                        else:
                            break
                    elif rnd < self.probs[2]:
                        self.lcs_done = True
                        if not self.jdb_done:
                            yield self.preprocess_captions(self.get_jdb())
                        elif not self.coco_done:
                            yield self.preprocess_captions(self.get_coco())
                        elif not self.sam_done:
                            yield self.preprocess_captions(next(self.iter_sam))
                        else:
                            break
                    else:
                        self.sam_done = True
                        if not self.jdb_done:
                            yield self.preprocess_captions(self.get_jdb())
                        elif not self.coco_done:
                            yield self.preprocess_captions(self.get_coco())
                        elif not self.lcs_done:
                            yield self.preprocess_captions(self.get_lcs())
                        else:
                            break

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def collate_fn(examples):
        # parse captions
        # pixel_values = torch.stack([example["pixel_values"] for example in examples])
        # pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        pixel_values = [example['pixel_values'] for example in examples]
        results = {"pixel_values": pixel_values}
        captions = [example['captions'] for example in examples]
        # tokens_t5 = tokenizer_t5(captions, max_length=tokenizer_t5.model_max_length,
        #                          padding="max_length", truncation=True, return_tensors="pt")
        # results["input_ids_t5"], results["attention_mask_t5"] = tokens_t5.input_ids, tokens_t5.attention_mask
        results['captions'] = captions
        return results

    # DataLoaders creation:
    # train_dataset = MixedIterableDataset(probs)
    # train_dataset = interleave_datasets(
    #     [train_dataset_jdb, train_dataset_coco, train_dataset_lcs, train_dataset_sam],
    #     probabilities=probs,
    #     seed=args.seed,
    # )
    train_dataset = train_dataset_sam
    train_dataset = train_dataset.map(preprocess_train)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    if args.max_train_steps is None:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Prepare everything with our `accelerator`.
    if args.tune_unet or args.learn_temperature:
        decomposer, unet = accelerator.prepare(
            decomposer, unet
        )
    else:
        decomposer = accelerator.prepare(
            decomposer
        )

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                if isinstance(model, TextDecomposer):
                    save_file(model.state_dict(), os.path.join(output_dir, "model.safetensors"))
                else:
                    unet_sd = model.state_dict()
                    save_sd = dict()
                    for n, p in unet_sd.items():
                        if 'lora' in n or 'temperature' in n:
                            save_sd[n] = p
                    save_file(save_sd, os.path.join(output_dir, "unet.safetensors"))
                weights.pop()

    def load_model_hook(models, input_dir):

        while len(models) > 0:
            # pop models so that they are not loaded again
            model = models.pop()

            if isinstance(model, TextDecomposer):
                model.load_state_dict(load_file(os.path.join(input_dir, "model.safetensors"), device="cpu"))
            else:
                model.load_state_dict(load_file(os.path.join(input_dir, "unet.safetensors"), device="cpu"), strict=False)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = args.max_train_steps
    if overrode_max_train_steps:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # # We need to initialize the trackers we use, and also store our configuration.
    # # The trackers initializes automatically on the main process.
    # if accelerator.is_main_process:
    #     accelerator.init_trackers("long_prompt_decompose_sampling", config=vars(args))
    
    pipeline = PromptDecomposePipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        vae=vae,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

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
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = 0

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Add hook for attn map regu
    acts_stu = {}
    mapping_layers_stu = [
        'down_blocks.0.attentions.0.transformer_blocks.0.attn2', 'down_blocks.0.attentions.1.transformer_blocks.0.attn2',
        'down_blocks.1.attentions.0.transformer_blocks.0.attn2', 'down_blocks.1.attentions.1.transformer_blocks.0.attn2',
        'down_blocks.2.attentions.0.transformer_blocks.0.attn2', 'down_blocks.2.attentions.1.transformer_blocks.0.attn2',
        'mid_block.attentions.0.transformer_blocks.0.attn2',
        'up_blocks.1.attentions.0.transformer_blocks.0.attn2', 'up_blocks.1.attentions.1.transformer_blocks.0.attn2', 'up_blocks.1.attentions.2.transformer_blocks.0.attn2',
        'up_blocks.2.attentions.0.transformer_blocks.0.attn2', 'up_blocks.2.attentions.1.transformer_blocks.0.attn2', 'up_blocks.2.attentions.2.transformer_blocks.0.attn2',
        'up_blocks.3.attentions.0.transformer_blocks.0.attn2', 'up_blocks.3.attentions.1.transformer_blocks.0.attn2', 'up_blocks.3.attentions.2.transformer_blocks.0.attn2',
    ]
    if args.compositional_entropy:
        add_hook(unet, acts_stu, mapping_layers_stu)

    # Training loop
    train_dataloader_iter = iter(train_dataloader)
    step = initial_global_step

    tic = time.time()
    out_meta = []
    with open('//data-juicer/DetailMaster_Dataset/DetailMaster_Dataset.json', 'r') as file:
        meta = json.load(file)
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(meta):
            caption = batch['polished_prompt']
            out_meta.append({"output_image_name": f"{step}.png", "image_id": f"{batch['dataset_target']}_{batch['image_id']}"})
        # for idx, caption in meta.items():
            # try:
            #     batch = next(train_dataloader_iter)
            # except (PIL.UnidentifiedImageError, OSError, IOError, Image.DecompressionBombError) as e:
            #     continue
            # save_base = batch['path'][0].split('.')[0]
            # save_base = '_'.join(save_base.split('/'))
            # meta[idx] = batch['captions'][0]
            with torch.no_grad():
                # run inference
                generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
                if torch.backends.mps.is_available():
                    autocast_ctx = nullcontext()
                else:
                    autocast_ctx = torch.autocast(accelerator.device.type)
                captions = ["", caption]
                encoder_hidden_states_pre = caption2embed_simple(captions)
                encoder_hidden_states_clip = encoder_hidden_states_pre['encoder_hidden_states_clip_concat']
                encoder_hidden_states_t5 = encoder_hidden_states_pre["encoder_hidden_states_t5"]
                # encoder_hidden_states = torch.cat([encoder_hidden_states_clip, encoder_hidden_states_t5], dim=1)
                encoder_hidden_states = encoder_hidden_states_t5
                with autocast_ctx:
                    image = pipeline(
                        accelerator.unwrap_model(decomposer),
                        encoder_hidden_states_t5=encoder_hidden_states_t5,
                        encoder_hidden_states_clip=encoder_hidden_states_clip,
                        prompt_embeds = encoder_hidden_states[1:],
                        negative_prompt_embeds = encoder_hidden_states[:1],
                        num_inference_steps=25,
                        guidance_scale=5.5,
                        generator=generator
                    ).images[0]
                image.save(f"{args.save_dir}/{step}.png")
                toc = time.time()
                logs = {
                    "time": toc-tic,
                }
                tic = toc
                # with autocast_ctx:
                #     image = pipeline.decompose(
                #         accelerator.unwrap_model(decomposer),
                #         prompt_embeds = encoder_hidden_states[1:],
                #         negative_prompt_embeds = encoder_hidden_states[:1],
                #         num_inference_steps=25,
                #         generator=generator
                #     ).images
                # images.extend(image)

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
