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
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
# limitations under the License.

import argparse
import contextlib
import copy
import functools
import gc
import io
import logging
import math
import os
import random
import shutil

# Add repo root to path to import from tests
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.utils.checkpoint
from safetensors.torch import save_file, load_file
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed, DataLoaderConfiguration
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from typing import Optional, Set

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

from lpd.pipeline_lpd_sd3 import PromptDecomposePipeline
from lpd.models import PromptResampler, PromptResampler_ct5, PromptDecomposer, PromptDecomposer_v2
from lpd.tools import caption2embed, caption2embed_sd3


if is_wandb_available():
    import wandb

logger = get_logger(__name__)


def _prompt_to_text(prompt):
    if isinstance(prompt, list):
        return " ".join(prompt)
    return prompt


def log_validation(decomposer, caption2embed_simple, transformer, vae, text_encoder, text_encoder_2, text_encoder_3, tokenizer, tokenizer_2, tokenizer_3, args, accelerator, weight_dtype, epoch, return_mask=False):
    validation_prompt = _prompt_to_text(args.validation_prompt)
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {validation_prompt}."
    )
    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = PromptDecomposePipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        text_encoder_3=text_encoder_3,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        tokenizer_3=tokenizer_3,
        transformer=accelerator.unwrap_model(transformer),
        vae=vae,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    images = []
    compose_image = []
    if torch.backends.mps.is_available():
        autocast_ctx = contextlib.nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    with torch.no_grad():
        captions = ["", validation_prompt]
        encoder_hidden_states_pre = caption2embed_simple(captions)
        # encoder_hidden_states_clip = encoder_hidden_states_pre['encoder_hidden_states_clip_concat']
        encoder_hidden_states_t5 = encoder_hidden_states_pre["encoder_hidden_states_t5"]
        # encoder_hidden_states_clip_pooled = encoder_hidden_states_pre['encoder_hidden_states_clip_pooled']
        clip_prompt_embeds, pooled_prompt_embeds = clip_encode_prompt(
            ((text_encoder, text_encoder_2), text_encoder_3),
            ((tokenizer, tokenizer_2), tokenizer_3),
            captions,
            args.token_length,
            device=None,
        )
    with autocast_ctx:
        image = pipeline(
            accelerator.unwrap_model(decomposer),
            encoder_hidden_states_t5,
            clip_prompt_embeds,
            pooled_prompt_embeds,
            prompt="",
            num_inference_steps=20,
            generator=generator
        ).images[0]
    compose_image.append(image)
    with autocast_ctx:
        image = pipeline.decompose(
            accelerator.unwrap_model(decomposer),
            encoder_hidden_states_t5,
            clip_prompt_embeds,
            pooled_prompt_embeds,
            prompt="",
            num_inference_steps=20,
            generator=generator
        ).images
    images.extend(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    "compose": [
                        wandb.Image(image, caption=f"{i}: {validation_prompt}") for i, image in enumerate(compose_image)
                    ],
                    "decompose": [
                        wandb.Image(image, caption=f"Component {i}") for i, image in enumerate(images)
                    ],
                    # "masks": [
                    #     wandb.Image(
                    #         image.unsqueeze(-1).cpu().numpy(),
                    #         caption = f"mask {i}"
                    #     ) for i, image in enumerate(token_mask.detach())
                    # ],
                }
            )

    del pipeline
    free_memory()
    return images


class TextDecomposer(torch.nn.Module):
    def __init__(
        self,
        num_components=4,
        num_tokens=1,
        width=None,
        heads=None,
        layers=None,
        text_hidden_dim=None,
        unet_hidden_dim=None,
    ):
        super().__init__()
        self.num_components = num_components

        # ELLA
        self.mask_head = PromptResampler(
            width=width,
            heads=heads,
            layers=layers,
            num_tokens=num_tokens,
            num_components=num_components,
            input_dim=text_hidden_dim,
            output_dim=unet_hidden_dim,
        )

    def forward(self, encoder_hidden_state_t5,):
        bs, seq_len, hidden_dim = encoder_hidden_state_t5.shape
        encoder_hidden_state_list = self.mask_head(encoder_hidden_state_t5,)
        return encoder_hidden_state_list


# Copied from dreambooth sd3 example
def load_text_encoders(class_one, class_two, class_three):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    text_encoder_three = class_three.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two, text_encoder_three


# Copied from dreambooth sd3 example
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--num_components",
        type=int,
        default=4,
        help="How many textual inversion vectors shall be used to learn the concept.",
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=128,
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
    parser.add_argument("--token_length", type=int, default=512)
    parser.add_argument(
        "--decomposer_width",
        type=int,
        default=None,
        help="Decomposer hidden dimension. Defaults to text_encoder_3 hidden size for SD3.5.",
    )
    parser.add_argument(
        "--decomposer_heads",
        type=int,
        default=32,
        help="Number of decomposer attention heads.",
    )
    parser.add_argument(
        "--decomposer_layers",
        type=int,
        default=6,
        help="The rank of the LoRA projection matrix.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether or not to use center crop other wise random.",
    )
    parser.add_argument(
        "--component_dropout",
        type=float,
        default=0.1,
        help="The dropout probability for the dropout layer added before applying the LoRA to each layer input.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
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
        "--upcast_vae",
        action="store_true",
        help="Whether or not to upcast vae to fp32",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
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
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--precondition_outputs",
        type=int,
        default=1,
        help="Flag indicating if we are preconditioning the model outputs or not as done in EDM. This affects how "
        "model `target` is calculated.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
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
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="luping-liu/LongAlign",
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "Fallback local dataset folder for `imagefolder` format. Ignored when `dataset_name` is provided."
        ),
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Load dataset in streaming mode (recommended for large datasets).",
    )
    parser.add_argument(
        "--no_streaming",
        action="store_false",
        dest="streaming",
        help="Disable streaming mode.",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Root directory for image files when dataset rows contain relative paths (e.g., LongAlign `path`).",
    )
    parser.add_argument(
        "--path_column",
        type=str,
        default="path",
        help="Column containing relative image paths.",
    )
    parser.add_argument(
        "--source_column",
        type=str,
        default="source",
        help="Column containing source name (used by `--source_filter`).",
    )
    parser.add_argument(
        "--source_filter",
        type=str,
        default="coco",
        help="Comma-separated sources to keep, e.g. `coco` or `coco,llava`. Use `*` to disable filtering.",
    )
    parser.add_argument(
        "--skip_missing_images",
        action="store_true",
        default=True,
        help="Skip missing/corrupt images instead of failing.",
    )
    parser.add_argument(
        "--no_skip_missing_images",
        action="store_false",
        dest="skip_missing_images",
        help="Fail on missing/corrupt images.",
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help='previous ckpt',
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default="A sea turtle close up side view in the ocean water is swimming with a background of large boulder rocks. In the background are a few different breeds of small fish swimming in the opposite direction of the sea turtle. There is a bigger fish, which is blue-grey in color, with a sleek body shape, swimming in the upper right part of the image where the tail and mid-body are in view to the right. The sea turtle, which looks old, has light hitting the back of its shell, suggesting an outdoor setting with natural light. The image is backlit, with the lighting being soft, creating a calm and serene underwater atmosphere. The background features large boulder rocks and various small fish swimming in the ocean, with a larger fish partially visible to the right. The lighting highlights the sea turtle's shell, adding depth to the underwater scene. The style of the image is a realistic photo. The sea turtle in close-up side view is positioned in the foreground, with the large boulder rocks in the background behind it. The small fish swimming in the background are located further away from the sea turtle, moving in the opposite direction. The bigger fish with tail and mid-body in view is situated to the right of the sea turtle, with its body partially obscured by the rocks.",
        help=(
            "Validation prompt evaluated every `--validation_steps` and logged to trackers."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
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
    parser.add_argument(
        "--num_update_steps_per_epoch",
        type=int,
        default=1000,
        help="Virtual epoch length for iterable/streaming datasets.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="lpd_sd3",
        help="Tracker project name (W&B project when `--report_to wandb`).",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional W&B run name.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Optional W&B entity/team name.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    prompt_embeds = torch.stack([torch.tensor(example["prompt_embeds"]) for example in examples])
    pooled_prompt_embeds = torch.stack([torch.tensor(example["pooled_prompt_embeds"]) for example in examples])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
    }


# Copied from dreambooth sd3 example
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

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


# Copied from dreambooth sd3 example
def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    num_images_per_prompt: int = 1,
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
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


# Copied from dreambooth sd3 example
def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
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

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds


def clip_encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
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

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, 4096 - clip_prompt_embeds.shape[-1])
    )
    # prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return clip_prompt_embeds, pooled_prompt_embeds


class LongCaptionIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset,
        transform,
        caption_column: str,
        image_column: str,
        path_column: str,
        source_column: str,
        source_filter: Optional[Set[str]],
        image_root: Optional[str],
        skip_missing_images: bool,
    ):
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.caption_column = caption_column
        self.image_column = image_column
        self.path_column = path_column
        self.source_column = source_column
        self.source_filter = source_filter
        self.image_root = image_root
        self.skip_missing_images = skip_missing_images

    def set_epoch(self, epoch: int):
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def _get_caption(self, example):
        for key in [self.caption_column, "captions", "caption", "text", "prompt"]:
            if key in example and example[key] is not None:
                value = example[key]
                if isinstance(value, (list, tuple)):
                    return str(random.choice(value)) if value else ""
                return str(value)
        return None

    def _load_image(self, example):
        image = example.get(self.image_column)
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, dict):
            if image.get("bytes") is not None:
                return Image.open(io.BytesIO(image["bytes"]))
            if image.get("path"):
                return Image.open(image["path"])

        candidate_paths = []
        if "image_path" in example and example["image_path"]:
            candidate_paths.append(str(example["image_path"]))
        if self.path_column in example and example[self.path_column]:
            rel_path = str(example[self.path_column])
            candidate_paths.append(rel_path)
            if self.image_root:
                candidate_paths.append(os.path.join(self.image_root, rel_path))

        for path in candidate_paths:
            if os.path.isfile(path):
                return Image.open(path)

        raise FileNotFoundError(f"Could not resolve image path for sample: {candidate_paths}")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            dataset_iter = iter(self.dataset)
        elif hasattr(self.dataset, "shard"):
            dataset_iter = iter(self.dataset.shard(num_shards=worker_info.num_workers, index=worker_info.id))
        else:
            # Fallback sharding for iterables that do not implement `.shard`.
            def _manual_shard():
                for idx, item in enumerate(self.dataset):
                    if idx % worker_info.num_workers == worker_info.id:
                        yield item

            dataset_iter = _manual_shard()

        for example in dataset_iter:
            if self.source_filter is not None:
                source = str(example.get(self.source_column, "")).lower()
                if source not in self.source_filter:
                    continue

            caption = self._get_caption(example)
            if caption is None:
                continue

            try:
                image = self._load_image(example)
                pixel_values = self.transform(image.convert("RGB"))
            except Exception:
                if self.skip_missing_images:
                    continue
                raise

            yield {"pixel_values": pixel_values, "captions": caption}


def main(args):
    if args.report_to in {"wandb", "all"} and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    dataloader_config = DataLoaderConfiguration(dispatch_batches=False)     # optional: split_batches=True
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        dataloader_config=dataloader_config,
    )

    if args.report_to in {"wandb", "all"}:
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
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load the tokenizer
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=args.revision,
    )

    # import correct text encoder class
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three
    )
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    text_encoder_three.to(accelerator.device, dtype=weight_dtype)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )
    t5_hidden_size = text_encoder_three.config.d_model
    if args.decomposer_width is None:
        args.decomposer_width = t5_hidden_size
    if args.decomposer_width != t5_hidden_size:
        raise ValueError(
            f"`--decomposer_width` must match text_encoder_3 hidden size ({t5_hidden_size}) for this script; "
            f"got {args.decomposer_width}."
        )
    if args.decomposer_width % args.decomposer_heads != 0:
        raise ValueError(
            f"`--decomposer_width` ({args.decomposer_width}) must be divisible by `--decomposer_heads` ({args.decomposer_heads})."
        )
    decomposer = TextDecomposer(
        width=args.decomposer_width,
        heads=args.decomposer_heads,
        layers=args.decomposer_layers,
        num_tokens=args.num_tokens,
        num_components=args.num_components,
    )
    if args.ckpt is not None:
        test_sd = load_file(f"{args.ckpt}/model.safetensors", device="cpu")
        print(f'loading ckpt {args.ckpt}')
        decomposer.load_state_dict(test_sd)

    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)
    decomposer.train()

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    optimizer = optimizer_class(
        decomposer.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Move vae, transformer and text_encoder to device and cast to weight_dtype
    if args.upcast_vae:
        vae.to(accelerator.device, dtype=torch.float32)
    else:
        vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    decomposer.to(accelerator.device)

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Pass either `--dataset_name` or `--train_data_dir`.")

    if args.dataset_name is not None:
        local_dataset_dir = Path(args.dataset_name)
        if local_dataset_dir.exists() and local_dataset_dir.is_dir():
            parquet_files = sorted(str(path) for path in local_dataset_dir.rglob("*.parquet"))
            if parquet_files:
                raw_train_dataset = load_dataset(
                    "parquet",
                    data_files=parquet_files,
                    split=args.dataset_split,
                    cache_dir=args.cache_dir,
                    streaming=args.streaming,
                )
            else:
                raw_train_dataset = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    split=args.dataset_split,
                    cache_dir=args.cache_dir,
                    streaming=args.streaming,
                )
        else:
            raw_train_dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=args.dataset_split,
                cache_dir=args.cache_dir,
                streaming=args.streaming,
            )
    else:
        raw_train_dataset = load_dataset(
            "imagefolder",
            data_dir=args.train_data_dir,
            split=args.dataset_split,
            cache_dir=args.cache_dir,
            streaming=args.streaming,
        )

    if args.max_train_samples is not None:
        if args.streaming:
            raw_train_dataset = raw_train_dataset.take(args.max_train_samples)
        else:
            max_samples = min(args.max_train_samples, len(raw_train_dataset))
            raw_train_dataset = raw_train_dataset.shuffle(seed=args.seed).select(range(max_samples))

    source_filter = None
    if args.source_filter and args.source_filter.strip() and args.source_filter.strip() != "*":
        source_filter = {item.strip().lower() for item in args.source_filter.split(",") if item.strip()}

    tokenizers = [(tokenizer_one, tokenizer_two), tokenizer_three]
    text_encoders = [(text_encoder_one, text_encoder_two), text_encoder_three]
    caption2embed_simple = lambda captions: caption2embed_sd3(
        captions,
        [None, tokenizer_three],
        [None, text_encoder_three],
        accelerator.device,
        weight_dtype,
        args=args,
        token_length=args.token_length,
    )

    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    train_dataset = LongCaptionIterableDataset(
        dataset=raw_train_dataset,
        transform=train_transforms,
        caption_column=args.caption_column,
        image_column=args.image_column,
        path_column=args.path_column,
        source_column=args.source_column,
        source_filter=source_filter,
        image_root=args.image_root,
        skip_missing_images=args.skip_missing_images,
    )

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        captions = [example["captions"] if random.random() >= args.component_dropout else "" for example in examples]
        return {"pixel_values": pixel_values, "captions": captions}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    try:
        first_batch = next(iter(train_dataloader))
        logger.info(
            "Loaded first batch successfully: pixel_values shape=%s",
            tuple(first_batch["pixel_values"].shape),
        )
    except StopIteration as exc:
        raise RuntimeError(
            "No valid training samples found. Check `--image_root`, `--source_filter`, and dataset columns."
        ) from exc

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = args.num_update_steps_per_epoch
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    decomposer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        decomposer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    def save_decomposer_weights(save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        state_dict = {
            key: value.detach().cpu()
            for key, value in unwrap_model(decomposer).state_dict().items()
        }
        save_file(state_dict, os.path.join(save_dir, "model.safetensors"))

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt", None)

        init_kwargs = {}
        if args.report_to in {"wandb", "all"}:
            wandb_kwargs = {}
            if args.wandb_run_name:
                wandb_kwargs["name"] = args.wandb_run_name
            if args.wandb_entity:
                wandb_kwargs["entity"] = args.wandb_entity
            if wandb_kwargs:
                init_kwargs["wandb"] = wandb_kwargs

        if init_kwargs:
            accelerator.init_trackers(args.tracker_project_name, config=tracker_config, init_kwargs=init_kwargs)
        else:
            accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    # logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
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
            global_step = int(path.split("-")[1])

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

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(decomposer):
                kd_loss = None
                diff_loss = None
                with torch.no_grad():
                    # Convert images to latent space
                    pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                    model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                    model_input = model_input.to(dtype=weight_dtype)

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(model_input)
                    bsz = model_input.shape[0]
                    # Sample a random timestep for each image
                    # for weighting schemes where we sample timesteps non-uniformly
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                    # Add noise according to flow matching.
                    # zt = (1 - texp) * x + texp * z1
                    sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                    noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
                    if args.precondition_outputs >= 2:
                        prompt_embeds, pooled_prompt_embeds = encode_prompt(
                            text_encoders,
                            tokenizers,
                            batch['captions'],
                            args.token_length,
                            device=None,
                        )
                        org_model_pred = transformer(
                            hidden_states=noisy_model_input,
                            timestep=timesteps,
                            encoder_hidden_states=prompt_embeds,
                            pooled_projections=pooled_prompt_embeds,
                            return_dict=False,
                        )[0]

                    # Get the text embedding for conditioning
                    encoder_hidden_states_pre = caption2embed_simple(batch['captions'])
                    # encoder_hidden_states_clip = encoder_hidden_states_pre['encoder_hidden_states_clip_concat']
                    # encoder_hidden_states_clip_pooled = encoder_hidden_states_pre['encoder_hidden_states_clip_pooled']
                    encoder_hidden_states_t5 = encoder_hidden_states_pre["encoder_hidden_states_t5"]
                    clip_prompt_embeds, pooled_prompt_embeds = clip_encode_prompt(
                        text_encoders,
                        tokenizers,
                        batch['captions'],
                        args.token_length,
                        device=None,
                    )
                token_hidden_states = decomposer(encoder_hidden_states_t5)
                decomposed_encoder_hidden_states = torch.cat(token_hidden_states).to(dtype=weight_dtype)
                exp_clip_prompt_embeds = torch.cat([clip_prompt_embeds]*args.num_components)
                prompt_embeds = torch.cat([exp_clip_prompt_embeds, decomposed_encoder_hidden_states], dim=-2)

                # Predict the noise residual
                model_pred_batch = transformer(
                    hidden_states=torch.cat([noisy_model_input]*args.num_components),
                    timestep=torch.cat([timesteps]*args.num_components),
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=torch.cat([pooled_prompt_embeds]*args.num_components),
                    return_dict=False,
                )[0]
                model_pred = sum(model_pred_batch.chunk(args.num_components)) / args.num_components

                # flow matching loss
                if args.precondition_outputs > 0:
                    target = model_input
                else:
                    target = noise - model_input

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                loss = 0.0
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                if args.precondition_outputs >= 2:
                    kd_loss = torch.mean(
                        (weighting.float() * (model_pred.float() - org_model_pred.float()) ** 2).reshape(target.shape[0], -1),
                        1,
                    ).mean()
                    loss += kd_loss


                # Follow: Section 5 of https://huggingface.co/papers/2206.00364.
                # Preconditioning of the model outputs.
                if args.precondition_outputs > 0:
                    model_pred = model_pred * (-sigmas) + noisy_model_input

                # Compute regular loss.
                if args.precondition_outputs != 2:
                    diff_loss = torch.mean(
                        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                        1,
                    ).mean()
                    loss += diff_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = decomposer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        save_decomposer_weights(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        images = log_validation(
                            decomposer, caption2embed_simple, transformer, vae, text_encoder_one, text_encoder_two, text_encoder_three, tokenizer_one, tokenizer_two, tokenizer_three, args, accelerator, weight_dtype, epoch
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if diff_loss is not None:
                logs["diff"] = diff_loss.detach().item()
            if kd_loss is not None:
                logs["kd"] = kd_loss.detach().item()
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_decomposer_weights(args.output_dir)
        logger.info(f"Saved final decomposer weights to {os.path.join(args.output_dir, 'model.safetensors')}")
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
