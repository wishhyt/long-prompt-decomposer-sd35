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
from accelerate.utils import ProjectConfiguration, DataLoaderConfiguration, set_seed
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
from lpd.models import MLP, PromptResampler, PromptDecomposer


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


class DecomposingAttnProcessor(nn.Module):

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
        decompose_mask: nn.Module = None
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

        # Derive actual batch size
        batch_size = batch_times_comp // self.num_components

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = query.view(batch_times_comp, sequence_length, attn.heads, dim_head).transpose(1, 2)
        key = key.view(batch_times_comp, encoder_sequence_length, attn.heads, dim_head).transpose(1, 2)
        value = value.view(batch_times_comp, encoder_sequence_length, attn.heads, dim_head).transpose(1, 2)

        # # TODO: replace this with torch.badmm() if attention_mask is needed
        # attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        # query = query.reshape(batch_times_comp * attn.heads, encoder_sequence_length, dim_head)
        # key = key.reshape(batch_times_comp * attn.heads, encoder_sequence_length, dim_head)
        # value = value.reshape(batch_times_comp * attn.heads, encoder_sequence_length, dim_head)

        if self.learn_temperature:
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.temperature + 1e-8)
        else:
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) * attn.scale

        attention_scores = attention_scores.view(self.num_components, batch_size, attn.heads, sequence_length, encoder_sequence_length)
        # attention_scores = attention_scores.view(self.num_components, batch_size * attn.heads, sequence_length, encoder_sequence_length)

        attention_weights = F.softmax(attention_scores, dim=0)
        # # Previous version: Normalize along last dimension (enc_seq_len) by dividing by sum along last dimension
        # attention_weights_sum = attention_weights.sum(dim=-1, keepdim=True)  # [batch_times_comp, num_heads, seq_len, 1]
        # attention_weights = attention_weights / (attention_weights_sum + 1e-8)  # Add small epsilon to avoid division by zero

        if self.return_entropy:
            compositional_entropy = (attention_weights*(attention_weights).log()).sum(dim=0)
        attention_weights = attention_weights * (self.num_components / encoder_sequence_length)
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


class DecomposingAttnProcessor_pad(nn.Module):

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
        pad_length = 1,
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
        pad_hidden_states = torch.zeros_like(encoder_hidden_states).repeat(1, pad_length, 1)

        # Derive actual batch size
        batch_size = batch_times_comp // self.num_components

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            pad_hidden_states = attn.norm_encoder_hidden_states(pad_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        key_pad = attn.to_k(pad_hidden_states)
        value_pad = attn.to_v(pad_hidden_states)
        value = torch.cat([value, value_pad], dim=1)

        query = query.view(batch_times_comp, sequence_length, attn.heads, dim_head).transpose(1, 2)
        key = key.view(batch_times_comp, encoder_sequence_length, attn.heads, dim_head).transpose(1, 2)
        value = value.view(batch_times_comp, encoder_sequence_length + pad_length, attn.heads, dim_head).transpose(1, 2)
        key_pad = key_pad.view(batch_times_comp, pad_length, attn.heads, dim_head).transpose(1, 2)

        # # TODO: replace this with torch.badmm() if attention_mask is needed
        # attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        # query = query.reshape(batch_times_comp * attn.heads, encoder_sequence_length, dim_head)
        # key = key.reshape(batch_times_comp * attn.heads, encoder_sequence_length, dim_head)
        # value = value.reshape(batch_times_comp * attn.heads, encoder_sequence_length, dim_head)

        if self.learn_temperature:
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.temperature + 1e-8)
        else:
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) * attn.scale

        attention_scores_pad = torch.matmul(query, key_pad.transpose(-2, -1)) * attn.scale

        attention_scores = attention_scores.view(self.num_components, batch_size, attn.heads, sequence_length, encoder_sequence_length)
        attention_scores_pad = attention_scores_pad.view(self.num_components, batch_size, attn.heads, sequence_length, pad_length)
        # attention_scores = attention_scores.view(self.num_components, batch_size * attn.heads, sequence_length, encoder_sequence_length)

        attention_weights = F.softmax(attention_scores, dim=0)
        attention_weights_pad = F.softmax(attention_scores_pad, dim=0) * self.num_components / pad_length * (1-attention_weights)

        if self.return_entropy:
            compositional_entropy = (attention_weights*(attention_weights).log()).sum(dim=0)

        attention_weights = attention_weights.view(batch_times_comp, attn.heads, sequence_length, encoder_sequence_length)
        attention_weights_pad = attention_weights_pad.view(batch_times_comp, attn.heads, sequence_length, pad_length)
        attention_weights = torch.cat([attention_weights, attention_weights_pad], dim=-1)

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


class TextDecomposer(nn.Module):
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
        # FIXME: hard-coded to SD-2.1
        super().__init__()
        self.num_components = num_components

        # PromptResampler
        self.mask_head = PromptResampler(
            width=width,
            heads=heads,
            layers=layers,
            num_components=num_components,
            num_tokens=num_tokens,
            input_dim=text_hidden_dim,
            output_dim=unet_hidden_dim,
        )

        # Attention
        # self.mask_head = Attention(hidden_dim, num_attention_heads=1, decompose=True)

        # MLP
        # self.mask_head = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(hidden_dim, num_components)
        # )

        # head
        # self.mask_head = nn.Linear(hidden_dim, num_components)

    def forward(self, encoder_hidden_states, adopt_index=None, return_regu=False, split=False):
        bs, seq_len, hidden_dim = encoder_hidden_states.shape
        # Get the mask tensor from each DecompositionalMask instance

        if split:
            # Attention
            # token_mask = self.mask_head(encoder_hidden_states, encoder_hidden_states, output_attentions=True)

            token_mask = self.mask_head(encoder_hidden_states)
            # token_mask = F.softmax(logits, dim=-1)
            if adopt_index is None:
                adopt_index = torch.randint(0, self.num_components, (bs,), device=encoder_hidden_states.device)
            # token_mask = token_mask[torch.arange(bs),:,adopt_index].unsqueeze(-1)
            token_mask = token_mask[torch.arange(bs),adopt_index].unsqueeze(-1)
            encoder_hidden_state = token_mask * encoder_hidden_states
            encoder_hidden_state_bg = (1 - token_mask) * encoder_hidden_states
            return encoder_hidden_state, encoder_hidden_state_bg
        # token_mask_list = torch.chunk(token_mask, self.num_components, dim=-1)
        if return_regu:
            encoder_hidden_state_list, latents_regu = self.mask_head(encoder_hidden_states, return_regu=True)
            # entropy = -(token_mask * token_mask.log()).sum(dim=(-1))
            return encoder_hidden_state_list, latents_regu
        encoder_hidden_state_list = self.mask_head(encoder_hidden_states)
        return encoder_hidden_state_list


def get_module_kohya_state_dict(module, prefix: str, dtype: torch.dtype, adapter_name: str = "default"):
    kohya_ss_state_dict = {}
    for peft_key, weight in get_peft_model_state_dict(module, adapter_name=adapter_name).items():
        kohya_key = peft_key.replace("base_model.model", prefix)
        kohya_key = kohya_key.replace("lora_A", "lora_down")
        kohya_key = kohya_key.replace("lora_B", "lora_up")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_ss_state_dict[kohya_key] = weight.to(dtype)

        # Set alpha parameter
        if "lora_down" in kohya_key:
            alpha_key = f"{kohya_key.split('.')[0]}.alpha"
            kohya_ss_state_dict[alpha_key] = torch.tensor(module.peft_config[adapter_name].lora_alpha).to(dtype)

    return kohya_ss_state_dict


def log_validation(decomposer, caption2embed_simple, unet, vae, args, accelerator, weight_dtype, epoch, return_mask=False):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = PromptDecomposePipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        # text_encoder=text_encoder,
        # tokenizer=tokenizer,
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
    if return_mask:
        inputs_ids = pipeline.tokenizer(
            [args.validation_prompt, ""], max_length=pipeline.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids
        encoder_hidden_states = pipeline.text_encoder(inputs_ids.to(accelerator.device))[0]
        token_mask = accelerator.unwrap_model(decomposer)(encoder_hidden_states[0].unsqueeze(0), encoder_hidden_states[0].unsqueeze(0), split=False)

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    images = []
    compose_image = []
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    captions = ["", args.validation_prompt]
    encoder_hidden_states_pre = caption2embed_simple(captions)
    encoder_hidden_states_clip = encoder_hidden_states_pre['encoder_hidden_states_clip_concat']
    encoder_hidden_states_t5 = encoder_hidden_states_pre["encoder_hidden_states_t5"]
    # encoder_hidden_states = torch.cat([encoder_hidden_states_clip, encoder_hidden_states_t5], dim=1)
    encoder_hidden_states = encoder_hidden_states_t5
    with autocast_ctx:
        image = pipeline(
            accelerator.unwrap_model(decomposer),
            prompt_embeds = encoder_hidden_states[1:],
            negative_prompt_embeds = encoder_hidden_states[:1],
            num_inference_steps=25,
            generator=generator
        ).images[0]
    compose_image.append(image)
    with autocast_ctx:
        image = pipeline.decompose(
            accelerator.unwrap_model(decomposer),
            prompt_embeds = encoder_hidden_states[1:],
            negative_prompt_embeds = encoder_hidden_states[:1],
            num_inference_steps=25,
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
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(compose_image)
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
    torch.cuda.empty_cache()
    return images


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
        "--cwa",
        default=False,
        action="store_true",
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
        default=1,
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
    dataloader_config = DataLoaderConfiguration(dispatch_batches=False)     # optional: split_batches=True
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        # dataloader_config=dataloader_config,
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
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    tokenizer_t5 = AutoTokenizer.from_pretrained("//zoo/ella/models--google--flan-t5-xl--text_encoder", model_max_length=512)
    text_encoder_t5 = T5EncoderModel.from_pretrained("//zoo/ella/models--google--flan-t5-xl--text_encoder",)
    decomposer = TextDecomposer(
        width=args.decomposer_width,
        heads=args.decomposer_heads,
        layers=args.decomposer_layers,
        num_tokens=args.num_tokens,
        num_components=args.num_components,
        text_hidden_dim=2048
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    # teacher_unet = UNet2DConditionModel.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    # )

    # Freeze vae and unet
    unet.requires_grad_(False)
    vae.requires_grad_(False)
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
    decomposer.to(accelerator.device)
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

    # Component-wise Attention
    if args.cwa:
        attn_procs = {}
        for k,v in unet.attn_processors.items():
            if '.attn2.' in k:
                attn_procs[k] = DecomposingAttnProcessor_pool(
                    args.num_components,
                    learn_temperature=args.learn_temperature,
                    dtype=torch.float32 if weight_dtype==torch.float16 else weight_dtype
                )
            else:
                attn_procs[k] = AttnProcessor()
        unet.set_attn_processor(attn_procs)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

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

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        list(decomposer.parameters()) + [p for n, p in unet.named_parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

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
        train_dataset_jdb = load_dataset(
            '//data/JourneyDB',
            cache_dir="//data/JourneyDB/.cache",
            split="train",
            trust_remote_code=True,
            streaming=True,
        )
    if "coco" in args.train_data_dir:
        train_dataset_coco = load_dataset(
            '//data/coco2017',
            cache_dir=f"//data/coco2017/.cache",
            split="train",
            trust_remote_code=True,
            streaming=True,
        )
    if "llava" in args.train_data_dir:
        train_dataset_lcs = load_dataset(
            '//data/LLaVA-Pretrain',
            cache_dir=f"//data/LLaVA-Pretrain/.cache",
            split="train",
            trust_remote_code=True,
            streaming=True,
        )
    if "sam" in args.train_data_dir:
        train_dataset_sam = load_dataset(
            '//data/sa-1b',
            cache_dir=f"//data/sa-1b/.cache",
            split="train",
            trust_remote_code=True,
            streaming=True,
        )

    # Preprocessing the datasets.
    image_column, caption_column = 'pixel_values', 'captions'

    with open('journeydb_caption.json', 'r') as file:
        longcaption = json.load(file)

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_caption(caption, is_train=True):
        input_ids = tokenizer(caption, return_tensors="pt").input_ids
        # inputs = tokenizer(
        #     captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        # )
        return input_ids

    caption2embed_simple = lambda captions: caption2embed(captions, [tokenizer, tokenizer_t5], [text_encoder, text_encoder_t5],
                                                          accelerator.device, weight_dtype, args=args)
    ids2embed_simple = lambda batch: ids2embed(batch, [tokenizer, tokenizer_t5], [text_encoder, text_encoder_t5],
                                                          args, accelerator.device, weight_dtype)

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

        def preprocess_train(self, examples: dict):
            examples["pixel_values"] = train_transforms(
                examples[image_column].convert("RGB").resize(
                    (args.train_size, args.train_size), resample=args.interpolation
                )
            )
            return examples

        def __iter__(self):
            # Create iterators for both datasets
            iter_jdb = iter(self.jdb)
            iter_coco = iter(self.coco)
            iter_lcs = iter(self.lcs)
            iter_sam = iter(self.sam)

            while not (self.jdb_done and self.coco_done and self.lcs_done and self.sam_done):
                rnd = random.uniform(0, 1)

                try:
                    if rnd < self.probs[0]:
                        while True:
                            data = next(iter_jdb)
                            path = f"journeydb/images/{data['url']}"
                            if path in longcaption:
                                break
                        data['captions'] = longcaption[path]
                        yield self.preprocess_train(data)
                    elif rnd < self.probs[1]:
                        yield self.preprocess_train(next(iter_coco))
                    elif rnd < self.probs[2]:
                        yield self.preprocess_train(next(iter_lcs))
                    else:
                        yield self.preprocess_train(next(iter_sam))
                except StopIteration as e:
                    # If one dataset is exhausted, try another
                    if rnd < self.probs[0]:
                        self.jdb_done = True
                        if not self.coco_done:
                            yield self.preprocess_train(next(iter_coco))
                        elif not self.lcs_done:
                            yield self.preprocess_train(next(iter_lcs))
                        elif not self.sam_done:
                            yield self.preprocess_train(next(iter_sam))
                        else:
                            break
                    elif rnd < self.probs[1]:
                        self.coco_done = True
                        if not self.jdb_done:
                            yield self.preprocess_train(next(iter_jdb))
                        elif not self.lcs_done:
                            yield self.preprocess_train(next(iter_lcs))
                        elif not self.sam_done:
                            yield self.preprocess_train(next(iter_sam))
                        else:
                            break
                    elif rnd < self.probs[2]:
                        self.lcs_done = True
                        if not self.jdb_done:
                            yield self.preprocess_train(next(iter_jdb))
                        elif not self.coco_done:
                            yield self.preprocess_train(next(iter_coco))
                        elif not self.sam_done:
                            yield self.preprocess_train(next(iter_sam))
                        else:
                            break
                    else:
                        self.sam_done = True
                        if not self.jdb_done:
                            yield self.preprocess_train(next(iter_jdb))
                        elif not self.coco_done:
                            yield self.preprocess_train(next(iter_coco))
                        elif not self.lcs_done:
                            yield self.preprocess_train(next(iter_lcs))
                        else:
                            break

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    train_dataset = MixedIterableDataset([0.5, 0.6, 0.8, 1.0])

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        results = {"pixel_values": pixel_values}
        captions = [example["captions"] if random.random() >= args.component_dropout else '' for example in examples]
        # FIXME: add CLIP
        tokens_t5 = tokenizer_t5(captions, max_length=tokenizer_t5.model_max_length,
                                 padding="max_length", truncation=True, return_tensors="pt")
        results["input_ids_t5"], results["attention_mask_t5"] = tokens_t5.input_ids, tokens_t5.attention_mask
        return results

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

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
    )

    decomposer.train()
    # Prepare everything with our `accelerator`.
    if args.tune_unet or args.learn_temperature:
        unet.train()
        decomposer, unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            decomposer, unet, optimizer, train_dataloader, lr_scheduler
        )
        trainable_module = (decomposer, unet)
    else:
        decomposer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            decomposer, optimizer, train_dataloader, lr_scheduler
        )
        trainable_module = (decomposer,)

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

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("long_prompt_decompose", config=vars(args))

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
            path = args.resume_from_checkpoint
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
            if path.startswith("checkpoint"):
                accelerator.load_state(os.path.join(args.output_dir, path))
                global_step = int(path.split("-")[1])
            else:
                accelerator.load_state(args.resume_from_checkpoint)
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
    for epoch in range(first_epoch, args.num_train_epochs):
        while global_step < args.max_train_steps:
            try:
                batch = next(train_dataloader_iter)
            except (PIL.UnidentifiedImageError, OSError, IOError, Image.DecompressionBombError) as e:
                continue
            with accelerator.accumulate(*trainable_module):
                with torch.no_grad():
                    # Convert images to latent space
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                    latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    # Get the text embedding for conditioning
                    encoder_hidden_states_pre = ids2embed_simple(batch)
                    encoder_hidden_states_clip = encoder_hidden_states_pre['encoder_hidden_states_clip_concat']
                    encoder_hidden_states_t5 = encoder_hidden_states_pre["encoder_hidden_states_t5"]
                    # encoder_hidden_states = torch.cat([encoder_hidden_states_clip, encoder_hidden_states_t5], dim=1)
                    encoder_hidden_states = encoder_hidden_states_t5

                    # # Teacher predict the noise residual
                    # uncond_input_ids = torch.cat([UG_IDS] * bsz).to(dtype=torch.int64, device=accelerator.device)
                    # ug_encoder_hidden_states = text_encoder(uncond_input_ids)[0].to(dtype=weight_dtype)
                    # ug_pred = teacher_unet(noisy_latents, timesteps, ug_encoder_hidden_states).sample.repeat(args.num_components, 1, 1, 1)
                    # teacher_pred = teacher_unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Decompose the text embedding
                drop_mask = torch.rand((bsz*args.num_components,1,1,1), device=accelerator.device) > args.component_dropout
                if args.dropout_v1:
                    token_hidden_states = decomposer(torch.cat([ug_encoder_hidden_states, encoder_hidden_states], dim=0))
                    decomposed_encoder_hidden_states = torch.cat([e.chunk(2, dim=0)[1] for e in token_hidden_states]).to(dtype=weight_dtype)
                    ug_decomposed_encoder_hidden_states = torch.cat([e.chunk(2, dim=0)[0] for e in token_hidden_states]).to(dtype=weight_dtype)
                    token_hidden_states = [e.chunk(2, dim=0)[1] for e in token_hidden_states]

                    model_pred_batch = unet(
                        torch.cat([noisy_latents]*args.num_components),
                        torch.cat([timesteps]*args.num_components),
                        decomposed_encoder_hidden_states
                    ).sample
                    ug_model_pred_batch = unet(
                        torch.cat([noisy_latents]*args.num_components),
                        torch.cat([timesteps]*args.num_components),
                        ug_decomposed_encoder_hidden_states
                    ).sample
                    model_pred_batch = torch.where(drop_mask, model_pred_batch, ug_model_pred_batch)
                else:
                    token_hidden_states = decomposer(encoder_hidden_states)     # a list of length `num_components`, each element is a batch
                    decomposed_encoder_hidden_states = torch.cat(token_hidden_states).to(dtype=weight_dtype)

                    model_pred_batch = unet(
                        torch.cat([noisy_latents]*args.num_components),
                        torch.cat([timesteps]*args.num_components),
                        decomposed_encoder_hidden_states
                    ).sample

                model_pred = sum(model_pred_batch.chunk(args.num_components)) / args.num_components

                diff_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                # diff_loss = F.mse_loss(model_pred.float(), teacher_pred.float(), reduction="mean")
                loss = diff_loss

                # Compositional entropy
                if args.compositional_entropy:
                    losses_comp_entropy = []
                    for module_key in mapping_layers_stu:
                        a_stu = acts_stu[module_key]
                        tmp = -a_stu.mean()
                        losses_comp_entropy.append(tmp)
                    loss_comp_entropy = sum(losses_comp_entropy)
                    loss += loss_comp_entropy

                # # Sub-prompt similarity
                batch_decomposed_hidden_states = torch.stack(token_hidden_states,dim=1).mean(dim=2)
                prompt_norm = torch.norm(
                    batch_decomposed_hidden_states.reshape(bsz*args.num_components, -1), dim=1
                ).mean(dim=0)
                loss += args.l2_norm_coeff * prompt_norm

                normalized_batch_decomposed_hidden_states = torch.nn.functional.normalize(batch_decomposed_hidden_states, p=2, dim=2, eps=1e-8)
                cosine_sim_matrix = torch.bmm(normalized_batch_decomposed_hidden_states, normalized_batch_decomposed_hidden_states.transpose(1, 2))

                # # FIXME: it should be the <eot> token the pooler
                # pooled_encoder_hidden_states = encoder_hidden_states[:,:1,:].to(torch.float32)
                # normalized_pooled_encoder_hidden_states = torch.nn.functional.normalize(pooled_encoder_hidden_states, p=2, dim=2, eps=1e-8)
                # # pooled_output = last_hidden_state[
                # #     torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                # #     input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
                # # ]
                # semantic_sim_matrix = torch.bmm(normalized_batch_decomposed_hidden_states, normalized_pooled_encoder_hidden_states.transpose(1, 2))
                # semantic_sim = semantic_sim_matrix.pow(2).mean()
                # # loss += args.regu_weight * semantic_sim

                # # Score similarity
                # _, C, H, W = model_pred_batch.shape
                # model_pred_batch_view = model_pred_batch.view(bsz, args.num_components, -1)
                # # The score function is assumed to be Gaussian
                # cosine_sim_matrix = torch.bmm(model_pred_batch_view, model_pred_batch_view.transpose(1, 2)) / (C * H * W)

                triu_indices = torch.triu_indices(args.num_components, args.num_components, offset=1)
                upper_triangular = cosine_sim_matrix[:, triu_indices[0], triu_indices[1]]
                # ABLATION: MSE
                latents_regu = (upper_triangular).pow(2).mean()
                # ABLATION: L2
                # latents_regu = (upper_triangular).pow(2).sum().sqrt()

                loss += args.regu_weight * latents_regu

                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

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

                        if args.tune_unet:
                            unwrapped_unet = unwrap_model(unet)
                            unet_lora_state_dict = convert_state_dict_to_diffusers(
                                get_peft_model_state_dict(unwrapped_unet)
                            )

                            StableDiffusionPipeline.save_lora_weights(
                                save_directory=save_path,
                                unet_lora_layers=unet_lora_state_dict,
                                safe_serialization=True,
                            )

                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        images = log_validation(
                            decomposer, caption2embed_simple, unet, vae, args, accelerator, weight_dtype, epoch
                        )

            logs = {
                "loss": diff_loss.detach().item(),
                "regu": latents_regu.detach().item(),
                # "sim": semantic_sim.detach().item(),
                "norm": prompt_norm.detach().item(),
            }
            if args.compositional_entropy:
                logs['entropy'] = loss_comp_entropy.detach().item()
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
