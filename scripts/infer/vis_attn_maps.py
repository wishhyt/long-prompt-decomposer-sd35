import os
import sys

# BOOTSTRAP_PATHS: allow running scripts directly without installing the package.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import os
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from einops import rearrange
from PIL import Image
from safetensors.torch import load_file
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
)
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    AutoTokenizer,
    T5EncoderModel,
)
from lpd.pipeline_prompt_decomposition import PromptDecomposePipeline
from scripts.train.long_prompt_decomposition import TextDecomposer
from diffusers.models.attention_processor import AttnProcessor
from lpd.attention_map_diffusers import (
    init_pipeline,
    hook_function,
    replace_call_method_for_unet,
    save_attention_image,
    save_attention_maps,
)
from lpd.tools import caption2embed, ids2embed


class DecomposingAttnProcessor(nn.Module):

    def __init__(self, num_components):
        super().__init__()
        self.num_components = num_components

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states = None,
        attention_mask = None,
        temb = None,
        height: int = None,
        width: int = None,
        timestep: torch.Tensor = None,
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

        # attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

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
        # query = query.reshape(batch_times_comp * attn.heads, encoder_sequence_length, dim_head)
        # key = key.reshape(batch_times_comp * attn.heads, encoder_sequence_length, dim_head)
        # value = value.reshape(batch_times_comp * attn.heads, encoder_sequence_length, dim_head)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * attn.scale

        attention_scores = attention_scores.view(self.num_components, batch_size, attn.heads, sequence_length, encoder_sequence_length)
        # attention_scores = attention_scores.view(self.num_components, batch_size * attn.heads, sequence_length, encoder_sequence_length)

        attention_weights = F.softmax(attention_scores, dim=0)
        # # compositional entropy
        # compositional_entropy = attention_weights.sum(dim=0).mean()
        attention_weights = attention_weights.view(batch_times_comp, attn.heads, sequence_length, encoder_sequence_length)

        # Normalize along last dimension (enc_seq_len) by dividing by sum along last dimension
        attention_weights_sum = attention_weights.sum(dim=-1, keepdim=True)  # [batch_times_comp, num_heads, seq_len, 1]
        attention_weights = attention_weights / (attention_weights_sum + 1e-8)  # Add small epsilon to avoid division by zero

        ####################################################################################################
        if hasattr(self, "store_attn_map"):
            self.attn_map = rearrange(
                attention_weights,
                'batch attn_head (h w) attn_dim -> batch attn_head h w attn_dim ',
                h=height
            ) # detach height*width
            self.timestep = int(timestep.item())
        ####################################################################################################

        # Apply attention to values
        hidden_states = torch.matmul(attention_weights, value)
        hidden_states = hidden_states.transpose(1, 2).contiguous().view(
            batch_times_comp, sequence_length, hidden_dim
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_times_comp, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class DecomposingAttnProcessor_pad(nn.Module):

    def __init__(self, num_components):
        super().__init__()
        self.num_components = num_components
        self.vanilla_proc = AttnProcessor()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states = None,
        attention_mask = None,
        temb = None,
        height: int = None,
        width: int = None,
        timestep: torch.Tensor = None,
        decompose_mask: nn.Module = None,
        pad_length = 4,     # length 4 for good visualization
        decomp = True,
        attn_map = None,
    ):
        assert encoder_hidden_states is not None
        if not decomp:
            return self.vanilla_proc(attn, hidden_states, encoder_hidden_states)

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

        # attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

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
        # query = query.reshape(batch_times_comp * attn.heads, encoder_sequence_length, dim_head)
        # key = key.reshape(batch_times_comp * attn.heads, encoder_sequence_length, dim_head)
        # value = value.reshape(batch_times_comp * attn.heads, encoder_sequence_length, dim_head)

        if attn_map is not None:
            attention_weights = attn_map[self.name].to(value.device, dtype=value.dtype)
        else:
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) * attn.scale
            attention_scores_pad = torch.matmul(query, key_pad.transpose(-2, -1)) * attn.scale

            attention_scores = attention_scores.view(self.num_components, batch_size, attn.heads, sequence_length, encoder_sequence_length)
            attention_scores_pad = attention_scores_pad.view(self.num_components, batch_size, attn.heads, sequence_length, pad_length)
            # attention_scores = attention_scores.view(self.num_components, batch_size * attn.heads, sequence_length, encoder_sequence_length)

            attention_weights = F.softmax(attention_scores, dim=0)
            attention_weights_pad = F.softmax(attention_scores_pad, dim=0) * self.num_components / pad_length * (1-attention_weights)

            attention_weights = attention_weights.view(batch_times_comp, attn.heads, sequence_length, encoder_sequence_length)
            attention_weights_pad = attention_weights_pad.view(batch_times_comp, attn.heads, sequence_length, pad_length)
            attention_weights = torch.cat([attention_weights, attention_weights_pad], dim=-1)

            ####################################################################################################
            if hasattr(self, "store_attn_map"):
                self.attn_map = attention_weights
                self.timestep = int(timestep.item())
            ####################################################################################################

        # Apply attention to values
        hidden_states = torch.matmul(attention_weights, value)
        hidden_states = hidden_states.transpose(1, 2).contiguous().view(
            batch_times_comp, sequence_length, hidden_dim
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_times_comp, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class DecomposingAttnProcessor_pad_tmp(nn.Module):

    def __init__(self, num_components):
        super().__init__()
        self.num_components = num_components
        self.temperature = nn.Parameter(torch.ones(1))

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states = None,
        attention_mask = None,
        temb = None,
        height: int = None,
        width: int = None,
        timestep: torch.Tensor = None,
        decompose_mask: nn.Module = None,
        pad_length = 4,     # length 4 for good visualization
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

        # attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

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
        # query = query.reshape(batch_times_comp * attn.heads, encoder_sequence_length, dim_head)
        # key = key.reshape(batch_times_comp * attn.heads, encoder_sequence_length, dim_head)
        # value = value.reshape(batch_times_comp * attn.heads, encoder_sequence_length, dim_head)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.temperature + 1e-8)
        attention_scores_pad = torch.matmul(query, key_pad.transpose(-2, -1)) * attn.scale

        attention_scores = attention_scores.view(self.num_components, batch_size, attn.heads, sequence_length, encoder_sequence_length)
        attention_scores_pad = attention_scores_pad.view(self.num_components, batch_size, attn.heads, sequence_length, pad_length)
        # attention_scores = attention_scores.view(self.num_components, batch_size * attn.heads, sequence_length, encoder_sequence_length)

        attention_weights = F.softmax(attention_scores, dim=0)
        attention_weights_pad = F.softmax(attention_scores_pad, dim=0) * self.num_components / pad_length * (1-attention_weights)

        attention_weights = attention_weights.view(batch_times_comp, attn.heads, sequence_length, encoder_sequence_length)
        attention_weights_pad = attention_weights_pad.view(batch_times_comp, attn.heads, sequence_length, pad_length)
        attention_weights = torch.cat([attention_weights, attention_weights_pad], dim=-1)

        ####################################################################################################
        if hasattr(self, "store_attn_map"):
            self.attn_map = rearrange(
                attention_weights,
                'batch attn_head (h w) attn_dim -> batch attn_head h w attn_dim ',
                h=height
            ) # detach height*width
            self.timestep = int(timestep.item())
        ####################################################################################################

        # Apply attention to values
        hidden_states = torch.matmul(attention_weights, value)
        hidden_states = hidden_states.transpose(1, 2).contiguous().view(
            batch_times_comp, sequence_length, hidden_dim
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_times_comp, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class DecomposingAttnProcessor_pool(nn.Module):

    def __init__(self, num_components, return_entropy = False, learn_temperature = False, dtype = torch.float32):
        super().__init__()
        self.name = None
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
        height: int = None,
        width: int = None,
        timestep: torch.Tensor = None,
        decompose_mask: nn.Module = None,
        attn_map = None,
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

        if attn_map is None:
            if self.learn_temperature:
                attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.temperature + 1e-8)
            else:
                attention_scores = torch.matmul(query, key.transpose(-2, -1)) * attn.scale

            attention_scores = attention_scores.view(self.num_components, batch_size, attn.heads, sequence_length, encoder_sequence_length)
            attention_scores_pool = attention_scores.mean(dim=-1, keepdim=True)

            # apply soft-max
            attention_weights_pooled = F.softmax(attention_scores_pool, dim=0)
            attention_weights = F.softmax(attention_scores, dim=-1) * attention_weights_pooled

            if self.return_entropy:
                compositional_entropy = (attention_weights_pooled*(attention_weights_pooled).log()).sum(dim=0)

            attention_weights = attention_weights.view(batch_times_comp, attn.heads, sequence_length, encoder_sequence_length)

            ####################################################################################################
            if hasattr(self, "store_attn_map"):
                self.attn_map = rearrange(
                    attention_weights,
                    'batch attn_head (h w) attn_dim -> batch attn_head h w attn_dim ',
                    h=height
                ) # detach height*width
                self.timestep = int(timestep.item())
            ####################################################################################################
        else:
            attention_weights = attn_map[self.name].to(value.device, dtype=value.dtype)

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


def transposeAttnMaps(attn_map):
    attn_map_t = {}
    timesteps = list(attn_map.keys())
    layers = list(attn_map[timesteps[0]].keys())
    for layer in layers:
        layer_map = {}
        for ts in timesteps:
            layer_map[ts] = attn_map[ts][layer]
        attn_map_t[layer] = layer_map
    return attn_map_t


def save_attention_image_cat_decomp(attn_map, num_tokens, batch_dir):
    cat_attention_image = []
    for i, a in enumerate(attn_map[:1]):
        a = (a-a.min()) / (a.max() - a.min()) * 255
        # a = a * 255
        # a = (a-a.min()).clip(0.0, 1.0) * 255
        cat_attention_image.append(a)
    cat_attention_image = torch.cat(cat_attention_image, dim=-1).to(torch.float32)
    # cat_attention_np = cat_attention_image.numpy().astype(np.int8)
    # Image.fromarray(cat_attention_np, mode='L').save(f"{batch_dir}_attn_map.png")
    return cat_attention_image


def save_attention_maps_decomp(attn_maps, num_tokens, base_dir='attn_maps', unconditional=True):
    
    os.makedirs(base_dir, exist_ok=True)
    
    total_attn_map = list(list(attn_maps.values())[0].values())[0].sum(1)
    if unconditional:
        total_attn_map = total_attn_map.chunk(2)[1]  # (batch, height, width, attn_dim)
    total_attn_map = total_attn_map.permute(0, 3, 1, 2)
    total_attn_map = torch.zeros_like(total_attn_map, dtype=torch.float32)
    spatial_shape = total_attn_map.shape[-2:]
    total_attn_map_number = 0
    
    for timestep, layers in attn_maps.items():
        timestep_dir = os.path.join(base_dir, f'{timestep}')
        os.makedirs(timestep_dir, exist_ok=True)
        layer_attn_map = torch.zeros_like(total_attn_map, dtype=torch.float32)
        layer_attn_map_number = 0
        
        for layer, attn_map in layers.items():
            print(attn_map.shape)
            exit()
            attn_map = attn_map.mean(1).permute(0, 3, 1, 2)
            if unconditional:
                attn_map = attn_map.reshape(-1, 2, *attn_map.shape[-3:]).chunk(2, dim=1)[1].sum(dim=1)
            resized_attn_map = F.interpolate(attn_map, size=spatial_shape, mode='bilinear', align_corners=False)
            total_attn_map += resized_attn_map
            total_attn_map_number += 1
            layer_attn_map += resized_attn_map
            layer_attn_map_number += 1
            
            # # Save inner-most maps
            # layer_dir = os.path.join(timestep_dir, f'{layer}')
            # os.makedirs(layer_dir, exist_ok=True)
            # layer_map = []
            # for batch, attn in enumerate(attn_map):
            #     batch_dir = f"{layer_dir}/{batch}_component"
            #     layer_cat_map = save_attention_image_cat_decomp(attn, num_tokens, batch_dir)
            #     layer_map.append(layer_cat_map)
            # layer_map = torch.cat(layer_map, dim=0)
            # cat_attention_np = layer_map.numpy().astype(np.int8)
            # Image.fromarray(cat_attention_np, mode='L').save(f"{layer_dir}/components_attn_map.png")

        layer_attn_map /= layer_attn_map_number
        cat_component_map = []
        for batch, attn_map in enumerate(layer_attn_map):
            batch_dir = f"{timestep_dir}/{batch}_component"
            cat_attn_map = save_attention_image_cat_decomp(attn_map, num_tokens, batch_dir)
            cat_component_map.append(cat_attn_map)
        cat_component_map = torch.cat(cat_component_map, dim=1)
        cat_attention_np = cat_component_map.numpy().astype(np.int8)
        Image.fromarray(cat_attention_np, mode='L').save(f"{timestep_dir}/components_attn_map.png")
    
    total_attn_map /= total_attn_map_number
    cat_component_map = []
    for batch, attn_map in enumerate(total_attn_map):
        batch_dir = f"{base_dir}/{batch}_component"
        cat_attn_map = save_attention_image_cat_decomp(attn_map, num_tokens, batch_dir)
        cat_component_map.append(cat_attn_map)
    cat_component_map = torch.cat(cat_component_map, dim=1)
    cat_attention_np = cat_component_map.numpy().astype(np.int8)
    Image.fromarray(cat_attention_np, mode='L').save(f"{base_dir}/components_attn_map.png")




model_id = "//zoo/stabilityai/stable-diffusion-2-1-base"
base_pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.bfloat16).to("cuda")

prompts = [
    '''The image showcases a box of Advocate flea and \
heartworm treatment for dogs. The box is (1) predominantly \
orange in color, standing upright against a white background. \
The front of the box is (2) adorned with a photograph of a \
black and white dog, who appears to be (3) standing on a \
grassy field. The dog's gaze is directed towards the camera, \
adding a sense of engagement to the image. Overall, the box is \
designed to provide essential information about the product in \
a clear and concise manner, while also emphasizing the \
importance of safety.''',
    '''In the image, a Great Spotted Woodpecker, a bird \
known for its distinctive black and white plumage, is captured \
in a moment of feeding. The bird is (1) perched on a wire bird \
feeder, which is (2) filled with a mix of brown and white \
birdseed. The feeder is suspended from a tree branch, \
providing a perfect spot for the woodpecker to access its meal. \
The background is (3) a blur of green foliage, suggesting that \
this scene is taking place in a natural, outdoor setting. The \
woodpecker is in the process of eating from the feeder.''',
    '''The image is a digital artwork depicting a fantastical \
winter scene. At the center of the composition is (1) a large, \
circular archway made of white, snow-covered branches. \
The archway is (2) adorned with red berries and green \
leaves, adding a touch of color to the otherwise monochromatic \
scene. Beyond the archway, a snowy landscape unfolds. The \
ground is (3) blanketed in a thick layer of snow, and the \
trees are dusted with snow, their branches heavy with \
the snow. The overall effect is a serene, winter wonderland.''',
    '''The image presents a vibrant Christmas sale \
advertisement. Dominating the center of the image is (1) a \
heart shape, filled with a variety of shopping items. The \
items, (2) depicted in shades of red, green, and white, \
include dresses, purses, and shoes, suggesting a wide range \
of clothing and accessories available for purchase. The heart is \
set (3) against a backdrop of red and white stripes, adding a \
festive touch to the advertisement. The overall layout of the \
image suggests a well-organized and enticing shopping event.''',
    '''The image presents a 3D rendering of a horse, captured in a profile view. The horse is depicted \
in a state of motion, with its mane and tail flowing behind it. The horse's body is composed \
of a network of lines and curves, suggesting a complex mechanical structure. This intricate \
design is further emphasized by the presence of gears and other mechanical components, which \
are integrated into the horse's body. The background of the image is a dark blue, providing a \
stark contrast to the horse and its mechanical components. The overall composition of the image \
suggests a blend of organic and mechanical elements, creating a unique and intriguing visual.''',
    '''The image presents a close-up view of a human eye, which is the central focus. The eye is \
surrounded by a vibrant array of flowers, predominantly in shades of blue and purple. These \
flowers are arranged in a semi-circle around the eye, creating a sense of depth and perspective. \
The background of the image is a dark blue sky, which contrasts with the bright colors of the \
flowers and the eye itself. The overall composition of the image suggests a theme of nature and beauty.''',
    '''The image presents a detailed illustration of a submarine, which is the central focus of the art-\
work. The submarine is depicted in a three-quarter view, with its bow facing towards the right \
side of the image. The submarine is constructed from wood, giving it a rustic and aged appear-\
ance. It features a dome-shaped conning tower, which is a common feature on submarines, and a \
large propeller at the front. The submarine is not alone in the image. It is surrounded by a variety \
of sea creatures, including fish and sharks, which are swimming around it. These creatures add \
a sense of life and movement to the otherwise static image of the submarine. The background of \
the image is a light beige color, which provides a neutral backdrop that allows the submarine and \
the sea creatures to stand out. However, the background is not devoid of detail. It is adorned with \
various lines and text, which appear to be a map or a chart of some sort. This adds an element \
of intrigue to the image, suggesting that the submarine might be on a mission or an expedition. \
Overall, the image is a detailed and intricate piece of art that captures the essence of a submarine \
voyage, complete with the submarine, the sea creatures, and the map in the background. It's a \
snapshot of a moment in time, frozen in the image, inviting the viewer to imagine the stories and \
adventures that might be taking place beneath the surface of the water.''',
    '''In the image, there's a charming scene featuring a green frog figurine. The frog, with its body \
painted in a vibrant shade of green, is the main subject of the image. It's wearing a straw hat, \
adding a touch of whimsy to its appearance. The frog is positioned in front of a white window, \
which is adorned with a green plant, creating a harmonious color palette with the frog's body. The \
frog appears to be looking directly at the camera, giving the impression of a friendly encounter. \
The overall image exudes a sense of tranquility and simplicity.''',
    '''The image portrays a female character with a fantasy-inspired design. She has long, dark hair \
that cascades down her shoulders. Her skin is pale, and her eyes are a striking shade of blue. \
The character's face is adorned with intricate gold and pink makeup, which includes elaborate \
patterns and designs around her eyes and on her cheeks. Atop her head, she wears a crown made \
of gold and pink roses, with the roses arranged in a circular pattern. The crown is detailed, with \
each rose appearing to have a glossy finish. The character's attire consists of a gold and pink dress \
that is embellished with what appears to be feathers or leaves, adding to the fantasy aesthetic. The \
background of the image is dark, which contrasts with the character's pale skin and the bright \
colors of her makeup and attire. The lighting in the image highlights the character's features \
and the details of her makeup and attire, creating a dramatic and captivating effect. There are no \
visible texts or brands in the image. The style of the image is highly stylized and artistic, with \
a focus on the character's beauty and the intricate details of her makeup and attire. The image \
is likely a digital artwork or a concept illustration, given the level of detail and the fantastical \
elements present.''',
'''The image captures a scene of a large, modern building perched on a cliff. The building, painted \
in shades of blue and gray, stands out against the backdrop of a cloudy sky. The cliff itself is \
a mix of dirt and grass, adding a touch of nature to the otherwise man-made structure. In the \
foreground, a group of people can be seen walking along a path that leads up to the building. \
Their presence adds a sense of scale to the image, highlighting the grandeur of the building. The \
sky above is filled with clouds, casting a soft, diffused light over the scene. This light enhances \
the colors of the building and the surrounding landscape, creating a visually striking image. \
Overall, the image presents a harmonious blend of architecture and nature, with the modern \
building seamlessly integrated into the natural landscape.''',
'''The image is a digital artwork featuring a female character standing in a rocky environment. The \
character is dressed in a fantasy-style armor with a predominantly dark color scheme, highlighted \
by accents of blue and purple. The armor includes a corset-like bodice, a skirt, and arm guards, \
all adorned with intricate designs and patterns. The character's hair is styled in a high ponytail, \
and she has a serious expression on her face. The armor is illuminated by a purple glow, which \
appears to emanate from the character's body, creating a contrast against the darker elements of \
the armor and the surrounding environment. The glow also casts a soft light on the character's \
face and the armor, enhancing its details and textures. The background consists of a rocky land-\
scape with a purple hue, suggesting a magical or otherworldly setting. The rocks are jagged and \
uneven, with some areas appearing to be on fire, adding to the dramatic and intense atmosphere of \
the image. There are no visible texts or logos in the image, and the style of the artwork is realistic \
with a focus on fantasy elements. The image is likely intended for a gaming or fantasy-themed \
context, given the character's attire and the overall aesthetic.''',
'''The image presents a scene of elegance and luxury. Dominating the center of the image is a \
brown Louis Vuitton suitcase, standing upright. The suitcase is adorned with a gold handle and a \
gold lock, adding a touch of opulence to its appearance. Emerging from the top of the suitcase is \
a bouquet of pink and white roses, interspersed with green leaves. The roses, in full bloom, seem \
to be spilling out of the suitcase, creating a sense of abundance and luxury. The entire scene is \
set against a white background, which accentuates the colors of the suitcase and the roses. The \
image does not contain any text or other discernible objects. The relative position of the objects \
is such that the suitcase is in the center, with the bouquet of roses emerging from its top.''',
'''The image captures the grandeur of the Toledo Town Hall, a renowned landmark in Toledo, \
Spain. The building, constructed from stone, stands tall with two prominent towers on either \
side. Each tower is adorned with a spire, adding to the overall majesty of the structure. The \
facade of the building is punctuated by numerous windows and arches, hinting at the intricate \
architectural details within. In the foreground, a pink fountain adds a splash of color to the scene. \
A few people can be seen walking around the fountain, their figures small in comparison to the \
imposing structure of the town hall. The sky above is a clear blue, providing a beautiful backdrop \
to the scene. The image is taken from a low angle, which emphasizes the height of the town hall \
and gives the viewer a sense of being in the scene. The perspective also allows for a detailed \
view of the building and its surroundings. The image does not contain any discernible text. The \
relative positions of the objects confirm that the town hall is the central focus of the image, with \
the fountain and the people providing context to its location.''',
]


pipe = PromptDecomposePipeline.from_pretrained(
    model_id,
    # text_encoder=base_pipe.text_encoder,
    # tokenizer=base_pipe.tokenizer,
    unet=base_pipe.unet,
    vae=base_pipe.vae,
    safety_checker=None,
    torch_dtype=torch.bfloat16
).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
tokenizer_t5 = AutoTokenizer.from_pretrained("//zoo/google-t5/t5-large", model_max_length=512)
text_encoder_t5 = T5EncoderModel.from_pretrained("//zoo/google-t5/t5-large",).to('cuda', dtype=torch.bfloat16)

ckpt = 20000
num_components = 4
num_tokens = 64
test_path = "jdb/4_component_seg_mean_T64_ResamplerV3-1024-8-6"
os.makedirs(f"//samples/long_PDD/sd2/{test_path}", exist_ok=True)
test_sd = load_file(
    f"//logs/long_PDD/sd2/{test_path}/checkpoint-{ckpt}/model.safetensors",
device="cpu")

decomposer = TextDecomposer(
    width=1024,
    heads=8,
    layers=6,
    num_components=num_components,
    num_tokens=num_tokens
).to('cuda', dtype=torch.bfloat16)
decomposer.load_state_dict(test_sd)

caption2embed_simple = lambda captions: caption2embed(
    captions,
    [None, tokenizer_t5], [None, text_encoder_t5],
    pipe.device, pipe.dtype
)

################# dev versions #################
attn_procs = {}
unet_learnable_module_list = []
for k,v in pipe.unet.attn_processors.items():
    if '.attn2.' in k:
        attn_procs[k] = DecomposingAttnProcessor_pool(num_components)
        attn_procs[k].store_attn_map = True
        unet_learnable_module_list.append(attn_procs[k])
    else:
        attn_procs[k] = AttnProcessor()
pipe.unet.set_attn_processor(attn_procs)
################################################

# # load UNet learnable parameters: temperature or LoRA
# unet_learnable_module_list = nn.ModuleList(unet_learnable_module_list)
# unet_learnable_module_list.load_state_dict(
#     load_file(
#         f"//logs/long_PDD/sd2/{test_path}/checkpoint-{ckpt}/temperature.safetensors",
#         device="cpu"
#     )
# )

attn_maps = {}
pipe = init_pipeline(pipe, attn_maps)
sampling_steps = 25

for i, prompt in enumerate(prompts):
    generator = torch.Generator(device=pipe.device).manual_seed(3467)
    encoder_hidden_states = caption2embed_simple(["", prompt])
    encoder_hidden_states_clip = encoder_hidden_states['encoder_hidden_states_clip_concat']
    encoder_hidden_states_t5 = encoder_hidden_states["encoder_hidden_states_t5"]
    # encoder_hidden_states = torch.cat([encoder_hidden_states_clip, encoder_hidden_states_t5], dim=1)
    encoder_hidden_states = encoder_hidden_states_t5
    # print(pipe(
    #     decomposer,
    #     prompt_embeds = encoder_hidden_states[1:],
    #     negative_prompt_embeds = encoder_hidden_states[:1],
    #     num_inference_steps=sampling_steps,
    #     generator=generator,
    # ))
    # continue
    image = pipe(
        decomposer,
        prompt_embeds = encoder_hidden_states[1:],
        negative_prompt_embeds = encoder_hidden_states[:1],
        num_inference_steps=sampling_steps,
        generator=generator,
    ).images[0]

    image.save(f"//samples/long_PDD/sd2/{test_path}/{ckpt}_{i}.png")
    save_attention_maps_decomp(
        transposeAttnMaps(attn_maps),
        num_tokens,
        base_dir=f"//samples/long_PDD/sd2/{test_path}/{ckpt}_attn_maps_{i}",
        unconditional=True
    )

# for i, prompt in enumerate(prompts):
#     generator = torch.Generator(device=pipe.device).manual_seed(3467)
#     image = pipe.decompose(
#         decomposer,
#         prompt=prompt,
#         num_inference_steps=30,
#         guidance_scale=7.5,
#         generator=generator,
#     ).images

#     grid_image = Image.new('RGB', (512 * num_components, 512))
#     for index, img in enumerate(image):
#         col = index % num_components
#         row = index // num_components

#         x_offset = col * 512
#         y_offset = row * 512

#         grid_image.paste(img, (x_offset, y_offset))
#     grid_image.save(f"//samples/sd2/{test_path}/{i}_components.png")
#     save_attention_maps_decomp(
#         transposeAttnMaps(attn_maps),
#         num_tokens,
#         base_dir=f"//samples/sd2/{test_path}/decompose_attn_maps_{i}",
#         unconditional=True
#     )