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
    UniPCMultistepScheduler,
)
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    AutoTokenizer,
    T5EncoderModel,
)
from lpd.pipeline_lpd_ct5 import PromptDecomposePipeline
from scripts.train.lpd_ella_ct5 import TextDecomposer
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
                self.attn_map = attention_weights
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
                attn_map = attn_map.reshape(-1, 2, *attn_map.shape[-3:]).chunk(2, dim=1)[1].squeeze(1)
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
'''The image portrays a solitary knight in full plate armor, captured from a low-angle perspective that emphasizes his heroic stature. The armor is forged from burnished, dark steel, its surface intricately etched with silver filigree along the edges of the pauldrons and greaves. A single, glowing rune is visible on the center of the chest plate, casting a faint, ethereal light. The knight stands in a stoic pose on a windswept, rocky precipice, his gaze directed towards a colossal castle looming in the background. A heavy, crimson cape, its texture appearing weathered and frayed at the hem, billows dramatically behind him. The castle is a formidable gothic structure, constructed from dark, moss-covered stone that suggests great age. Its towering spires and crenelated walls are silhouetted against a tumultuous sky filled with swirling, dark gray storm clouds. A dramatic shaft of golden sunlight breaks through the cloud cover, illuminating the knight and creating a high-contrast effect with long, sharp shadows. The foreground is composed of jagged rock and sparse, hardy grass. The overall color palette is muted, dominated by shades of gray and deep blue, punctuated by the stark crimson of the cape and the golden light. The style is that of hyper-realistic digital concept art, with a focus on the textural details of the metal, stone, and fabric, creating a dramatic and epic fantasy scene.''',
'''A majestic German Shepherd with strong features, is depicted in a full-body pose sitting attentively. The entire form of the dog is rendered in a vibrant, intricate stained glass style. Each segment of its body is a separate piece of colored glass, outlined with thick, dark lines that mimic the lead strips in a real stained glass window. The colors are rich and varied, with deep ambers, brilliant golds, and earthy browns for its coat, and perhaps some striking blues or greens in the eyes to draw focus. The light source seems to be coming from behind the dog, causing the colors to glow with an inner luminescence. The texture of the glass is not uniform; some pieces are smooth and translucent, while others have a more textured, almost liquid appearance, creating a dynamic interplay of light and color across the dog's form. The focus is entirely on the dog, with no background detail to detract from the intricate glasswork of its body.''',
'''A hyper-detailed, macro shot of a human eye, presented not as an organ of sight, but as a gateway to a lost world of intricate craftsmanship. The iris is a masterfully crafted, antique horological mechanism, a complex universe of miniature, interlocking gears and cogs made from polished brass, copper, and tarnished silver. Each metallic piece is exquisitely detailed, with tiny, functional teeth that seem to pulse with a slow, rhythmic, and almost imperceptible life. The vibrant color of the iris is replaced by the warm, metallic sheen of the gears, with ruby and sapphire jewels embedded as tiny, gleaming pivots. At the center, the pupil is not a void but the deep, dark face of a miniature clock, its impossibly thin, filigreed hands frozen at a moment of profound significance. The delicate, thread-like veins in the sclera are reimagined as fine, coiling copper wires, connecting the central mechanism to the unseen power source at the edge of the frame. The entire piece is captured under a soft, focused light that highlights the metallic textures and casts deep, dramatic shadows within the complex machinery, suggesting immense depth. The background is a stark, velvety black, ensuring nothing distracts from the mesmerizing, mechanical soul of the eye.''',
'''A formidable knight stands in a powerful, regal pose, not forged in the fires of a smithy, but grown from the heart of an ancient, mystical forest. The knight's entire suit of armor is composed of living, enchanted wood, a seamless blend of natural defense and elegant design. The breastplate is sculpted from the dense, unyielding heartwood of an ironwood tree, its surface a tapestry of swirling grain and deep, textured bark that has been polished to a low, earthy luster. Intricate patterns of phosphorescent fungi and glowing moss creep along the crevices and joints of the armor, tracing the contours of the pauldrons, gauntlets, and greaves with a soft, ethereal light in shades of vibrant emerald and cool sapphire. The helmet is carved from a single, massive burl of an ancient oak, its form both protective and organic, with a visor made of tightly woven, thorny vines that conceals the knight's face entirely. From behind this natural grille, a soft, warm light emanates, suggesting a being of pure, natural energy within. The knight's sword is a single, massive thorn of petrified wood, impossibly sharp and infused with a faint, magical aura. The background is a simple, deep, and uniform black, serving to highlight the intricate details and the soft, natural glow of this woodland protector.''',
'''The image presents a 3D rendering of a horse, captured in a profile view. The horse is depicted \
in a state of motion, with its mane and tail flowing behind it. The horse's body is composed \
of a network of lines and curves, suggesting a complex mechanical structure. This intricate \
design is further emphasized by the presence of gears and other mechanical components, which \
are integrated into the horse's body. The background of the image is a dark blue, providing a \
stark contrast to the horse and its mechanical components. The overall composition of the image \
suggests a blend of organic and mechanical elements, creating a unique and intriguing visual.''',
'''The image is a surrealist, photorealistic close-up of a human eye, rendered with a fantastical and ethereal quality. The iris is a mesmerizing, deep sapphire blue, with intricate, swirling patterns of silver and gold that seem to shift and catch the light. Within the glossy, reflective pupil, a miniature galaxy of distant stars and nebulae is mirrored. Long, delicate eyelashes, dusted with a fine, shimmering silver powder, curl upwards, each strand individually defined. The eye is framed by a lush, cascading wreath of bioluminescent flowers in full, radiant bloom. Predominantly in shades of cosmic blue, royal purple, and deep indigo, the arrangement includes luminous irises with velvety petals, delicate lavender sprigs that emit a soft glow, and pansies with faces like miniature galaxies. Glistening dewdrops cling to the petals, each one reflecting the starlit scene in the eye's pupil. The background is a deep, velvety indigo night sky, filled with a dense field of twinkling stars and faint, ethereal wisps of a purple nebula. The primary light source emanates from the flowers themselves, casting a soft, magical glow that illuminates the intricate details of the eye and creates a captivating, high-contrast image. The overall style is that of a hyper-detailed digital painting, blending realism with fantasy elements to evoke a sense of wonder and natural beauty.''',
'''A vibrant hummingbird, a jewel of nature, is captured mid-hover, its form a masterpiece of photorealistic detail. The bird's tiny body is a whirlwind of motion, its wings beating so rapidly they are rendered as a translucent, shimmering blur against the soft-focus background of lush, green foliage. Every feather is meticulously defined, from the iridescent emerald and sapphire plumage on its back to the downy white of its underbelly. Sunlight catches the metallic sheen of its feathers, creating a dazzling play of light and color that shifts with every imperceptible movement. Its long, delicate beak, a needle-thin instrument of precision, is poised just before a flower, though the flower itself remains indistinct. The bird's eye, a tiny bead of polished obsidian, is sharp and intelligent, a focal point of life and energy in the composition. The overall atmosphere is one of vibrant life and ephemeral beauty, a fleeting moment of natural perfection frozen in time. The bright, high-key lighting enhances the scene's realism and imbues it with a sense of joy and vitality. The background, a soft, creamy bokeh of indistinguishable leaves and sunlight, serves to isolate the hummingbird, making it the undisputed star of the image.''',
'''A sleek, enigmatic feline, a cat of indeterminate breed, is the central figure, poised in a state of serene contemplation. Its body is not of flesh and bone, but meticulously sculpted from a complex lattice of polished, interlocking obsidian shards. Each piece is perfectly fitted against the next, creating a mosaic of deep, lustrous black that absorbs the light. The cat's form is defined by the sharp, clean edges of these volcanic glass fragments, giving its natural curves a subtle, geometric undertone. Glimmering veins of molten gold run through the cracks between the shards, glowing with a soft, internal heat that pulses rhythmically, like a slow heartbeat. These golden rivers trace the contours of the cat's muscles and skeleton, outlining its elegant spine, the delicate structure of its paws, and the graceful curve of its tail. Its eyes are two brilliant, round-cut rubies, catching an unseen light source and casting a faint, crimson glow. The whiskers are impossibly thin strands of spun platinum, fanning out from its muzzle with metallic precision. The entire figure rests upon a simple, unadorned, and dimly lit surface, ensuring that all focus remains on the cat's extraordinary construction—a masterful fusion of natural grace and exquisite, dark craftsmanship.''',
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
'''A magnificent jellyfish, a creature of ethereal beauty, commands the center of the frame, captured in a moment of serene, balletic grace. Its bell, a perfect, translucent dome, is a marvel of natural architecture, rendered with hyper-realistic detail. Through its glassy surface, the intricate, labyrinthine network of its internal structures is faintly visible, a delicate filigree of soft pinks and purples. The surface of the bell catches and refracts the light, creating a dazzling, iridescent sheen that shifts with every subtle movement. From the bell's lower edge, a cascade of tentacles descends, a symphony of color and form. Some are long and trailing, like silken ribbons of neon pink and electric blue, while others are shorter and frilled, a delicate, lacy curtain of vibrant orange and sunshine yellow. The entire creature is imbued with a gentle, bioluminescent glow, a soft, internal light that seems to pulse with a life of its own. The jellyfish is set against a backdrop of the deep, cerulean sea, the water so clear that the sunlight from above penetrates its depths, creating a brilliant, sun-drenched environment. In the lower corners of the frame, vibrant coral formations, in shades of fiery red and deep violet, add a touch of contrasting color and texture, grounding the ethereal jellyfish in a thriving, underwater ecosystem.''',
'''A close-up, almost intimate, shot of a knight's helm, but it is forged not from steel, but from the very fabric of a captured nebula. The entire helmet swirls with the deep indigos, magentas, and cyans of a distant galaxy, with miniature stars igniting and dying within its cosmic-spun material. The visor is a sheet of pure, polished obsidian, so dark it seems to drink the light, and behind it, two points of intense, white-hot starlight burn with a steady, unwavering gaze, hinting at the consciousness within. The helmet’s crest is not of feather or metal, but a standing wave of solidified light, a blade of pure energy that cuts through the dimness. Light from an unseen source catches on the helmet’s contours, not with a metallic sheen, but by causing the internal galaxies to glow brighter, the nebulae to churn, and the star-fire to pulse with a slow, silent rhythm. The surface isn't smooth but has a subtle, crystalline texture, as if space itself has been faceted and polished. The background is a simple, deep black, a void that serves only to emphasize the celestial majesty of the figure, making the knight appear as a solitary constellation in the vast emptiness of space.''',
'''A magnificent castle, seemingly carved from a single, colossal amethyst, stands in silent grandeur. Its towering spires and crenelated walls are not constructed from stone but are instead faceted and polished surfaces of the deep purple gemstone. Light from an unseen source refracts through the crystalline structure, creating a mesmerizing internal luminescence that pulses with a soft, violet glow. The castle's architecture is both familiar and fantastical, with classic medieval towers and archways rendered in the sharp, geometric lines of a cut gem. Intricate filigree patterns, like frozen lightning, are etched into the amethyst, their silver-white lines glowing with a faint, ethereal light. These patterns trace the contours of the castle, defining its gates, windows, and the delicate tracery of its highest towers. The drawbridge is a solid sheet of polished quartz, its transparent surface revealing the shimmering, crystalline depths below. The entire structure rests on a smooth, dark, and reflective surface, creating a perfect, mirrored image of the glowing amethyst castle against an endless, dark void. This masterful creation is a breathtaking fusion of formidable fortification and delicate, crystalline beauty, a fortress of light and shadow.''',
]


pipe = PromptDecomposePipeline.from_pretrained(
    model_id,
    text_encoder=base_pipe.text_encoder,
    tokenizer=base_pipe.tokenizer,
    unet=base_pipe.unet,
    vae=base_pipe.vae,
    safety_checker=None,
    torch_dtype=torch.bfloat16
).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
tokenizer_t5 = AutoTokenizer.from_pretrained("//zoo/ella/models--google--flan-t5-xl--text_encoder", model_max_length=512)
text_encoder_t5 = T5EncoderModel.from_pretrained("//zoo/ella/models--google--flan-t5-xl--text_encoder",).to('cuda', dtype=torch.bfloat16)

ckpt = 6000
num_components = 4
num_tokens = 64
test_path = "sd2/full/4_component_T64_ct5ELLA-1024-8-4_D10_43"
# "//logs/long_PDD/sd2/reward/4_component_T64_ct5ELLA-1024-8-4_D10_1021"
out_dir = f"//samples/long_PDD/{test_path}"
os.makedirs(out_dir, exist_ok=True)
test_sd = load_file(
    f"//logs/long_PDD/{test_path}/checkpoint-{ckpt}/model.safetensors",
device="cpu")

decomposer = TextDecomposer(
    width=1024,
    heads=8,
    layers=4,
    num_components=num_components,
    num_tokens=num_tokens,
    text_hidden_dim=text_encoder_t5.config.d_model,
).to('cuda', dtype=torch.bfloat16)
decomposer.load_state_dict(test_sd)

caption2embed_simple = lambda captions: caption2embed(
    captions,
    [pipe.tokenizer, tokenizer_t5], [pipe.text_encoder, text_encoder_t5],
    pipe.device, pipe.dtype, token_length=512
)

# ################ CWA #################
# attn_procs = {}
# unet_learnable_module_list = []
# for k,v in pipe.unet.attn_processors.items():
#     if '.attn2.' in k:
#         attn_procs[k] = DecomposingAttnProcessor_pool(num_components)
#         attn_procs[k].store_attn_map = True
#         unet_learnable_module_list.append(attn_procs[k])
#     else:
#         attn_procs[k] = AttnProcessor()
# pipe.unet.set_attn_processor(attn_procs)
# #######################################

# # load UNet learnable parameters: temperature or LoRA
# unet_learnable_module_list = nn.ModuleList(unet_learnable_module_list)
# unet_learnable_module_list.load_state_dict(
#     load_file(
#         f"//logs/long_PDD/sd2/{test_path}/checkpoint-{ckpt}/temperature.safetensors",
#         device="cpu"
#     )
# )

attn_maps = {}
# pipe = init_pipeline(pipe, attn_maps)
sampling_steps = 50
cfg_scale = 5.5

for i, prompt in enumerate(prompts):
    generator = torch.Generator(device=pipe.device).manual_seed(3467)
    encoder_hidden_states = caption2embed_simple(["", prompt])
    encoder_hidden_states_clip = encoder_hidden_states['encoder_hidden_states_clip_concat']
    encoder_hidden_states_t5 = encoder_hidden_states["encoder_hidden_states_t5"]
    encoder_hidden_states = encoder_hidden_states_t5
    image = pipe(
        decomposer,
        encoder_hidden_states_t5=encoder_hidden_states_t5,
        encoder_hidden_states_clip=encoder_hidden_states_clip,
        prompt_embeds = encoder_hidden_states[1:],
        negative_prompt_embeds = encoder_hidden_states[:1],
        num_inference_steps=sampling_steps,
        guidance_scale=cfg_scale,
        generator=generator,
    ).images[0]

    image.save(f"{out_dir}/{ckpt}_{i}.png")

    generator = torch.Generator(device=pipe.device).manual_seed(3467)
    image = pipe.decompose(
        decomposer,
        encoder_hidden_states_t5=encoder_hidden_states_t5,
        encoder_hidden_states_clip=encoder_hidden_states_clip,
        prompt_embeds = encoder_hidden_states[1:],
        negative_prompt_embeds = encoder_hidden_states[:1],
        num_inference_steps=sampling_steps,
        generator=generator,
        guidance_scale=cfg_scale,
        # attn_maps=attn_maps,
    ).images

    grid_image = Image.new('RGB', (512 * num_components, 512))
    for index, img in enumerate(image):
        col = index % num_components
        row = index // num_components

        x_offset = col * 512
        y_offset = row * 512

        grid_image.paste(img, (x_offset, y_offset))
    grid_image.save(f"{out_dir}/{ckpt}_{i}_components.png")
