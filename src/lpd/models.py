from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from typing import Any, Callable, Dict, List, Optional, Union, Tuple


class SquaredReLU(nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.square(torch.relu(x))


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.sigmoid(1.702 * input)


class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, time_embedding_dim: Optional[int] = None):
        super().__init__()

        if time_embedding_dim is None:
            time_embedding_dim = embedding_dim

        self.silu = nn.SiLU()
        self.linear = nn.Linear(time_embedding_dim, 2 * embedding_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self, x: torch.Tensor, timestep_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(timestep_embedding))
        shift, scale = emb.view(len(x), 1, -1).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class MLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.activation_fn = QuickGELUActivation()
        self.fc1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, hidden_size=1024, num_attention_heads=16, attention_dropout=0.0, decompose=False):
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.decompose = decompose
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = attention_dropout
        self.is_causal = False
        self.upcast_attention = True
        self.upcast_softmax = True

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        if not self.decompose:
            self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
            self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
            self.drop = nn.Dropout(self.dropout)
        else:
            self._init_as_identity([self.k_proj, self.q_proj])

    def _init_as_identity(self, init_layers):
        """Initialize all linear layers to behave like identity functions"""
        for layer in init_layers:
            # Initialize weight as identity matrix
            nn.init.eye_(layer.weight)
            # Initialize bias as zeros (if bias exists)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    
    def get_attention_scores(
        self,
        query: torch.Tensor, key: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        transpose_qk = False,
    ) -> torch.Tensor:
        r"""
        Compute the attention scores.

        Args:
            query (`torch.Tensor`): The query tensor.
            key (`torch.Tensor`): The key tensor.
            attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `torch.Tensor`: The attention probabilities/scores.
        """
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        softmax_dim = -2 if transpose_qk else -1
        attention_probs = attention_scores.softmax(dim=softmax_dim)
        del attention_scores    # TODO: del this del?

        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, seq_length, embed_dim = hidden_states.shape
        is_final = self.decompose
        if is_final: assert encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if is_final else hidden_states

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(encoder_hidden_states)

        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        queries = queries.reshape(batch_size * self.num_heads, seq_length, self.head_dim)
        keys = keys.reshape(batch_size * self.num_heads, seq_length, self.head_dim)

        if not is_final:
            values = self.v_proj(encoder_hidden_states)
            values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
            values = values.reshape(batch_size * self.num_heads, seq_length, self.head_dim)

        if attention_mask is not None and causal_attention_mask is not None:
            attention_mask = attention_mask + causal_attention_mask
        elif causal_attention_mask is not None:
            attention_mask = causal_attention_mask

        attn_weights = self.get_attention_scores(queries, keys, attention_mask, transpose_qk=is_final)
        if is_final:
            # attn_weights = attn_weights.reshape(batch_size, self.num_heads, seq_length, seq_length)
            # attn_weights = attn_weights.transpose(1,2).reshape(batch_size, seq_length, seq_length).contiguous()
            return attn_weights
        attn_output = torch.bmm(attn_weights, values)
        attn_output = attn_output.reshape(batch_size, self.num_heads, seq_length, self.head_dim)
        attn_output = attn_output.transpose(1,2).reshape(batch_size, seq_length, embed_dim).contiguous()

        attn_output = self.out_proj(attn_output)
        attn_output = self.drop(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, layer_norm_eps=1e-6):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.ffn = MLP(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

    def forward(self, x):
        # Self-attention
        normed_x = self.norm1(x)
        attn_output, _ = self.attention(normed_x, normed_x, normed_x)
        x = x + attn_output

        # Feed-forward
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output
        return x


class PerceiverAttentionBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, time_embedding_dim: Optional[int] = None
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("sq_relu", SquaredReLU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )

        self.ln_1 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_2 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_ff = AdaLayerNorm(d_model, time_embedding_dim)

    def attention(self, q: torch.Tensor, kv: torch.Tensor):
        attn_output, attn_output_weights = self.attn(q, kv, kv, need_weights=False)
        return attn_output, attn_output_weights

    def forward(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
        timestep_embedding: torch.Tensor = None,
    ):
        normed_latents = self.ln_1(latents, timestep_embedding)
        attn_weight = None
        attn_out = self.attention(
            q=normed_latents,
            kv=torch.cat([normed_latents, self.ln_2(x, timestep_embedding)], dim=1),
        )
        attn_weight = attn_out[1]
        attn_out = attn_out[0]
        latents = latents + attn_out
        latents = latents + self.mlp(self.ln_ff(latents, timestep_embedding))
        if attn_weight is not None:
            return latents, attn_weight
        return latents


class PerceiverAttentionBlock_v2(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, time_embedding_dim: Optional[int] = None
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("sq_relu", SquaredReLU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )

        self.ln_q_t5 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_kv_t5 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_q_clip = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_kv_clip = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_ff = AdaLayerNorm(d_model, time_embedding_dim)

    def attention(self, q: torch.Tensor, kv: torch.Tensor):
        attn_output, attn_output_weights = self.attn(q, kv, kv, need_weights=False)
        return attn_output

    def forward(
        self,
        x_t5: torch.Tensor,
        x_clip: torch.Tensor,
        latents: torch.Tensor,
        timestep_embedding: torch.Tensor = None,
    ):
        # T5
        normed_latents = self.ln_q_t5(latents, timestep_embedding)
        latents = latents + self.attention(
            q=normed_latents,
            kv=torch.cat([normed_latents, self.ln_kv_t5(x_t5, timestep_embedding)], dim=1),
        )

        # CLIP
        normed_latents = self.ln_q_clip(latents, timestep_embedding)
        latents = latents + self.attention(
            q=normed_latents,
            kv=torch.cat([normed_latents, self.ln_kv_clip(x_clip, timestep_embedding)], dim=1),
        )

        # MLP out
        latents = latents + self.mlp(self.ln_ff(latents, timestep_embedding))
        return latents


class PerceiverBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, time_embedding_dim: Optional[int] = None
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("sq_relu", SquaredReLU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )

        self.ln_1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ln_2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ln_ff = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)

    def attention(self, q: torch.Tensor, kv: torch.Tensor):
        attn_output, attn_output_weights = self.attn(q, kv, kv, need_weights=False)
        return attn_output

    def forward(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
        timestep_embedding: torch.Tensor = None,
    ):
        normed_latents = self.ln_1(latents)
        latents = latents + self.attention(
            q=normed_latents,
            kv=torch.cat([normed_latents, self.ln_2(x)], dim=1),
        )
        latents = latents + self.mlp(self.ln_ff(latents))
        return latents


class PerceiverBlock_v2(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, input_dim: int, num_components: int,
    ):
        super().__init__()
        self.num_components = num_components

        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("sq_relu", SquaredReLU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )

        self.proj_kv = nn.Linear(input_dim, d_model * num_components)

        self.ln_1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ln_2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ln_ff = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)

    def attention(self, q: torch.Tensor, kv: torch.Tensor):
        attn_output, attn_output_weights = self.attn(q, kv, kv, need_weights=False)
        return attn_output

    def forward(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
    ):
        bs, seq_length = x.shape[0], x.shape[1]
        width = latents.shape[-1]

        normed_latents = self.ln_1(latents)
        x = self.proj_kv(x).reshape(bs, seq_length, self.num_components, width)
        x = x.transpose(1, 2).reshape(bs*self.num_components, seq_length, width)

        latents = latents + self.attention(
            q=normed_latents,
            kv=self.ln_2(x),
        )
        latents = latents + self.mlp(self.ln_ff(latents))
        return latents


class PerceiverBlock_v3(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("sq_relu", SquaredReLU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )

        self.ln_q_t5 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ln_kv_t5 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ln_q_clip = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ln_kv_clip = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ln_ff = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)

    def attention(self, q: torch.Tensor, kv: torch.Tensor):
        attn_output, attn_output_weights = self.attn(q, kv, kv, need_weights=False)
        return attn_output

    def forward(
        self,
        x_t5: torch.Tensor,
        x_clip: torch.Tensor,
        latents: torch.Tensor,
        timestep_embedding: torch.Tensor = None,
    ):
        # T5
        normed_latents = self.ln_q_t5(latents)
        latents = latents + self.attention(
            q=normed_latents,
            kv=torch.cat([normed_latents, self.ln_kv_t5(x_t5)], dim=1),
        )

        # CLIP
        normed_latents = self.ln_q_clip(latents)
        latents = latents + self.attention(
            q=normed_latents,
            kv=torch.cat([normed_latents, self.ln_kv_clip(x_clip)], dim=1),
        )

        # MLP out
        latents = latents + self.mlp(self.ln_ff(latents))
        return latents


class PerceiverResampler_ct5(nn.Module):
    '''
    Concat flan-t5-large and CLIP-H
    '''
    def __init__(
        self,
        width,
        layers,
        heads,
        num_tokens,
        num_components,
        input_dim,
        output_dim=None,
        time_embedding_dim: Optional[int] = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.num_components = num_components
        self.latents = nn.Parameter(width**-0.5 * torch.randn(num_components, num_tokens, width))
        self.time_aware_linear = nn.Linear(
            time_embedding_dim or width, width, bias=True
        )

        self.proj_in = nn.Linear(input_dim, width)

        self.perceiver_blocks = nn.Sequential(
            *[
                PerceiverBlock(
                    width, heads, time_embedding_dim=time_embedding_dim
                )
                for _ in range(layers)
            ]
        )

        if self.output_dim is not None:
            self.proj_out = nn.Sequential(
                nn.Linear(width, output_dim), nn.LayerNorm(output_dim)
            )

    def forward(self, x: torch.Tensor, timestep_embedding: torch.Tensor = None):
        # # batchify: [B * num_components, num_tokens, d]
        bs = x.shape[0]
        # learnable_latents = self.latents.unsqueeze(dim=0).repeat(len(x), 1, 1, 1)
        # latents = learnable_latents.reshape(len(x) * self.num_components, self.num_tokens, -1)
        latents = self.latents.repeat(bs, 1, 1)
        x = x.repeat_interleave(self.num_components, dim=0)     # each data in the batch decompose to multiple components
        timestep_embedding = timestep_embedding.repeat_interleave(self.num_components, dim=0)

        latents = latents + self.time_aware_linear(
            torch.nn.functional.silu(timestep_embedding)
        )
        if self.input_dim is not None:
            x = self.proj_in(x)
        for p_block in self.perceiver_blocks:
            latents = p_block(x, latents, timestep_embedding=timestep_embedding)
        if self.output_dim is not None:
            latents = self.proj_out(latents)

        decomposed_latents = latents.reshape(bs, self.num_components, self.num_tokens, -1).chunk(self.num_components, dim=1)

        return decomposed_latents


class PerceiverResampler(nn.Module):
    '''
    use flan-t5-xl as ELLA
    '''
    def __init__(
        self,
        width,
        layers,
        heads,
        num_tokens,
        num_components,
        output_dim=None,
        input_dim=None,
        time_embedding_dim: Optional[int] = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.num_components = num_components
        self.latents = nn.Parameter(width**-0.5 * torch.randn(num_components, num_tokens, width))
        # self.time_aware_linear = nn.Linear(
        #     time_embedding_dim or width, width, bias=True
        # )

        if self.input_dim is not None:
            self.proj_in = nn.Linear(input_dim, width)

        self.perceiver_blocks = nn.Sequential(
            *[
                PerceiverAttentionBlock(
                    width, heads, time_embedding_dim=time_embedding_dim
                )
                for _ in range(layers)
            ]
        )

        if self.output_dim is not None:
            self.proj_out = nn.Sequential(
                nn.Linear(width, output_dim), nn.LayerNorm(output_dim)
            )

    def forward(self, x: torch.Tensor, timestep_embedding):
        # # batchify: [B * num_components, num_tokens, d]
        bs = x.shape[0]
        # learnable_latents = self.latents.unsqueeze(dim=0).repeat(len(x), 1, 1, 1)
        # latents = learnable_latents.reshape(len(x) * self.num_components, self.num_tokens, -1)
        latents = self.latents.repeat(bs, 1, 1)
        x = x.repeat_interleave(self.num_components, dim=0)     # each data in the batch decompose to multiple components
        timestep_embedding = timestep_embedding.repeat_interleave(self.num_components, dim=0)

        attn_weight = 0.0
        if self.input_dim is not None:
            x = self.proj_in(x)
        for p_block in self.perceiver_blocks:
            latents = p_block(x, latents, timestep_embedding=timestep_embedding)
            if isinstance(latents, tuple):
                attn_weight += latents[1]
                latents = latents[0]
        if self.output_dim is not None:
            latents = self.proj_out(latents)

        decomposed_latents = latents.reshape(bs, self.num_components, self.num_tokens, -1).chunk(self.num_components, dim=1)
        decomposed_latents = [latent.squeeze(1) for latent in decomposed_latents]
        if isinstance(attn_weight, torch.Tensor):
            return decomposed_latents, attn_weight
        return decomposed_latents


class PerceiverResampler_v2(nn.Module):
    '''
    ELLA with flan-t5-xl and clip
    '''
    def __init__(
        self,
        width,
        layers,
        heads,
        num_tokens,
        num_components,
        output_dim=None,
        input_dim=None,
        time_embedding_dim: Optional[int] = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.num_components = num_components
        self.latents = nn.Parameter(width**-0.5 * torch.randn(num_components, num_tokens, width))

        if self.input_dim is not None and input_dim != width:
            self.proj_in_t5 = nn.Linear(input_dim, width)

        self.perceiver_blocks = nn.Sequential(
            *[
                PerceiverAttentionBlock_v2(
                    width, heads, time_embedding_dim=time_embedding_dim
                )
                for _ in range(layers)
            ]
        )

        if self.output_dim is not None:
            self.proj_out = nn.Sequential(
                nn.Linear(width, output_dim), nn.LayerNorm(output_dim)
            )

    def forward(self, x_t5: torch.Tensor, x_clip: torch.Tensor, timestep_embedding: torch.Tensor):
        # # batchify: [B * num_components, num_tokens, d]
        bs = x_t5.shape[0]
        # learnable_latents = self.latents.unsqueeze(dim=0).repeat(len(x), 1, 1, 1)
        # latents = learnable_latents.reshape(len(x) * self.num_components, self.num_tokens, -1)
        latents = self.latents.repeat(bs, 1, 1)
        x_t5 = x_t5.repeat_interleave(self.num_components, dim=0)     # each data in the batch decompose to multiple components
        x_clip = x_clip.repeat_interleave(self.num_components, dim=0)
        timestep_embedding = timestep_embedding.repeat_interleave(self.num_components, dim=0)

        # latents = latents + self.time_aware_linear(
        #     torch.nn.functional.silu(timestep_embedding)
        # )
        if self.input_dim is not None:
            x_t5 = self.proj_in_t5(x_t5)
        for p_block in self.perceiver_blocks:
            latents = p_block(x_t5, x_clip, latents, timestep_embedding=timestep_embedding)
        if self.output_dim is not None:
            latents = self.proj_out(latents)

        decomposed_latents = latents.reshape(bs, self.num_components, self.num_tokens, -1).chunk(self.num_components, dim=1)
        decomposed_latents = [latent.squeeze(1) for latent in decomposed_latents]
        return decomposed_latents


class Resampler(nn.Module):
    def __init__(
        self,
        width=None,
        layers=None,
        heads=None,
        num_tokens=None,
        num_components=None,
        output_dim=None,
        input_dim=None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.num_components = num_components
        self.latents = nn.Parameter(width**-0.5 * torch.randn(num_components, num_tokens, width))

        if self.input_dim is not None and input_dim != width:
            self.proj_in = nn.Linear(input_dim, width)

        self.perceiver_blocks = nn.Sequential(
            *[
                PerceiverBlock(
                    width, heads,
                )
                for _ in range(layers)
            ]
        )

        if self.output_dim is not None:
            self.proj_out = nn.Sequential(
                nn.Linear(width, output_dim), nn.LayerNorm(output_dim)
            )

    def forward(self, x: torch.Tensor,):
        # # batchify: [B * num_components, num_tokens, d]
        bs = x.shape[0]
        # learnable_latents = self.latents.unsqueeze(dim=0).repeat(len(x), 1, 1, 1)
        # latents = learnable_latents.reshape(len(x) * self.num_components, self.num_tokens, -1)
        latents = self.latents.repeat(bs, 1, 1)                 # 
        x = x.repeat_interleave(self.num_components, dim=0)     # each data in the batch decompose to multiple components

        if self.input_dim is not None:
            assert False, 'now hard coded'
            x = self.proj_in(x)
        for p_block in self.perceiver_blocks:
            latents = p_block(x, latents,)
        if self.output_dim is not None:
            assert False, 'now hard coded'
            latents = self.proj_out(latents)

        decomposed_latents = latents.reshape(bs, self.num_components, self.num_tokens, -1).chunk(self.num_components, dim=1)
        decomposed_latents = [latent.squeeze(1) for latent in decomposed_latents]
        return decomposed_latents


class Resampler_v2(nn.Module):
    def __init__(
        self,
        width=None,
        layers=None,
        heads=None,
        num_tokens=None,
        num_components=None,
        output_dim=None,
        input_dim=None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.num_components = num_components
        self.latents = nn.Parameter(width**-0.5 * torch.randn(num_components, num_tokens, width))

        if self.input_dim is not None:
            self.proj_in = nn.Linear(input_dim, width)

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    width, heads,
                )
                for _ in range(layers)
            ]
        )

        self.perceiver_blocks = nn.Sequential(
            *[
                PerceiverBlock(
                    width, heads,
                )
                for _ in range(layers)
            ]
        )

        if self.output_dim is not None:
            self.proj_out = nn.Sequential(
                nn.Linear(width, output_dim), nn.LayerNorm(output_dim)
            )

    def forward(self, x: torch.Tensor, pad_to_max = False):

        if self.input_dim is not None:
            x = self.proj_in(x)
        for t_block in self.transformer_blocks:
            x = t_block(x)

        # Batchify: [B * num_components, num_tokens, d]
        bs = x.shape[0]
        # each data in the batch decompose to multiple components
        latents = self.latents.repeat(bs, 1, 1)
        x = x.repeat_interleave(self.num_components, dim=0)

        for p_block in self.perceiver_blocks:
            latents = p_block(x, latents,)

        if self.output_dim is not None:
            latents = self.proj_out(latents)

        decomposed_latents = latents.reshape(bs, self.num_components, self.num_tokens, -1).chunk(self.num_components, dim=1)
        if pad_to_max:
            padding_length = 2 - self.num_tokens
            decomposed_latents = [F.pad(latent.squeeze(1), (0, 0, 0, padding_length, 0, 0), mode='constant', value=0) for latent in decomposed_latents]
        else:
            decomposed_latents = [latent.squeeze(1) for latent in decomposed_latents]
        return decomposed_latents


class Resampler_v3(nn.Module):
    def __init__(
        self,
        width=None,
        layers=None,
        heads=None,
        num_tokens=None,
        num_components=None,
        output_dim=None,
        input_dim=None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.num_components = num_components
        self.latents = nn.Parameter(width**-0.5 * torch.randn(num_components, num_tokens, width))

        if self.input_dim is not None:
            self.proj_in = nn.Linear(input_dim, width)

        self.perceiver_blocks = nn.Sequential(
            *[
                PerceiverBlock(
                    width, heads,
                )
                for _ in range(layers)
            ]
        )

        if self.output_dim is not None:
            self.proj_out = nn.Linear(width, output_dim)

    def forward(self, x: torch.Tensor, pad_to_max=None):
        # # batchify: [B * num_components, num_tokens, d]
        bs = x.shape[0]
        # learnable_latents = self.latents.unsqueeze(dim=0).repeat(len(x), 1, 1, 1)
        # latents = learnable_latents.reshape(len(x) * self.num_components, self.num_tokens, -1)
        latents = self.latents.repeat(bs, 1, 1)                 # 
        x = x.repeat_interleave(self.num_components, dim=0)     # each data in the batch decompose to multiple components

        if self.input_dim is not None:
            x = self.proj_in(x)
        for p_block in self.perceiver_blocks:
            latents = p_block(x, latents,)
        if self.output_dim is not None:
            latents = self.proj_out(latents)

        decomposed_latents = latents.reshape(bs, self.num_components, self.num_tokens, -1).chunk(self.num_components, dim=1)
        decomposed_latents = [latent.squeeze(1) for latent in decomposed_latents]
        return decomposed_latents


class Resampler_v4(nn.Module):
    def __init__(
        self,
        width=None,
        layers=None,
        heads=None,
        num_tokens=None,
        num_components=None,
        output_dim=None,
        input_dim=None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.width = width
        self.num_tokens = num_tokens
        self.num_components = num_components
        self.latents = nn.Parameter(width**-0.5 * torch.randn(num_components, num_tokens, width))

        self.proj_in = nn.Linear(input_dim, num_components * width)

        self.perceiver_blocks = nn.Sequential(
            *[
                PerceiverBlock(
                    width, heads,
                )
                for _ in range(layers)
            ]
        )

        if self.output_dim is not None:
            self.proj_out = nn.Linear(width, output_dim)

    def forward(self, x: torch.Tensor, pad_to_max = False):
        # # batchify: [B * num_components, num_tokens, d]
        bs = x.shape[0]
        # learnable_latents = self.latents.unsqueeze(dim=0).repeat(len(x), 1, 1, 1)
        # latents = learnable_latents.reshape(len(x) * self.num_components, self.num_tokens, -1)
        latents = self.latents.repeat(bs, 1, 1)
        x = self.proj_in(x).reshape(bs, -1, self.num_components, self.width)
        x = x.transpose(1, 2).reshape(bs*self.num_components, -1, self.width)

        for p_block in self.perceiver_blocks:
            latents = p_block(x, latents,)
        if self.output_dim is not None:
            latents = self.proj_out(latents)

        decomposed_latents = latents.reshape(bs, self.num_components, self.num_tokens, -1).chunk(self.num_components, dim=1)
        if pad_to_max:
            padding_length = 2 - self.num_tokens
            decomposed_latents = [F.pad(latent.squeeze(1), (0, 0, 0, padding_length, 0, 0), mode='constant', value=0) for latent in decomposed_latents]
        else:
            decomposed_latents = [latent.squeeze(1) for latent in decomposed_latents]
        return decomposed_latents


class Resampler_v5(nn.Module):
    def __init__(
        self,
        width=None,
        layers=None,
        heads=None,
        num_tokens=None,
        num_components=None,
        output_dim=None,
        input_dim=None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.num_components = num_components
        self.latents = nn.Parameter(width**-0.5 * torch.randn(num_components, num_tokens, width))

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    input_dim, heads,
                )
                for _ in range(layers)
            ]
        )

        self.perceiver_blocks = nn.Sequential(
            *[
                PerceiverBlock_v2(
                    width, heads, input_dim, num_components
                )
                for _ in range(layers)
            ]
        )

        if self.output_dim is not None:
            self.proj_out = nn.Linear(width, output_dim)

    def forward(self, x: torch.Tensor, pad_to_max = False):
        # # batchify: [B * num_components, num_tokens, d]
        bs = x.shape[0]
        # learnable_latents = self.latents.unsqueeze(dim=0).repeat(len(x), 1, 1, 1)
        # latents = learnable_latents.reshape(len(x) * self.num_components, self.num_tokens, -1)
        latents = self.latents.repeat(bs, 1, 1)

        for t_block, p_block in zip(self.transformer_blocks, self.perceiver_blocks):
            x = t_block(x)
            latents = p_block(x, latents,)

        if self.output_dim is not None:
            latents = self.proj_out(latents)

        decomposed_latents = latents.reshape(bs, self.num_components, self.num_tokens, -1).chunk(self.num_components, dim=1)
        if pad_to_max:
            padding_length = 2 - self.num_tokens
            decomposed_latents = [F.pad(latent.squeeze(1), (0, 0, 0, padding_length, 0, 0), mode='constant', value=0) for latent in decomposed_latents]
        else:
            decomposed_latents = [latent.squeeze(1) for latent in decomposed_latents]
        return decomposed_latents


class Resampler_ct5(nn.Module):
    def __init__(
        self,
        width=None,
        layers=None,
        heads=None,
        num_tokens=None,
        num_components=None,
        output_dim=None,
        input_dim=None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.num_components = num_components
        self.latents = nn.Parameter(width**-0.5 * torch.randn(num_components, num_tokens, width))

        if self.input_dim is not None and input_dim != width:
            self.proj_in = nn.Linear(input_dim, width)

        self.perceiver_blocks = nn.Sequential(
            *[
                PerceiverBlock_v3(width, heads)
                for _ in range(layers)
            ]
        )

        if self.output_dim is not None:
            self.proj_out = nn.Sequential(
                nn.Linear(width, output_dim), nn.LayerNorm(output_dim)
            )

    def forward(self, x_t5: torch.Tensor, x_clip: torch.Tensor,):
        # # batchify: [B * num_components, num_tokens, d]
        bs = x_t5.shape[0]
        # learnable_latents = self.latents.unsqueeze(dim=0).repeat(len(x), 1, 1, 1)
        # latents = learnable_latents.reshape(len(x) * self.num_components, self.num_tokens, -1)
        latents = self.latents.repeat(bs, 1, 1)
        x_t5 = x_t5.repeat_interleave(self.num_components, dim=0)     # each data in the batch decompose to multiple components
        x_clip = x_clip.repeat_interleave(self.num_components, dim=0)

        if self.input_dim is not None:
            assert False, 'now hard coded'
            x_t5 = self.proj_in(x_t5)
        for p_block in self.perceiver_blocks:
            latents = p_block(x_t5, x_clip, latents,)
        if self.output_dim is not None:
            assert False, 'now hard coded'
            latents = self.proj_out(latents)

        decomposed_latents = latents.reshape(bs, self.num_components, self.num_tokens, -1).chunk(self.num_components, dim=1)
        decomposed_latents = [latent.squeeze(1) for latent in decomposed_latents]
        return decomposed_latents


class PromptResampler(nn.Module):
    def __init__(
        self,
        width=1024,
        heads=8,
        layers=6,
        num_tokens=64,
        num_components=4,
        input_dim=None,
        output_dim=None,
    ):
        super().__init__()

        self.connector = Resampler(
            width=width,
            layers=layers,
            heads=heads,
            num_tokens=num_tokens,
            num_components=num_components,
            input_dim=input_dim,
            output_dim=output_dim
        )

    def forward(self, text_encode_features):
        device = text_encode_features.device
        dtype = text_encode_features.dtype

        encoder_hidden_states = self.connector(text_encode_features,)
        return encoder_hidden_states


class PromptResampler_ct5(nn.Module):
    def __init__(
        self,
        width=1024,
        heads=8,
        layers=6,
        num_tokens=64,
        num_components=4,
        input_dim=None,
        output_dim=None,
    ):
        super().__init__()

        self.connector = Resampler_ct5(
            width=width,
            layers=layers,
            heads=heads,
            num_tokens=num_tokens,
            num_components=num_components,
            input_dim=input_dim,
            output_dim=output_dim
        )

    def forward(self, text_encode_feature_t5, text_encode_feature_clip):
        device = text_encode_feature_t5.device
        dtype = text_encode_feature_t5.dtype
        encoder_hidden_states = self.connector(text_encode_feature_t5, text_encode_feature_clip)
        return encoder_hidden_states


class PromptDecomposer(nn.Module):
    '''
    T5 only
    '''
    def __init__(
        self,
        act_fn: str = "silu",
        out_dim: Optional[int] = None,
        time_channel=320,
        time_embed_dim=768,
        width=1024,
        heads=8,
        layers=6,
        num_tokens=64,
        num_components=4,
        input_dim=2048,
        output_dim=None,
    ):
        super().__init__()

        self.position = Timesteps(
            time_channel, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedding = TimestepEmbedding(
            in_channels=time_channel,
            time_embed_dim=time_embed_dim,
            act_fn=act_fn,
            out_dim=out_dim,
        )

        self.connector = PerceiverResampler(
            width=width,
            layers=layers,
            heads=heads,
            num_tokens=num_tokens,
            num_components=num_components,
            input_dim=input_dim,
            output_dim=output_dim,
            time_embedding_dim=time_embed_dim,
        )
        self.attn_map = dict()

    def forward(self, text_encode_features, timesteps):
        device = text_encode_features.device
        dtype = text_encode_features.dtype

        ori_time_feature = self.position(timesteps.view(-1)).to(device, dtype=dtype)
        ori_time_feature = (
            ori_time_feature.unsqueeze(dim=1)
            if ori_time_feature.ndim == 2
            else ori_time_feature
        )
        ori_time_feature = ori_time_feature.expand(len(text_encode_features), -1, -1)
        time_embedding = self.time_embedding(ori_time_feature)

        encoder_hidden_states = self.connector(
            text_encode_features, timestep_embedding=time_embedding
        )
        if isinstance(encoder_hidden_states, tuple):
            self.attn_map[timesteps.item()] = encoder_hidden_states[1]
            encoder_hidden_states = encoder_hidden_states[0]
        return encoder_hidden_states


class PromptDecomposer_v2(nn.Module):
    '''
    CLIP + T5
    '''
    def __init__(
        self,
        act_fn: str = "silu",
        out_dim: Optional[int] = None,
        time_channel=320,
        time_embed_dim=1280,
        width=1024,
        heads=8,
        layers=6,
        num_tokens=64,
        num_components=4,
        input_dim=None,
        output_dim=None,
    ):
        super().__init__()

        self.position = Timesteps(
            time_channel, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedding = TimestepEmbedding(
            in_channels=time_channel,
            time_embed_dim=time_embed_dim,
            act_fn=act_fn,
            out_dim=out_dim,
        )

        self.connector = PerceiverResampler_v2(
            width=width,
            layers=layers,
            heads=heads,
            num_tokens=num_tokens,
            num_components=num_components,
            input_dim=input_dim,
            output_dim=output_dim,
            time_embedding_dim=time_embed_dim,
        )

    def forward(self, text_encode_feature_t5, text_encode_feature_clip, timesteps):
        device = text_encode_feature_t5.device
        dtype = text_encode_feature_t5.dtype

        ori_time_feature = self.position(timesteps.view(-1)).to(device, dtype=dtype)
        ori_time_feature = (
            ori_time_feature.unsqueeze(dim=1)
            if ori_time_feature.ndim == 2
            else ori_time_feature
        )
        ori_time_feature = ori_time_feature.expand(len(text_encode_feature_t5), -1, -1)
        time_embedding = self.time_embedding(ori_time_feature)

        encoder_hidden_states = self.connector(
            text_encode_feature_t5, text_encode_feature_clip, timestep_embedding=time_embedding
        )
        return encoder_hidden_states