"""TimeXer Attention Layers"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt


class FullAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, scale=None, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(p=attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, _ = keys.shape
        _, _, _, D = values.shape
        scale = self.scale or 1 / sqrt(E)

        # Attention: softmax(QK^T / sqrt(d_k)) * V
        attn_scores = torch.einsum('blhe,bshe->bhls', queries, keys)
        A = self.dropout(torch.softmax(scale * attn_scores, dim=-1))
        V = torch.einsum('bhls,bshd->blhd', A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FlashAttention(nn.Module):
    """PyTorch 2.0+ Flash Attention"""

    def __init__(self, scale=None, attention_dropout=0.1, output_attention=False, causal=False):
        super().__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout_p = attention_dropout
        self.causal = causal

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape

        q = queries.permute(0, 2, 1, 3)
        k = keys.permute(0, 2, 1, 3)
        v = values.permute(0, 2, 1, 3)

        dropout_p = self.dropout_p if self.training else 0.0

        if self.scale is not None:
            q = q * self.scale

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=self.causal
        )

        out = out.permute(0, 2, 1, 3).contiguous()
        return out, None


class AttentionLayer(nn.Module):
    """Multi-Head Attention with Linear Projections"""

    def __init__(self, attention, d_model=512, n_heads=8, d_keys=None, d_values=None):
        super().__init__()
        self.inner_attention = attention
        self.n_heads = n_heads

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.query_embed = nn.Linear(d_model, n_heads * d_keys)
        self.key_embed = nn.Linear(d_model, n_heads * d_keys)
        self.value_embed = nn.Linear(d_model, n_heads * d_values)
        self.out_projection = nn.Linear(n_heads * d_values, d_model)

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_embed(queries).view(B, L, H, -1)
        keys = self.key_embed(keys).view(B, S, H, -1)
        values = self.value_embed(values).view(B, S, H, -1)

        output, attn = self.inner_attention(queries=queries, keys=keys, values=values)

        output = output.view(B, L, -1)
        output = self.out_projection(output)

        return output, attn
