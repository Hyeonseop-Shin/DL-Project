
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from math import sqrt

class FullAttention(nn.Module):
    def __init__(self, scale=None,
                 attention_dropout=0.1,
                 output_attention=False):
        super().__init__()

        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(p=attention_dropout)

    
    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape  # [batch, query_len, n_heads, d_k]
        _, S, _, _ = keys.shape     # [batch, key_len, n_heads, d_k]
        _, _, _, D = values.shape   # [batch, value_len=key_len, n_heads, d_v]
        scale = self.scale or 1 / sqrt(E)

        attn_scores = torch.einsum('blhe,bshe->bhls', queries, keys)    # [batch, n_heads, query_len, key_len]

        A = self.dropout(torch.softmax(scale * attn_scores, dim=-1))    # [batch, n_heads, query_len, key_len]
        V = torch.einsum('bhls,bshd->blhd', A, values)  # [batch, query_len, n_heads, d_v]

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None
        
        # contiguous() methods relocate the tensor memory and make it contiguous
        # to prevent mixing data with transformation, such as transpose, einsum, view, etc.


class FlashAttention(nn.Module):
    def __init__(self, scale=None,
                 attention_dropout=0.1,
                 output_attention=False,
                 causal: bool=False):
        super().__init__()

        self.scale = scale
        self.output_attention = output_attention
        self.dropout_p = attention_dropout
        self.causal = causal  # True for autoregressive

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape   # [batch, query_len, n_heads, d_k]
        _, S, _, _ = keys.shape      # [batch, key_len, n_heads, d_k]
        _, _, _, D = values.shape    # [batch, value_len=key_len, n_heads, d_v]

        q = queries.permute(0, 2, 1, 3)   # [B, H, L_q, d_k]
        k = keys.permute(0, 2, 1, 3)      # [B, H, L_k, d_k]
        v = values.permute(0, 2, 1, 3)    # [B, H, L_v, d_v]

        dropout_p = self.dropout_p if self.training else 0.0

        if self.scale is not None:
            q = q * self.scale

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=self.causal,  # apply causal mask if causal=True
        )  # [B, H, L_q, d_v]

        out = out.permute(0, 2, 1, 3).contiguous()  # [B, L, H, d_v]

        attn = None
        return out, attn


class AttentionLayer(nn.Module):
    def __init__(self, attention: FullAttention,
                 d_model=512,
                 n_heads=8,
                 d_keys=None,
                 d_values=None):
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
        B, L, _ = queries.shape # [batch, query_len, d_model]
        _, S, _ = keys.shape    # [batch, key_len, d_model]
        H = self.n_heads

        queries = self.query_embed(queries).view(B, L, H, -1)   # [batch, query_len, n_heads, d_k]
        keys = self.key_embed(keys).view(B, S, H, -1)   # [batch, key_len, n_heads, d_k]
        values = self.value_embed(values).view(B, S, H, -1)   # [batch, value_len, n_heads, d_v]

        output, attn = self.inner_attention(
            queries=queries,
            keys=keys,
            values=values
        )
        output = output.view(B, L, -1)  # [batch, query_len, n_heads * d_v]
        output = self.out_projection(output)    # [batch, query_len, d_model]

        return output, attn