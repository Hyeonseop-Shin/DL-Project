"""iTransformer Encoder Layers"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .Attention import AttentionLayer


class EncoderLayer(nn.Module):
    """Self-Attention + FFN Layer"""

    def __init__(self, attention, d_model=512, d_ff=None, dropout=0.1, activation='relu'):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        self.dropout = nn.Dropout(p=dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x):
        # Self-attention
        new_x, attn = self.attention(queries=x, keys=x, values=x)
        x = x + self.dropout(new_x)

        # LayerNorm + FFN
        y = x = self.norm1(x)
        y = self.activation(self.conv1(y.transpose(-1, 1)))
        y = self.dropout(y)
        y = self.conv2(y).transpose(-1, 1)
        y = self.dropout(y)

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    """Stack of Encoder Layers"""

    def __init__(self, attn_layers, norm_layer):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm_layer = norm_layer

    def forward(self, x):
        attentions = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x)
            attentions.append(attn)

        if self.norm_layer is not None:
            x = self.norm_layer(x)

        return x, attentions
