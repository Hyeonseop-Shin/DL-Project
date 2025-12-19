"""WaXer Embedding Layers"""

import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    """Sinusoidal Positional Encoding"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class EnEmbedding(nn.Module):
    """Patch Embedding with Global Token for TimeXer"""

    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, 1, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [B, N, T]
        Returns:
            embedded: [B*N, num_patches+1, d_model]
            n_vars: number of variables
        """
        batch_size, n_vars, _ = x.shape

        # Global token: [B, N, 1, d_model]
        glb = self.glb_token.expand(batch_size, n_vars, -1, -1)

        # Patch split: [B, N, num_patches, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)

        # Value embedding + positional encoding
        x = torch.reshape(x, (batch_size * n_vars, x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)

        # Add global token
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)

        # Reshape for encoder: [B*N, P+1, d_model]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        return self.dropout(x), n_vars
