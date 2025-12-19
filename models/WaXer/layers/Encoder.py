"""WaXer Encoder Layers"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlattenHead(nn.Module):
    """Flatten and project encoder output to prediction length"""

    def __init__(self, nf, target_window, head_dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)  # [B, N, d_model * (P+1)]
        x = self.linear(x)   # [B, N, pred_len]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    """Self-Attention + Cross-Attention + FFN Layer"""

    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.self_attention = self_attention
        self.cross_attention = cross_attention

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross):
        """
        Args:
            x: patch embeddings [B*N, P+1, d_model]
            cross: exogenous features [B*N, T, d_model]
        """
        B, _, D = cross.shape

        # Self-attention on all patches
        attn_out, _ = self.self_attention(x, x, x, attn_mask=None)
        x = self.norm1(x + self.dropout(attn_out))

        # Cross-attention: global token (Q) x exogenous (K, V)
        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))

        cross_out, _ = self.cross_attention(x_glb, cross, cross, attn_mask=None)

        cross_out = torch.reshape(
            cross_out, (cross_out.shape[0] * cross_out.shape[1], cross_out.shape[2])
        ).unsqueeze(1)
        x_glb = self.norm2(x_glb_ori + self.dropout(cross_out))

        # FFN
        y = torch.cat([x[:, :-1, :], x_glb], dim=1)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Encoder(nn.Module):
    """Stack of Encoder Layers"""

    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross):
        for layer in self.layers:
            x = layer(x, cross)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        return x
