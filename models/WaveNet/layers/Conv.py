"""
WaveNet Convolutional Layers
Implements causal convolutions and gated activation blocks.
"""

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """
    Causal 1D Convolution with dilation support.
    KeyPoint: Dilation -> Need to change padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )

    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class WaveNetBlock(nn.Module):
    """
    Residual Block: Dilated Conv + Gated Activation + Skip Connection
    """
    def __init__(self, residual_channels, skip_channels, kernel_size, dilation, dropout):
        super().__init__()

        # 1. Dilated Causal Convolution (Filter & Gate)
        self.filter_conv = CausalConv1d(residual_channels, residual_channels, kernel_size, dilation)
        self.gate_conv = CausalConv1d(residual_channels, residual_channels, kernel_size, dilation)

        # 2. 1x1 Conv for Residual connection
        self.res_conv = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)

        # 3. 1x1 Conv for Skip connection
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [Batch, Channel, Time]

        # Gated Activation Units: tanh(W_f * x) * sigmoid(W_g * x)
        filter_out = torch.tanh(self.filter_conv(x))
        gate_out = torch.sigmoid(self.gate_conv(x))
        x_gated = filter_out * gate_out
        x_gated = self.dropout(x_gated)

        # Residual Output
        res_out = self.res_conv(x_gated)
        x_res = (x + res_out) * 0.707  # Scale for stability

        # Skip Output
        s_out = self.skip_conv(x_gated)

        return x_res, s_out
