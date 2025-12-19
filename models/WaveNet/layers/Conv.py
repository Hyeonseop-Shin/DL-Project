"""WaveNet Convolutional Layers"""

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """Causal 1D Convolution (no future information leakage)"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )

    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class WaveNetBlock(nn.Module):
    """Gated Activation + Skip Connection Block"""

    def __init__(self, residual_channels, skip_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.filter_conv = CausalConv1d(residual_channels, residual_channels, kernel_size, dilation)
        self.gate_conv = CausalConv1d(residual_channels, residual_channels, kernel_size, dilation)

        self.res_conv = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Gated activation: tanh(filter) * sigmoid(gate)
        filter_out = torch.tanh(self.filter_conv(x))
        gate_out = torch.sigmoid(self.gate_conv(x))
        x_gated = self.dropout(filter_out * gate_out)

        # Residual output
        res_out = self.res_conv(x_gated)
        x_res = (x + res_out) * 0.707

        # Skip output
        s_out = self.skip_conv(x_gated)
        return x_res, s_out
