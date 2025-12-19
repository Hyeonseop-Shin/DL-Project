"""
WaveNet Feature Extractor for WaXer
Dilated causal convolutions for temporal feature extraction.
"""

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """Causal 1D Convolution with dilation support."""
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
    """Residual Block with Gated Activation."""
    def __init__(self, residual_channels, skip_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.filter_conv = CausalConv1d(residual_channels, residual_channels, kernel_size, dilation)
        self.gate_conv = CausalConv1d(residual_channels, residual_channels, kernel_size, dilation)
        self.res_conv = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        filter_out = torch.tanh(self.filter_conv(x))
        gate_out = torch.sigmoid(self.gate_conv(x))
        x_gated = filter_out * gate_out
        x_gated = self.dropout(x_gated)

        res_out = self.res_conv(x_gated)
        x_res = (x + res_out) * 0.707
        s_out = self.skip_conv(x_gated)
        return x_res, s_out


class WaveNetFeatureExtractor(nn.Module):
    """WaveNet for feature extraction in WaXer."""
    def __init__(self, seq_len, pred_len, c_in, d_model, dropout, layers, kernel_size):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_projection = nn.Conv1d(c_in, d_model, kernel_size=1)
        self.blocks = nn.ModuleList()
        self.skip_channels = d_model

        for i in range(layers):
            dilation = 2 ** i
            self.blocks.append(
                WaveNetBlock(
                    residual_channels=d_model,
                    skip_channels=self.skip_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )

    def forward(self, x_enc, return_feature=False):
        # 1. Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        B, T, N = x_enc.shape
        x = x_enc.transpose(1, 2)
        x = self.input_projection(x)

        skip_connections = 0
        for block in self.blocks:
            x, skip = block(x)
            skip_connections += skip

        if return_feature:
            return skip_connections

        return skip_connections
