"""WaveNet Feature Extractor for WaXer"""

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

        res_out = self.res_conv(x_gated)
        x_res = (x + res_out) * 0.707

        s_out = self.skip_conv(x_gated)
        return x_res, s_out


class WaveNetFeatureExtractor(nn.Module):
    """Dilated Causal Convolution Stack for Temporal Feature Extraction"""

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
                WaveNetBlock(d_model, self.skip_channels, kernel_size, dilation, dropout)
            )

    def forward(self, x_enc, return_feature=False):
        """
        Args:
            x_enc: [B, T, N]
        Returns:
            skip_connections: [B, d_model, T]
        """
        # Instance normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Transpose + projection
        x = x_enc.transpose(1, 2)  # [B, N, T]
        x = self.input_projection(x)  # [B, d_model, T]

        # WaveNet blocks with skip connections
        skip_connections = 0
        for block in self.blocks:
            x, skip = block(x)
            skip_connections += skip

        return skip_connections
