"""
WaveNet (Long-Term Forecasting for Periodic Time Series)
by Manyoung Han (2025/11/21)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.Conv import CausalConv1d, WaveNetBlock


class WaveNetForecaster(nn.Module):
    def __init__(self,
        seq_len=90,
        pred_len=30,
        c_in=7,
        d_model=64,
        dropout=0.1,
        layers=4,
        kernel_size=3,
    ):
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

        self.final_conv1 = nn.Conv1d(self.skip_channels, self.skip_channels, kernel_size=1)
        self.final_projection = nn.Linear(d_model, pred_len * c_in)

    def forward(self, x_enc, x_mark_enc=None):
        # x_enc shape: [B, T, N]

        # 1. Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        B, T, N = x_enc.shape

        # 2. Transpose for Conv1d: [B, T, N] -> [B, N, T]
        x = x_enc.transpose(1, 2)

        # 3. Input Projection & WaveNet Blocks
        x = self.input_projection(x)

        skip_connections = 0
        for block in self.blocks:
            x, skip = block(x)
            skip_connections += skip

        # 4. Output Generation
        x = F.relu(skip_connections)
        x = F.relu(self.final_conv1(x))

        x_last = x[:, :, -1]

        dec_out = self.final_projection(x_last)
        dec_out = dec_out.view(B, self.pred_len, N)

        # 5. De-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out
