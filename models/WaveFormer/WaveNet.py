
### WaveNet (Long-Term Forecasting for Periodic Time Series)
### by Manyoung Han (2025/11/21)

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):

    # KeyPoint : Dilation -> Need to change padding

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding = self.padding,
            dilation = dilation
        )

    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class WaveNetBlock(nn.Module):

    #Residual Block : Dilated Conv + Gated Activation + Skip Connection

    def __init__(self, residual_channels, skip_channels, kernel_size, dilation, dropout):
        super().__init__()

        # 1. Dilated Causal Convolution (Filter & Gate) -> Can change Channel_out
        self.filter_conv = CausalConv1d(residual_channels, residual_channels, kernel_size, dilation)
        self.gate_conv = CausalConv1d(residual_channels, residual_channels, kernel_size, dilation)

        # 2. 1x1 Conv for Residual connection -> Can change Channel_in
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

        # Residual Output -> No change in Time axis / Only Channel axis
        res_out = self.res_conv(x_gated)
        x_res = (x + res_out) * 0.707 # Scale for stability

        # Skip Output
        s_out = self.skip_conv(x_gated)

        return x_res, s_out


class WaveNetForecaster(nn.Module):
    def __init__(self,
        seq_len=90,      ##### CHANGE #####
        pred_len=30,     ##### CHANGE #####
        c_in=7,          # Input Channels
        d_model=64,      # Residual Channels
        dropout=0.1,
        layers=4,        # Number of WaveNet Block
        kernel_size=3,   # Size of Kernel
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len

        self.input_projection = nn.Conv1d(c_in, d_model, kernel_size=1)

        self.blocks = nn.ModuleList()
        self.skip_channels = d_model  # Can Change!!!

        for i in range(layers):
            dilation = 2 ** i
            self.blocks.append(
                WaveNetBlock(
                    residual_channels = d_model,
                    skip_channels = self.skip_channels,
                    kernel_size = kernel_size,
                    dilation = dilation,
                    dropout = dropout
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
        x = self.input_projection(x) # [B, N, T] -> [B, d_model, T]

        skip_connections = 0
        for block in self.blocks:
            x, skip = block(x)
            skip_connections += skip

        # 4. Output Generation
        # Skip connection & Activation
        x = F.relu(skip_connections)
        x = F.relu(self.final_conv1(x)) # [B, d_model, T]

        x_last = x[:, :, -1] # [B, d_model] only using last time step information

        dec_out = self.final_projection(x_last) # [B, d_model] -> [B, pred_len * N]

        dec_out = dec_out.view(B, self.pred_len, N) # [B, pred_len, N]

        # 5. De-Normalization (RevIN Inverse) -> stdev, means shape: [B, 1, N]
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out