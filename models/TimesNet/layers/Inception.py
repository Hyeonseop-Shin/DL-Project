"""
TimesNet Inception Layers
Implements 2D Inception blocks and FFT-based period detection for time series.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


def FFT_for_Period(x, k=2):
    """
    Use FFT to find top-k dominant periods in the time series.

    Args:
        x: Input tensor [B, T, N]
        k: Number of top periods to return

    Returns:
        period: List of top-k periods
        period_weight: Amplitude weights for each period
    """
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)  # batch mean, channel mean
    frequency_list[0] = 0  # Ignore DC component

    _, top_list = torch.topk(frequency_list, k)

    # Keep as Python list for indexing (avoid numpy for DDP compatibility)
    period = (x.shape[1] // top_list).tolist()

    return period, abs(xf).mean(-1)[:, top_list]


class Inception_Block(nn.Module):
    """
    2D Inception Block for processing 2D variations of time series.
    Standard Inception architecture with kernels [1, 3, 5, 7, 9, 11].
    """
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class TimesBlock(nn.Module):
    """
    Core module of TimesNet.
    1. FFT to find top-k periods.
    2. Reshape 1D time series to 2D based on period.
    3. Apply 2D Inception Block.
    4. Aggregate results based on amplitude weights.
    """
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ff, num_kernels):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k

        # 2D Convolutions (Inception Block)
        self.conv = nn.Sequential(
            Inception_Block(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block(d_ff, d_model, num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()

        # 1. FFT to find Top-k periods
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]

            # 2. Reshape 1D -> 2D
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x

            # Reshape: [B, T, N] -> [B, N, Length // Period, Period]
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()

            # 3. 2D Conv
            out = self.conv(out)

            # [B, N, Length // Period, Period] -> [B, Length, N]
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)

            res.append(out[:, :(self.seq_len + self.pred_len), :])

        # 4. Aggregation (Weighted Sum)
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)

        # Residual connection
        res = res + x
        return res
