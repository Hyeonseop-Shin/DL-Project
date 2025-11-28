
### WaTiFormer (Long-Term Forecasting for Periodic Time Series)
### by Manyoung Han (2025/11/27)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math

# ========================================== 
# NOTE : Input Data Sturcture is [Batch, Time, Channel]
# ==========================================

# ==========================================
# 1. Base Modules (WaveNet, TimesNet, PosEnc)
# ==========================================

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation
        )

    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x

class WaveNetBlock(nn.Module):
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
        x_res = (x + res_out)
        s_out = self.skip_conv(x_gated)
        return x_res, s_out

class Inception_Block(nn.Module):
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
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ff, num_kernels):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        self.conv = nn.Sequential(
            Inception_Block(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block(d_ff, d_model, num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = self.FFT_for_Period(x, self.k)
        
        res = []
        for i in range(self.k):
            period = period_list[i]
            total_len = self.seq_len + self.pred_len
            
            if total_len % period != 0:
                length = ((total_len // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - total_len), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = total_len
                out = x
            
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :total_len, :])

        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        
        res = res + x 
        return res

    def FFT_for_Period(self, x, k=2):
        xf = torch.fft.rfft(x, dim=1)
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, k)
        top_list = top_list.detach().cpu().numpy()
        period = x.shape[1] // top_list
        return period, abs(xf).mean(-1)[:, top_list]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ==========================================
# 2. Unified Block (WaTiFormer Block)
# ==========================================

class WaTiBlock(nn.Module):
    def __init__(self, seq_len, d_model, d_ff, top_k, num_kernels, n_heads, dropout, dilation=1):
        super().__init__()
        
        # 1. Times Block
        self.times_block = TimesBlock(seq_len, 0, top_k, d_model, d_ff, num_kernels)
        self.norm_times = nn.LayerNorm(d_model)

        # 2. Wave Block
        self.wave_block = WaveNetBlock(d_model, d_model, kernel_size=3, dilation=dilation, dropout=dropout)
        self.norm_wave = nn.LayerNorm(d_model)

        # 3. Transformer Block
        self.transformer_block = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True
        )

    def forward(self, x):

        x = self.times_block(x) 
        x = self.norm_times(x)

        x_perm = x.transpose(1, 2) # [B, T, D] -> [B, D, T]
        x_wave, _ = self.wave_block(x_perm) 
        x = x_wave.transpose(1, 2) # [B, D, T] -> [B, T, D]
        x = self.norm_wave(x)

        x = self.transformer_block(x)

        return x

# ==========================================
# 3. Main Model (WaTiFormer_Unified)
# ==========================================

class WaTiFormer_Unified(nn.Module):
    def __init__(self,
        seq_len=90, pred_len=30, c_in=7, d_model=64, d_ff=64,
        n_layers=3, top_k=5, num_kernels=6, n_heads=4, dropout=0.1
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 1. Input Embedding
        self.input_projection = nn.Linear(c_in, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)

        # 2. Unified Blocks
        self.blocks = nn.ModuleList([
            WaTiBlock(
                seq_len=seq_len, d_model=d_model, d_ff=d_ff, 
                top_k=top_k, num_kernels=num_kernels, n_heads=n_heads, dropout=dropout, 
                dilation=2**i
            )
            for i in range(n_layers)
        ])

        # 3. Output Head
        self.projection = nn.Linear(seq_len * d_model, pred_len * c_in)

    def forward(self, x_enc, x_mark_enc=None):

        # --- Normalization (RevIN) ---
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        B, T, N = x_enc.shape

        # 1. Embedding + Positional Encoding
        x = self.input_projection(x_enc)
        x = self.pos_encoder(x)

        # 2. Pass through N Unified Blocks
        for block in self.blocks:
            x = block(x)

        # 3. Prediction Head

        x_flat = x.reshape(B, -1)
        dec_out = self.projection(x_flat)
        dec_out = dec_out.reshape(B, self.pred_len, -1)

        # --- De-Normalization ---
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out