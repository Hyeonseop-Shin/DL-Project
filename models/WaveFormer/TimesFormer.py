
### TimesFormer (Long-Term Forecasting for Periodic Time Series)
### by Manyoung Han (2025/11/27)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math

# ==========================================
# 1. Base Modules (From TimesNet)
# ==========================================

class Inception_Block(nn.Module):
    """
    Standard Inception Block for TimesNet.
    Captures variations at multiple scales using different kernel sizes.
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
    Modified TimesBlock for TimesFormer.
    It functions as a Feature Extractor for the 'history' sequence.
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
        period_list, period_weight = self.FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            
            # 2. Reshape 1D -> 2D
            # Note: We use (seq_len + pred_len) logic. 
            # In TimesFormer, pred_len is set to 0, so it works on history only.
            total_len = self.seq_len + self.pred_len
            
            if total_len % period != 0:
                length = ((total_len // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - total_len), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = total_len
                out = x
            
            # Reshape: [B, T, N] -> [B, N, Length // Period, Period]
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            
            # 3. 2D Conv
            out = self.conv(out)
            
            # [B, N, Length // Period, Period] -> [B, Length, N]
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            
            # Return to original length (remove padding)
            res.append(out[:, :total_len, :])

        # 4. Aggregation (Weighted Sum)
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1) 
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1) 
        
        # Residual connection
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


# ==========================================
# 2. Bridge Module (Positional Encoding)
# ==========================================

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
# 3. Main Model (TimesFormer)
# ==========================================

class TimesFormer(nn.Module):
    def __init__(self,
        seq_len=90,      # Look-back window
        pred_len=30,     # Prediction horizon
        c_in=7,          # Input channels
        d_model=64,      # Model dimension (used for both TimesBlock & Transformer)
        d_ff=64,         # FeedForward dimension
        top_k=5,         # TimesBlock top_k periods
        num_kernels=6,   # Inception kernels
        times_layers=2,  # Number of TimesBlocks
        trans_layers=2,  # Number of Transformer Layers
        n_heads=4,       # Transformer heads
        dropout=0.1,
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 1. Embedding
        self.enc_embedding = nn.Linear(c_in, d_model)
        
        # 2. TimesNet Encoder (Local/Periodic Context)
        # Key: We set pred_len=0 here because we only want to extract features from history.
        self.times_blocks = nn.ModuleList([
            TimesBlock(seq_len, 0, top_k, d_model, d_ff, num_kernels)
            for _ in range(times_layers)
        ])
        
        # 3. Transformer Encoder (Global Context)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=trans_layers)

        # 4. Final Projection Head
        self.final_projection = nn.Linear(d_model, pred_len * c_in)

    def forward(self, x_enc, x_mark_enc=None):
        # x_enc: [Batch, Time, Channels]

        # --- 1. Normalization (RevIN) ---
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        # --- 2. TimesNet Processing (Periodic Feature Extraction) ---
        # Embedding: [B, T, C] -> [B, T, d_model]
        x = self.enc_embedding(x_enc)
        
        # Apply TimesBlocks
        # Note: Input/Output shape is [B, T, d_model]
        for block in self.times_blocks:
            x = block(x)
        
        # --- 3. Transformer Processing (Global Context) ---
        # Add Positional Encoding
        x = self.pos_encoder(x)
        
        # Apply Transformer Encoder
        x = self.transformer_encoder(x) # Output: [B, T, d_model]
        
        # --- 4. Prediction Head ---
        # Use the last time step information
        x_last = x[:, -1, :] # [B, d_model]
        
        dec_out = self.final_projection(x_last) # [B, pred_len * C_in]
        dec_out = dec_out.view(x.shape[0], self.pred_len, -1) # [B, pred_len, C_in]
        
        # --- 5. De-Normalization ---
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out