
### WaveFormer (Long-Term Forecasting for Periodic Time Series)
### by Manyoung Han (2025/11/27)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==========================================
# 1. Base Modules (From WaveNet)
# ==========================================

class CausalConv1d(nn.Module):
    """
    1D Causal Convolution to ensure no future information leakage.
    Dilated convolution is used to expand the receptive field.
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
            x = x[:, :, :-self.padding]  # Remove padding from the right (future)
        return x


class WaveNetBlock(nn.Module):
    """
    Residual Block combining Dilated Causal Conv, Gated Activation, and Skip Connection.
    Captures local patterns at different scales.
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

        # Residual Output -> Element-wise addition
        res_out = self.res_conv(x_gated)
        x_res = (x + res_out) * 0.707  # Scale for stability

        # Skip Output
        s_out = self.skip_conv(x_gated)

        return x_res, s_out


# ==========================================
# 2. Bridge Module (Positional Encoding)
# ==========================================

class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of the tokens in the sequence.
    Essential for Transformers as they are permutation invariant.
    """
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
        # x: [Batch, Time, d_model]
        return x + self.pe[:, :x.size(1), :]


# ==========================================
# 3. Main Model (WaveFormer)
# ==========================================

class WaveFormer(nn.Module):
    """
    Hybrid Architecture: WaveNet (Local Feature) + Transformer (Global Dependency)
    """
    def __init__(self,
        seq_len=90,      # Look-back window size
        pred_len=30,     # Prediction horizon
        c_in=7,          # Number of input variables
        d_model=64,      # Hidden dimension size
        n_heads=4,       # Number of Attention heads
        d_ff=256,        # FeedForward dimension in Transformer
        dropout=0.1,
        wave_layers=4,   # Number of WaveNet Blocks
        trans_layers=2,  # Number of Transformer Layers
        kernel_size=3,   # Kernel size for WaveNet
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # --- Local Feature Extraction (WaveNet) ---
        self.input_projection = nn.Conv1d(c_in, d_model, kernel_size=1)

        self.wave_blocks = nn.ModuleList()
        self.skip_channels = d_model 

        for i in range(wave_layers):
            dilation = 2 ** i
            self.wave_blocks.append(
                WaveNetBlock(
                    residual_channels = d_model,
                    skip_channels = self.skip_channels,
                    kernel_size = kernel_size,
                    dilation = dilation,
                    dropout = dropout
                )
            )
        
        # Mixing WaveNet outputs
        self.wave_out_conv = nn.Conv1d(self.skip_channels, d_model, kernel_size=1)

        # --- Global Dependency Modeling (Transformer) ---
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_ff, 
            dropout=dropout,
            batch_first=True # Expected input: [Batch, Time, Channel]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=trans_layers)

        # --- Forecasting Head ---
        # Predicts all future steps at once based on the last hidden state
        self.final_projection = nn.Linear(d_model, pred_len * c_in)

    def forward(self, x_enc, x_mark_enc=None):
        # x_enc: [Batch, Time, Channels]

        # 1. Normalization (RevIN approach)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        B, T, N = x_enc.shape

        # 2. WaveNet Processing (Channel-First: [B, N, T])
        x = x_enc.transpose(1, 2)
        x = self.input_projection(x) # [B, d_model, T]

        skip_connections = 0
        for block in self.wave_blocks:
            x, skip = block(x)
            skip_connections += skip
            
        # Combine Skip Connections
        x = F.relu(skip_connections)
        x = F.relu(self.wave_out_conv(x)) # [B, d_model, T]
        
        # 3. Transformer Processing (Time-First: [B, T, d_model])
        x = x.permute(0, 2, 1) 
        
        # Add Positional Encoding
        x = self.pos_encoder(x)
        
        # Transformer Encoder
        x = self.transformer_encoder(x) # [B, T, d_model]
        
        # 4. Prediction Head
        # Use the last time step's context to predict future sequence
        x_last = x[:, -1, :] # [B, d_model]
        
        dec_out = self.final_projection(x_last) # [B, pred_len * N]
        dec_out = dec_out.view(B, self.pred_len, N)
        
        # 5. De-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out