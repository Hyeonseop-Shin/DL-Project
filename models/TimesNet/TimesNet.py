"""
TimesNet (Long-Term Forecasting for Periodic Time Series)
by Manyoung Han (2025/11/25)
"""

import torch
import torch.nn as nn

from .layers.Inception import TimesBlock


class TimesNet(nn.Module):
    def __init__(self,
        seq_len=90,
        pred_len=30,
        c_in=5,
        c_out=5,
        d_model=64,
        d_ff=64,
        top_k=5,
        num_kernels=6,
        e_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.c_out = c_out

        # Embedding Layer
        self.enc_embedding = nn.Linear(c_in, d_model)

        # TimesBlocks
        self.layer = e_layers
        self.model = nn.ModuleList([
            TimesBlock(seq_len, pred_len, top_k, d_model, d_ff, num_kernels)
            for _ in range(e_layers)
        ])

        # For forecasting: Map seq_len -> seq_len + pred_len in time axis
        self.predict_linear = nn.Linear(self.seq_len, self.seq_len + self.pred_len)

        # Final Projection
        self.projection = nn.Linear(d_model, c_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_enc, x_mark_enc=None):
        # x_enc shape: [B, Seq_Len, N]

        # 1. Normalization (RevIN style)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # 2. Embedding
        enc_out = self.enc_embedding(x_enc)

        # 3. Extend Time Axis for Forecasting
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        # 4. Process TimesBlocks
        for i in range(self.layer):
            enc_out = self.model[i](enc_out)

        # 5. Final Projection
        dec_out = self.projection(enc_out)

        # 6. Slice the Prediction Part
        dec_out = dec_out[:, -self.pred_len:, :]

        # 7. De-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


class TimesNetFeatureExtractor(nn.Module):
    """
    Modified TimesNet to act as a feature extractor for TimeXer.
    It does NOT extend the time axis (predict_linear) but extracts features from the history.
    """
    def __init__(self, seq_len, c_in, d_model, d_ff, top_k, num_kernels, e_layers, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = 0

        self.enc_embedding = nn.Linear(c_in, d_model)

        self.model = nn.ModuleList([
            TimesBlock(seq_len, 0, top_k, d_model, d_ff, num_kernels)
            for _ in range(e_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_enc):
        # x_enc: [B, Seq_Len, N_Vars]

        # 1. Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # 2. Embedding
        enc_out = self.enc_embedding(x_enc)

        # 3. TimesBlocks (No time extension)
        for i in range(len(self.model)):
            enc_out = self.model[i](enc_out)

        # Returns: [B, T, d_model] (Rich periodic features)
        return enc_out
