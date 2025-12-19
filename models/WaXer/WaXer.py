"""
WaXer: TimeXer + WaveNet Feature Extractor
Combines TimeXer's patching mechanism with WaveNet's temporal feature extraction.
"""

import torch
import torch.nn as nn

from .layers.Embed import EnEmbedding
from .layers.Attention import FullAttention, AttentionLayer
from .layers.Encoder import FlattenHead, EncoderLayer, Encoder
from .layers.WaveNet import WaveNetFeatureExtractor


class WaXer(nn.Module):
    def __init__(self,
                 # TimeXer Params
                 seq_len=96,
                 pred_len=96,
                 d_model=512,
                 d_ff=2048,
                 dropout=0.1,
                 n_heads=8,
                 e_layers=2,
                 patch_len=16,
                 use_norm=True,
                 # WaveNet Params
                 wavenet_c_in=7,
                 wavenet_d_model=64,
                 wavenet_layers=3):
        super().__init__()

        assert seq_len % patch_len == 0, "seq_len must be divisible by patch_len"

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.patch_num = seq_len // patch_len
        self.use_norm = use_norm

        # 1. Internal Embedding (Endogenous)
        self.en_embedding = EnEmbedding(d_model, patch_len, dropout)

        # 2. Exogenous Extractor (WaveNet)
        self.wavenet = WaveNetFeatureExtractor(
            seq_len=seq_len,
            pred_len=pred_len,
            c_in=wavenet_c_in,
            d_model=wavenet_d_model,
            dropout=dropout,
            layers=wavenet_layers,
            kernel_size=3
        )

        # 3. Projection: WaveNet Feature -> TimeXer Dimension
        self.feature_projection = nn.Linear(wavenet_d_model, d_model)
        self.feature_dropout = nn.Dropout(dropout)

        # 4. TimeXer Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(attention_dropout=dropout, output_attention=False),
                        d_model, n_heads
                    ),
                    AttentionLayer(
                        FullAttention(attention_dropout=dropout, output_attention=False),
                        d_model, n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation="relu",
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        self.head_nf = d_model * (self.patch_num + 1)
        self.head = FlattenHead(self.head_nf, pred_len, head_dropout=dropout)

    def forward(self, x_enc, x_mark_enc=None):
        # x_enc: [Batch, Seq_Len, N_Vars]

        # --- A. WaveNet Feature Extraction ---
        wave_feat = self.wavenet(x_enc, return_feature=True)

        # Transpose to [B, T, d_wave] for Linear Projection
        wave_feat = wave_feat.transpose(1, 2)

        # Project to [B, T, d_model]
        ex_embed = self.feature_projection(wave_feat)
        ex_embed = self.feature_dropout(ex_embed)

        # --- B. TimeXer Pre-processing (Normalization) ---
        if self.use_norm:
            means = x_enc.mean(dim=1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev
        else:
            means, stdev = None, None

        batch_size, seq_len, n_vars = x_enc.shape

        # --- C. Embedding & Broadcasting ---
        en_embed, _ = self.en_embedding(x_enc.permute(0, 2, 1))

        # Broadcast WaveNet features for each variable
        ex_embed = ex_embed.unsqueeze(1).repeat(1, n_vars, 1, 1)
        ex_embed = ex_embed.reshape(batch_size * n_vars, seq_len, -1)

        # --- D. Encoder ---
        enc_out = self.encoder(en_embed, ex_embed)

        # --- E. Decoding & De-normalization ---
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out).permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out
