
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.Embedding import DataEmbedding_inverted
from .layers.Attention import FullAttention, AttentionLayer
from .layers.Encoder import Encoder, EncoderLayer

class iTransformer(nn.Module):
    def __init__(self,
        seq_len=90,
        pred_len=30,
        d_model=512,
        d_ff=4*512,
        dropout=0.1,
        scale_factor=1,
        n_heads=8,
        activation='relu',
        e_layers=1,
        ):
        super().__init__()

        self.task_name = 'long_term_forecast'
        self.seq_len = seq_len  # lookback length
        self.pred_len = pred_len    # prediction length

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(c_in=seq_len,
                                                    d_model=d_model,
                                                    dropout=dropout)
        
        # Encoder
        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    attention=AttentionLayer(
                        attention=FullAttention(
                            scale=scale_factor,
                            attention_dropout=dropout,
                            output_attention=False
                        ),
                        d_model=d_model,
                        n_heads=n_heads
                    ),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        # Decoder
        self.projection = nn.Linear(d_model, pred_len, bias=True)


    def forward(self, x_enc, x_mark_enc=None):
        # Normalization from Non-Stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()    # detach() => with no gradient
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape   # [batch, time, variable]

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)             # [batch, variable + time_feature, time]
        enc_out, attns = self.encoder(enc_out)   # [batch, query_len, d_model]
        dec_out = self.projection(enc_out).transpose(-1, 1)         # [batch, pred_len, query_len]
        dec_out = dec_out[:, :, :N]                                 # [batch, pred_len, variable]

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out[:, -self.pred_len:, :]