import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.Attention import FullAttention, AttentionLayer
from .layers.Embed import DataEmbedding_inverted, PositionalEmbedding


class FlattenHead(nn.Module):
    """Project encoder representations to the prediction horizon."""

    def __init__(self, nf, target_window, head_dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch, n_vars, d_model, patch_num + 1]
        """
        x = self.flatten(x)
        x = self.linear(x)
        return self.dropout(x)


class EnEmbedding(nn.Module):
    """Patchify the temporal dimension and append a global token."""

    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, 1, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Tensor [batch, n_vars, seq_len]
        Returns:
            Tensor [batch * n_vars, patch_num + 1, d_model], n_vars
        """
        batch_size, n_vars, _ = x.shape
        glb = self.glb_token.expand(batch_size, n_vars, -1, -1)

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (batch_size * n_vars, x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross):
        for layer in self.layers:
            x = layer(x, cross)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross):
        B, _, D = cross.shape
        attn_out, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        cross_out, _ = self.cross_attention(x_glb, cross, cross)
        cross_out = torch.reshape(
            cross_out, (cross_out.shape[0] * cross_out.shape[1], cross_out.shape[2])
        ).unsqueeze(1)
        x_glb = self.norm2(x_glb_ori + self.dropout(cross_out))

        y = torch.cat([x[:, :-1, :], x_glb], dim=1)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class TimeXer(nn.Module):
    def __init__(self,
                 seq_len=512,
                 pred_len=96,
                 d_model=512,
                 d_ff=2048,
                 dropout=0.2,
                 n_heads=8,
                 activation="relu",
                 e_layers=2,
                 patch_len=16,
                 use_norm=True):
        super().__init__()

        assert seq_len % patch_len == 0, "seq_len must be divisible by patch_len"

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.patch_num = seq_len // patch_len
        self.use_norm = use_norm

        self.en_embedding = EnEmbedding(d_model, patch_len, dropout)
        self.ex_embedding = DataEmbedding_inverted(seq_len, d_model, dropout=dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(attention_dropout=dropout, output_attention=False),
                        d_model,
                        n_heads
                    ),
                    AttentionLayer(
                        FullAttention(attention_dropout=dropout, output_attention=False),
                        d_model,
                        n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        self.head_nf = d_model * (self.patch_num + 1)
        self.head = FlattenHead(self.head_nf, pred_len, head_dropout=dropout)

    def forward(self, x_enc, x_mark_enc=None):
        """
        Args:
            x_enc: [batch, seq_len, n_vars]
            x_mark_enc: [batch, seq_len, time_features] or None
        Returns:
            [batch, pred_len, n_vars]
        """
        if self.use_norm:
            means = x_enc.mean(dim=1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev
        else:
            means, stdev = None, None

        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out).permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out[:, -self.pred_len:, :]