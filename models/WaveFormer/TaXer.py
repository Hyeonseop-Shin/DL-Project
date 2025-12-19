import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math

# ==============================================================================
# 1. Basic Layers (Attention, Embeddings) - TimeXer Dependencies
# ==============================================================================

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag and attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshe->blhe", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        query = self.query_projection(queries).view(B, L, H, -1)
        key = self.key_projection(keys).view(B, S, H, -1)
        value = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(query, key, value, attn_mask, tau, delta)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn

# ==============================================================================
# 2. TimesNet Components (Feature Extractor)
# ==============================================================================

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
            # Padding to handle periodicity
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            
            # Reshape [B, T, N] -> [B, N, Length//P, P]
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])

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

        # Keep as Python list for indexing (avoid numpy for DDP compatibility)
        period = (x.shape[1] // top_list).tolist()

        return period, abs(xf).mean(-1)[:, top_list]

class TimesNetFeatureExtractor(nn.Module):
    """
    Modified TimesNet to act as a feature extractor for TimeXer.
    It does NOT extend the time axis (predict_linear) but extracts features from the history.
    """
    def __init__(self, seq_len, c_in, d_model, d_ff, top_k, num_kernels, e_layers, dropout):
        super().__init__()
        self.seq_len = seq_len
        # pred_len is set to 0 here because we want features of the input sequence
        self.pred_len = 0 
        
        self.enc_embedding = nn.Linear(c_in, d_model)
        
        self.model = nn.ModuleList([
            TimesBlock(seq_len, 0, top_k, d_model, d_ff, num_kernels)
            for _ in range(e_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_enc):
        # x_enc: [B, Seq_Len, N_Vars]
        # 1. Normalization (RevIN inside TimesNet extractor)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # 2. Embedding
        enc_out = self.enc_embedding(x_enc) # [B, T, d_model]
        
        # 3. TimesBlocks (No time extension)
        for i in range(len(self.model)):
            enc_out = self.model[i](enc_out)
        
        # Returns: [B, T, d_model] (Rich periodic features)
        return enc_out 

# ==============================================================================
# 3. TimeXer Components
# ==============================================================================

class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return self.dropout(x)

class EnEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, 1, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, n_vars, _ = x.shape
        glb = self.glb_token.expand(batch_size, n_vars, -1, -1)

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (batch_size * n_vars, x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars

class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
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
        attn_out, _ = self.self_attention(x, x, x, attn_mask=None)
        x = self.norm1(x + self.dropout(attn_out))

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        
        # Cross Attention with TimesNet Features
        cross_out, _ = self.cross_attention(x_glb, cross, cross, attn_mask=None)
        
        cross_out = torch.reshape(
            cross_out, (cross_out.shape[0] * cross_out.shape[1], cross_out.shape[2])
        ).unsqueeze(1)
        x_glb = self.norm2(x_glb_ori + self.dropout(cross_out))

        y = torch.cat([x[:, :-1, :], x_glb], dim=1)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)

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

# ==============================================================================
# 4. Final Integrated Model: TimeXer + TimesNet
# ==============================================================================

class TaXer(nn.Module):
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
                 # TimesNet Params
                 times_c_in=7,
                 times_d_model=64,
                 times_d_ff=64,
                 times_top_k=5,
                 times_num_kernels=6,
                 times_layers=2):
        super().__init__()
        
        assert seq_len % patch_len == 0, "seq_len must be divisible by patch_len"
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.patch_num = seq_len // patch_len
        self.use_norm = use_norm

        # 1. Main Embedding (Endogenous)
        self.en_embedding = EnEmbedding(d_model, patch_len, dropout)

        # 2. Exogenous Extractor (TimesNet)
        self.timesnet_extractor = TimesNetFeatureExtractor(
            seq_len=seq_len,
            c_in=times_c_in,
            d_model=times_d_model,
            d_ff=times_d_ff,
            top_k=times_top_k,
            num_kernels=times_num_kernels,
            e_layers=times_layers,
            dropout=dropout
        )

        # 3. Projection: TimesNet Dim -> TimeXer Dim
        self.feature_projection = nn.Linear(times_d_model, d_model)
        self.feature_dropout = nn.Dropout(dropout)

        # 4. Encoder
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
        """
        x_enc: [Batch, Seq_Len, N_Vars]
        """
        # --- A. TimesNet Feature Extraction ---
        # Extracts periodic features from the raw input history
        # Output: [B, Seq_Len, times_d_model]
        times_feat = self.timesnet_extractor(x_enc)
        
        # Projection to TimeXer dimension
        # [B, Seq_Len, d_model]
        ex_embed = self.feature_projection(times_feat)
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
        # 1. Endogenous Patch Embedding
        # [Batch * N_vars, Patch_Num + 1, d_model]
        en_embed, _ = self.en_embedding(x_enc.permute(0, 2, 1))

        # 2. Exogenous (TimesNet) Broadcasting
        # TimesNet features are global (mixed N_vars inside TimesNet's inception).
        # We need to provide this global context to each variable's patch series.
        # [B, T, D] -> [B, 1, T, D] -> [B, N_vars, T, D]
        ex_embed = ex_embed.unsqueeze(1).repeat(1, n_vars, 1, 1)
        
        # Flatten Batch and N_vars to match TimeXer's inverted structure
        # [Batch * N_vars, T, D]
        ex_embed = ex_embed.reshape(batch_size * n_vars, seq_len, -1)

        # --- D. Encoder Processing ---
        # Query: en_embed (Patches of individual variables)
        # Key/Value: ex_embed (Global Periodic Context from TimesNet)
        enc_out = self.encoder(en_embed, ex_embed)

        # --- E. Decoding ---
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2) # [B, N, D, P]

        dec_out = self.head(enc_out).permute(0, 2, 1) # [B, Pred, N]

        # De-normalization
        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out