"""
TimeXer: Time Series Transformer with Exogenous Variables (NeurIPS 2024)

이 module은 exogenous variable을 처리할 수 있는 patch 기반 Transformer를
시계열 예측을 위해 구현합니다.

Architecture:
    Endogenous: Input → Patch Embedding (+ Global Token) → Self-Attention
                                                              ↓
    Exogenous:  Input → Inverted Embedding ─────────────→ Cross-Attention → FFN → Output

주요 특징:
    - Patch embedding: 시계열을 고정 길이 patch로 분할하여 처리
    - Global token: 전체 sequence 정보를 요약하는 learnable token
    - Dual attention: Self-attention (patch 간) + Cross-attention (exogenous)
    - Instance normalization: Distribution shift 문제 해결

참고 논문:
    - TimeXer: Empowering Transformers for Time Series Forecasting
      with Exogenous Variables (NeurIPS 2024)
    - 칭화대학교 연구팀 개발

클래스:
    FlattenHead: Encoder output을 prediction으로 projection
    EnEmbedding: Patch embedding + Global token
    Encoder: Encoder layer stack
    EncoderLayer: Self-attention + Cross-attention + FFN
    TimeXer: Main model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.Attention import FullAttention, AttentionLayer
from .layers.Embed import DataEmbedding_inverted, PositionalEmbedding


class FlattenHead(nn.Module):
    """
    Flatten Head: Encoder output을 prediction으로 projection

    Encoder의 multi-dimensional output을 flatten한 후
    linear layer로 prediction length에 맞게 projection합니다.

    Args:
        nf (int): Flattened feature 차원 (d_model * (patch_num + 1))
        target_window (int): Prediction length
        head_dropout (float): Dropout rate. Default: 0.0

    Shape:
        - Input: [B, n_vars, d_model, patch_num + 1]
        - Output: [B, n_vars, target_window]
    """

    def __init__(self, nf, target_window, head_dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """
        Forward pass

        Args:
            x (Tensor): Encoder output [B, n_vars, d_model, patch_num + 1]

        Returns:
            Tensor: Prediction [B, n_vars, target_window]
        """
        # Flatten: [B, n_vars, d_model, patch_num+1] → [B, n_vars, d_model*(patch_num+1)]
        x = self.flatten(x)
        # Linear projection to prediction length
        x = self.linear(x)
        return self.dropout(x)


class EnEmbedding(nn.Module):
    """
    Endogenous Embedding: Patch Embedding + Global Token

    시계열을 patch 단위로 분할하고 embedding한 후,
    전체 sequence를 요약하는 global token을 추가합니다.

    Args:
        d_model (int): Embedding 차원
        patch_len (int): Patch 길이
        dropout (float): Dropout rate

    Attributes:
        value_embedding: Patch를 d_model 차원으로 projection
        glb_token: Learnable global token [1, 1, 1, d_model]
        position_embedding: Sinusoidal positional encoding

    Shape:
        - Input: [B, n_vars, seq_len]
        - Output: [B * n_vars, patch_num + 1, d_model], n_vars
    """

    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.patch_len = patch_len
        # Patch를 d_model 차원으로 projection
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        # Learnable global token (전체 sequence 정보 요약)
        self.glb_token = nn.Parameter(torch.randn(1, 1, 1, d_model))
        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass

        Args:
            x (Tensor): Input [B, n_vars, seq_len]

        Returns:
            Tuple[Tensor, int]:
                - Embedded patches with global token [B * n_vars, patch_num + 1, d_model]
                - Number of variables (n_vars)
        """
        batch_size, n_vars, _ = x.shape

        # Global token을 batch와 변수 수에 맞게 확장
        glb = self.glb_token.expand(batch_size, n_vars, -1, -1)

        # ============================================
        # Patchify: seq_len → patch_num개의 patch
        # - unfold로 sliding window 방식으로 patch 추출
        # - step = patch_len이므로 non-overlapping patch
        # ============================================
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        # Shape: [B, n_vars, patch_num, patch_len]

        # Reshape for value embedding
        x = torch.reshape(x, (batch_size * n_vars, x.shape[2], x.shape[3]))
        # Shape: [B*n_vars, patch_num, patch_len]

        # ============================================
        # Value + Position Embedding
        # - 각 patch를 d_model 차원으로 projection
        # - Positional encoding 추가
        # ============================================
        x = self.value_embedding(x) + self.position_embedding(x)
        # Shape: [B*n_vars, patch_num, d_model]

        # Reshape back to include n_vars dimension
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        # Shape: [B, n_vars, patch_num, d_model]

        # ============================================
        # Global Token 추가
        # - 마지막 position에 global token concat
        # - Global token은 전체 sequence 정보를 집약
        # ============================================
        x = torch.cat([x, glb], dim=2)
        # Shape: [B, n_vars, patch_num+1, d_model]

        # Flatten batch and n_vars for encoder processing
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Shape: [B*n_vars, patch_num+1, d_model]

        return self.dropout(x), n_vars


class Encoder(nn.Module):
    """
    TimeXer Encoder: EncoderLayer Stack

    여러 encoder layer를 쌓아 시계열의 temporal dependency를 학습합니다.

    Args:
        layers (list): EncoderLayer instance들의 list
        norm_layer (nn.Module, optional): Final normalization layer
        projection (nn.Module, optional): Output projection layer

    Shape:
        - Input: (x, cross) where x: [B, L, D], cross: [B, S, D]
        - Output: [B, L, D]
    """

    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross):
        """
        Forward pass

        Args:
            x (Tensor): Endogenous embedding [B, L, D]
            cross (Tensor): Exogenous embedding [B, S, D]

        Returns:
            Tensor: Encoder output [B, L, D]
        """
        # 각 encoder layer 순차적으로 처리
        for layer in self.layers:
            x = layer(x, cross)

        # Final normalization
        if self.norm is not None:
            x = self.norm(x)

        # Optional projection
        if self.projection is not None:
            x = self.projection(x)

        return x


class EncoderLayer(nn.Module):
    """
    TimeXer Encoder Layer: Self-Attention + Cross-Attention + FFN

    각 layer는 다음을 순차적으로 수행합니다:
    1. Self-attention: Patch 간 dependency 학습
    2. Cross-attention: Global token과 exogenous feature 상호작용
    3. Feed-forward network: Non-linear transformation

    Args:
        self_attention (AttentionLayer): Self-attention module
        cross_attention (AttentionLayer): Cross-attention module
        d_model (int): Model 차원
        d_ff (int, optional): FFN hidden 차원. Default: 4 * d_model
        dropout (float): Dropout rate. Default: 0.1
        activation (str): Activation function ("relu" or "gelu"). Default: "relu"

    Shape:
        - Input: (x, cross) where x: [B, L, D], cross: [B, S, D]
        - Output: [B, L, D]
    """

    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.self_attention = self_attention
        self.cross_attention = cross_attention

        # FFN: Conv1d로 구현 (kernel_size=1은 Linear와 동일)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross):
        """
        Forward pass

        Args:
            x (Tensor): Endogenous embedding [B*n_vars, patch_num+1, d_model]
            cross (Tensor): Exogenous embedding [B, seq_len, d_model]

        Returns:
            Tensor: Layer output [B*n_vars, patch_num+1, d_model]
        """
        B, _, D = cross.shape

        # ============================================
        # 1단계: Self-Attention
        # - 모든 patch 간의 dependency 학습
        # - Residual connection + LayerNorm
        # ============================================
        attn_out, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # ============================================
        # 2단계: Cross-Attention (Global Token Only)
        # - Global token만 exogenous feature와 상호작용
        # - Global token: x[:, -1, :] (마지막 position)
        # ============================================
        x_glb_ori = x[:, -1, :].unsqueeze(1)  # [B*n_vars, 1, D]

        # Reshape for cross-attention: [B*n_vars, 1, D] → [B, n_vars, D]
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))

        # Cross-attention: global token ← exogenous features
        cross_out, _ = self.cross_attention(x_glb, cross, cross)

        # Reshape back: [B, n_vars, D] → [B*n_vars, 1, D]
        cross_out = torch.reshape(
            cross_out, (cross_out.shape[0] * cross_out.shape[1], cross_out.shape[2])
        ).unsqueeze(1)

        # Residual connection + LayerNorm for global token
        x_glb = self.norm2(x_glb_ori + self.dropout(cross_out))

        # ============================================
        # 3단계: Feed-Forward Network
        # - Updated global token을 다시 붙여서 FFN 통과
        # - Conv1d(kernel=1)는 position-wise linear와 동일
        # ============================================
        # Concat: [patch embeddings, updated global token]
        y = torch.cat([x[:, :-1, :], x_glb], dim=1)

        # FFN: Linear → Activation → Dropout → Linear → Dropout
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # Final residual connection + LayerNorm
        return self.norm3(x + y)


class TimeXer(nn.Module):
    """
    TimeXer: Time Series Transformer with Exogenous Variables

    Patch 기반 Transformer로 시계열을 예측하며,
    exogenous variable을 cross-attention으로 통합합니다.

    Args:
        seq_len (int): Input sequence 길이. Default: 512
        pred_len (int): Prediction 길이. Default: 96
        d_model (int): Embedding 차원. Default: 512
        d_ff (int): FFN hidden 차원. Default: 2048
        dropout (float): Dropout rate. Default: 0.2
        n_heads (int): Attention head 수. Default: 8
        activation (str): Activation function. Default: "relu"
        e_layers (int): Encoder layer 수. Default: 2
        patch_len (int): Patch 길이. Default: 16
        use_norm (bool): Instance normalization 사용 여부. Default: True

    Shape:
        - Input: [B, seq_len, n_vars]
        - Output: [B, pred_len, n_vars]

    Example:
        >>> model = TimeXer(seq_len=512, pred_len=16, d_model=32)
        >>> x = torch.randn(32, 512, 30)
        >>> output = model(x)  # [32, 16, 30]
    """

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

        # seq_len이 patch_len으로 나누어 떨어지는지 확인
        assert seq_len % patch_len == 0, "seq_len must be divisible by patch_len"

        # Configuration 저장
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.patch_num = seq_len // patch_len
        self.use_norm = use_norm

        # ============================================
        # Embedding Layers
        # - en_embedding: Endogenous (patch + global token)
        # - ex_embedding: Exogenous (inverted embedding)
        # ============================================
        self.en_embedding = EnEmbedding(d_model, patch_len, dropout)
        self.ex_embedding = DataEmbedding_inverted(seq_len, d_model, dropout=dropout)

        # ============================================
        # Encoder: Self-Attention + Cross-Attention layers
        # ============================================
        self.encoder = Encoder(
            [
                EncoderLayer(
                    # Self-attention: patch 간 dependency
                    AttentionLayer(
                        FullAttention(attention_dropout=dropout, output_attention=False),
                        d_model,
                        n_heads
                    ),
                    # Cross-attention: exogenous feature 통합
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

        # ============================================
        # Output Head
        # - head_nf = d_model * (patch_num + 1): global token 포함
        # ============================================
        self.head_nf = d_model * (self.patch_num + 1)
        self.head = FlattenHead(self.head_nf, pred_len, head_dropout=dropout)

    def forward(self, x_enc, x_mark_enc=None):
        """
        Forward pass

        Args:
            x_enc (Tensor): Input 시계열 [B, seq_len, n_vars]
            x_mark_enc (Tensor, optional): Time features [B, seq_len, time_features]

        Returns:
            Tensor: Prediction [B, pred_len, n_vars]
        """
        # ============================================
        # 1단계: Instance Normalization
        # - 각 변수별로 mean/std 계산하여 normalize
        # - Non-stationary Transformer 방식
        # ============================================
        if self.use_norm:
            means = x_enc.mean(dim=1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev
        else:
            means, stdev = None, None

        # ============================================
        # 2단계: Embedding
        # - Endogenous: Patch embedding + global token
        # - Exogenous: Inverted embedding (변수 → token)
        # ============================================
        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)

        # ============================================
        # 3단계: Encoder
        # - Self-attention + Cross-attention + FFN
        # ============================================
        enc_out = self.encoder(en_embed, ex_embed)

        # ============================================
        # 4단계: Output Projection
        # - [B*n_vars, patch_num+1, d_model] → [B, n_vars, d_model, patch_num+1]
        # ============================================
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Head로 prediction 생성
        dec_out = self.head(enc_out).permute(0, 2, 1)  # [B, pred_len, n_vars]

        # ============================================
        # 5단계: De-normalization
        # - 원래 scale로 복원
        # ============================================
        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out[:, -self.pred_len:, :]
