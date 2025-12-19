"""
iTransformer: Inverted Transformer for Time Series Forecasting (ICLR 2024 Spotlight)

이 module은 변수를 token으로 처리하는 역전된 Transformer를 구현합니다.
기존 Transformer가 time step을 token으로 처리하는 것과 달리,
iTransformer는 변수를 token으로 처리하여 변수 간 관계를 모델링합니다.

Architecture:
    Input [B, T, N] → Transpose [B, N, T] → Embedding → Encoder → Projection → Output

주요 특징:
    - Inverted attention: 변수를 token으로 처리하여 변수 간 dependency 학습
    - Instance normalization: Non-stationary data 처리
    - Simple & effective: Baseline으로 널리 사용

참고 논문:
    - iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
      (ICLR 2024 Spotlight)
    - 칭화대학교 연구팀 개발
    - GitHub: https://github.com/thuml/iTransformer

클래스:
    iTransformer: Inverted Transformer 시계열 예측 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.Embedding import DataEmbedding_inverted
from .layers.Attention import FullAttention, AttentionLayer
from .layers.Encoder import Encoder, EncoderLayer


class iTransformer(nn.Module):
    """
    iTransformer: Inverted Transformer 시계열 예측 모델

    변수를 token으로 처리하여 변수 간 관계를 attention으로 학습합니다.
    기존 time-wise attention 대신 variable-wise attention을 사용합니다.

    Args:
        seq_len (int): Input sequence 길이 (lookback window). Default: 90
        pred_len (int): Prediction 길이. Default: 30
        d_model (int): Embedding 차원. Default: 512
        d_ff (int): FFN hidden 차원. Default: 2048
        dropout (float): Dropout rate. Default: 0.1
        scale_factor (int): Attention scale factor. Default: 1
        n_heads (int): Attention head 수. Default: 8
        activation (str): Activation function. Default: "relu"
        e_layers (int): Encoder layer 수. Default: 1

    Attributes:
        enc_embedding: Inverted embedding (변수를 token으로 변환)
        encoder: Transformer encoder stack
        projection: Output projection layer

    Shape:
        - Input: [B, seq_len, n_vars]
        - Output: [B, pred_len, n_vars]

    Example:
        >>> model = iTransformer(seq_len=512, pred_len=16, d_model=32)
        >>> x = torch.randn(32, 512, 30)
        >>> output = model(x)  # [32, 16, 30]
    """

    def __init__(self,
                 seq_len=90,
                 pred_len=30,
                 d_model=512,
                 d_ff=4*512,
                 dropout=0.1,
                 scale_factor=1,
                 n_heads=8,
                 activation='relu',
                 e_layers=1):
        super().__init__()

        self.task_name = 'long_term_forecast'
        self.seq_len = seq_len      # Lookback length
        self.pred_len = pred_len    # Prediction length

        # ============================================
        # Inverted Embedding
        # - 일반 Transformer: [B, T, N] → token = time step
        # - iTransformer: [B, T, N] → [B, N, T] → token = 변수
        # - 각 변수의 전체 history를 하나의 token으로 embedding
        # ============================================
        self.enc_embedding = DataEmbedding_inverted(
            c_in=seq_len,
            d_model=d_model,
            dropout=dropout
        )

        # ============================================
        # Encoder
        # - Variable-wise self-attention
        # - 변수 간 dependency 학습
        # ============================================
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

        # ============================================
        # Output Projection
        # - d_model → pred_len으로 직접 projection
        # - 각 변수별로 prediction 생성
        # ============================================
        self.projection = nn.Linear(d_model, pred_len, bias=True)

    def forward(self, x_enc, x_mark_enc=None):
        """
        Forward pass

        Args:
            x_enc (Tensor): Input 시계열 [B, seq_len, n_vars]
            x_mark_enc (Tensor, optional): Time feature [B, seq_len, time_features]

        Returns:
            Tensor: Prediction [B, pred_len, n_vars]
        """
        # ============================================
        # 1단계: Instance Normalization
        # - Non-stationary Transformer 방식
        # - 각 변수별로 mean/std 계산하여 normalize
        # - Gradient 차단으로 학습 안정성 확보
        # ============================================
        means = x_enc.mean(1, keepdim=True).detach()  # [B, 1, N]
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape  # N = number of variables

        # ============================================
        # 2단계: Inverted Embedding
        # - Input: [B, T, N] → Embedding: [B, N, d_model]
        # - 각 변수의 전체 history를 하나의 embedding으로 압축
        # - Time feature가 있으면 concat
        # ============================================
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, N (+time_feat), d_model]

        # ============================================
        # 3단계: Encoder (Variable-wise Attention)
        # - Query = Key = Value = 변수 embedding
        # - 변수 간 상관관계를 attention으로 학습
        # ============================================
        enc_out, attns = self.encoder(enc_out)  # [B, N (+time_feat), d_model]

        # ============================================
        # 4단계: Output Projection
        # - d_model → pred_len으로 projection
        # - Transpose하여 [B, pred_len, N] 형태로 변환
        # ============================================
        dec_out = self.projection(enc_out).transpose(-1, 1)  # [B, pred_len, N (+time_feat)]
        dec_out = dec_out[:, :, :N]  # Time feature 제거, 원래 변수만 추출

        # ============================================
        # 5단계: De-normalization
        # - 원래 scale로 복원
        # - stdev * output + mean
        # ============================================
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out[:, -self.pred_len:, :]
