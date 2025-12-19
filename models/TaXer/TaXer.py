"""
TaXer: TimeXer + TimesNet Hybrid 모델

이 module은 TimeXer Transformer와 TimesNet Feature Extractor를 결합한
hybrid 시계열 예측 모델을 구현합니다.

Architecture:
    Input → TimesNet Feature Extraction (FFT + 2D Conv) → Feature Projection
                                                              ↓
    Input → Patch Embedding → TimeXer Encoder (Self-Attn + Cross-Attn) → Head → Output
                                                              ↑
                                                 Exogenous Features (from TimesNet)

주요 아이디어:
    - TimesNet의 FFT 기반 period 탐지로 주기적 pattern 발견
    - 2D convolution으로 intra-period, inter-period 관계 모델링
    - 추출된 feature를 TimeXer의 exogenous variable로 사용

참고 논문:
    - TimeXer: Empowering Transformers for Time Series Forecasting (NeurIPS 2024)
    - TimesNet: Temporal 2D-Variation Modeling (ICLR 2023)

클래스:
    TaXer: TimeXer + TimesNet hybrid 예측 모델
"""

import torch
import torch.nn as nn

from .layers.Embed import EnEmbedding
from .layers.Attention import FullAttention, AttentionLayer
from .layers.Encoder import FlattenHead, EncoderLayer, Encoder
from ..TimesNet import TimesNetFeatureExtractor


class TaXer(nn.Module):
    """
    TaXer: TimeXer + TimesNet Hybrid 시계열 예측 모델

    TimesNet의 FFT 기반 2D temporal feature를 TimeXer의
    exogenous variable로 사용하여 주기적 시계열을 예측합니다.

    Args:
        seq_len (int): Input sequence 길이 (lookback window). Default: 96
        pred_len (int): Prediction 길이. Default: 96
        d_model (int): TimeXer embedding 차원. Default: 512
        d_ff (int): Feed-forward network 차원. Default: 2048
        dropout (float): Dropout rate. Default: 0.1
        n_heads (int): Multi-head attention의 head 수. Default: 8
        e_layers (int): Encoder layer 수. Default: 2
        patch_len (int): Patch 길이 (seq_len은 patch_len으로 나누어 떨어져야 함). Default: 16
        use_norm (bool): Instance normalization 사용 여부. Default: True
        times_c_in (int): TimesNet input channel 수 (변수 개수). Default: 7
        times_d_model (int): TimesNet hidden 차원. Default: 64
        times_d_ff (int): TimesNet FFN 차원 (Inception block). Default: 64
        times_top_k (int): FFT에서 선택할 top-K frequency 개수. Default: 5
        times_num_kernels (int): Inception block의 kernel 개수. Default: 6
        times_layers (int): TimesNet layer 수. Default: 2

    Attributes:
        en_embedding (EnEmbedding): Endogenous variable을 위한 patch embedding
        timesnet_extractor (TimesNetFeatureExtractor): TimesNet 기반 feature extractor
        feature_projection (nn.Linear): TimesNet → TimeXer 차원 projection
        encoder (Encoder): TimeXer encoder stack
        head (FlattenHead): Output projection head

    Shape:
        - Input: (B, T, D) - Batch, Sequence length, Number of variables
        - Output: (B, P, D) - Batch, Prediction length, Number of variables

    Example:
        >>> model = TaXer(seq_len=512, pred_len=16, d_model=32, times_c_in=30)
        >>> x = torch.randn(32, 512, 30)  # [Batch, Seq, Vars]
        >>> output = model(x)  # [32, 16, 30]
    """

    def __init__(self,
                 # ===== TimeXer Parameters =====
                 seq_len=96,
                 pred_len=96,
                 d_model=512,
                 d_ff=2048,
                 dropout=0.1,
                 n_heads=8,
                 e_layers=2,
                 patch_len=16,
                 use_norm=True,
                 # ===== TimesNet Parameters =====
                 times_c_in=7,
                 times_d_model=64,
                 times_d_ff=64,
                 times_top_k=5,
                 times_num_kernels=6,
                 times_layers=2):
        super().__init__()

        # seq_len이 patch_len으로 나누어 떨어지는지 확인
        assert seq_len % patch_len == 0, "seq_len must be divisible by patch_len"

        # ===== Configuration 저장 =====
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.patch_num = seq_len // patch_len  # Patch 개수
        self.use_norm = use_norm

        # ============================================
        # 1단계: Endogenous Embedding
        # - Input을 patch 단위로 분할하여 embedding
        # - Global token 추가 (전체 sequence 정보 요약)
        # ============================================
        self.en_embedding = EnEmbedding(d_model, patch_len, dropout)

        # ============================================
        # 2단계: Exogenous Feature Extractor (TimesNet)
        # - FFT로 주요 period (주기) 탐지
        # - 1D sequence를 2D tensor로 reshape
        # - Inception block으로 multi-scale feature 추출
        # ============================================
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

        # ============================================
        # 3단계: Feature Projection
        # - TimesNet output 차원을 TimeXer 차원으로 변환
        # - times_d_model → d_model
        # ============================================
        self.feature_projection = nn.Linear(times_d_model, d_model)
        self.feature_dropout = nn.Dropout(dropout)

        # ============================================
        # 4단계: TimeXer Encoder
        # - Self-attention: Patch 간 관계 모델링
        # - Cross-attention: TimesNet exogenous feature 통합
        # ============================================
        self.encoder = Encoder(
            [
                EncoderLayer(
                    # Self-attention layer (patch 간 상호작용)
                    AttentionLayer(
                        FullAttention(attention_dropout=dropout, output_attention=False),
                        d_model, n_heads
                    ),
                    # Cross-attention layer (exogenous feature 통합)
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

        # ============================================
        # 5단계: Output Head
        # - Encoder output을 prediction length로 projection
        # - head_nf = d_model * (patch_num + 1): global token 포함
        # ============================================
        self.head_nf = d_model * (self.patch_num + 1)
        self.head = FlattenHead(self.head_nf, pred_len, head_dropout=dropout)

    def forward(self, x_enc, x_mark_enc=None):
        """
        Forward pass

        Args:
            x_enc (Tensor): Input 시계열 [B, T, D]
                - B: Batch size
                - T: Sequence length (seq_len)
                - D: Number of variables
            x_mark_enc (Tensor, optional): Time feature (미사용)

        Returns:
            Tensor: Prediction [B, P, D]
                - P: Prediction length (pred_len)
        """
        # x_enc shape: [Batch, Seq_Len, N_Vars]

        # ============================================
        # A. TimesNet Feature Extraction
        # - FFT로 top-K period 탐지 (주요 주기성 발견)
        # - 각 period에 따라 1D → 2D reshape
        # - Inception block으로 intra/inter-period feature 추출
        # - Output shape: [B, T, times_d_model]
        # ============================================
        times_feat = self.timesnet_extractor(x_enc)  # [B, T, times_d_model]

        # TimeXer 차원으로 projection
        ex_embed = self.feature_projection(times_feat)  # [B, T, d_model]
        ex_embed = self.feature_dropout(ex_embed)

        # ============================================
        # B. Instance Normalization
        # - 각 변수별로 mean/std 계산하여 normalize
        # - Distribution shift 문제 완화 및 학습 안정성 향상
        # - 예측 후 de-normalization으로 원래 scale 복원
        # ============================================
        if self.use_norm:
            # Mean 계산 [B, 1, D] - gradient 차단
            means = x_enc.mean(dim=1, keepdim=True).detach()
            x_enc = x_enc - means

            # Standard deviation 계산 [B, 1, D]
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev
        else:
            means, stdev = None, None

        batch_size, seq_len, n_vars = x_enc.shape

        # ============================================
        # C. Patch Embedding & Broadcasting
        # - Endogenous: 각 변수별로 patch embedding + global token
        # - Exogenous: TimesNet feature를 모든 변수에 broadcast
        # ============================================
        # Endogenous embedding: [B*D, patch_num+1, d_model]
        en_embed, _ = self.en_embedding(x_enc.permute(0, 2, 1))

        # Exogenous embedding broadcasting
        # [B, T, d_model] → [B, D, T, d_model] → [B*D, T, d_model]
        ex_embed = ex_embed.unsqueeze(1).repeat(1, n_vars, 1, 1)
        ex_embed = ex_embed.reshape(batch_size * n_vars, seq_len, -1)

        # ============================================
        # D. Encoder Processing
        # - Self-attention: Patch 간 dependency 학습
        # - Cross-attention: Exogenous feature (TimesNet) 통합
        # ============================================
        enc_out = self.encoder(en_embed, ex_embed)  # [B*D, patch_num+1, d_model]

        # ============================================
        # E. Output Projection & De-normalization
        # - Encoder output을 prediction 형태로 reshape
        # - Normalize된 예측을 원래 scale로 복원
        # ============================================
        # Reshape: [B*D, P+1, d_model] → [B, D, d_model, P+1]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Flatten head로 prediction 생성
        dec_out = self.head(enc_out).permute(0, 2, 1)  # [B, P, D]

        # De-normalization: 원래 scale 복원
        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out
