"""
TimesNet: Temporal 2D-Variation Modeling for Time Series Forecasting (ICLR 2023)

이 module은 시계열을 2D representation으로 변환하여 주기적 pattern을
효과적으로 모델링하는 TimesNet을 구현합니다.

Architecture:
    Input → Embedding → [FFT Period Detection → 2D Reshape → Inception Conv → 1D Reshape] × L → Output

주요 아이디어:
    - FFT로 시계열의 주요 주기(period) 탐지
    - 1D 시계열을 2D tensor로 reshape (period를 한 축으로)
    - 2D Inception convolution으로 intra/inter-period pattern 포착
    - Multi-scale feature extraction

Processing Pipeline:
    1. FFT로 top-K frequency component 찾기 (주요 주기 탐지)
    2. 각 period에 따라 1D → 2D reshape
       예: seq_len=512, period=7 → 73 × 7 (대략)
    3. Inception block으로 2D feature 추출
    4. 1D로 다시 reshape하여 weighted sum

참고 논문:
    - TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis
      (ICLR 2023)
    - 칭화대학교 연구팀 개발
    - GitHub: https://github.com/thuml/TimesNet

클래스:
    TimesNet: 시계열 예측 모델
    TimesNetFeatureExtractor: Feature extractor (WaXer/TaXer용)
"""

import torch
import torch.nn as nn

from .layers.Inception import TimesBlock


class TimesNet(nn.Module):
    """
    TimesNet: 2D Temporal Modeling 시계열 예측 모델

    FFT로 주기를 탐지하고, 2D convolution으로 intra-period와
    inter-period variation을 함께 모델링합니다.

    Args:
        seq_len (int): Input sequence 길이. Default: 90
        pred_len (int): Prediction 길이. Default: 30
        c_in (int): Input channel 수 (변수 개수). Default: 5
        c_out (int): Output channel 수. Default: 5
        d_model (int): Embedding 차원. Default: 64
        d_ff (int): Inception block hidden 차원. Default: 64
        top_k (int): FFT에서 선택할 top-K period 수. Default: 5
        num_kernels (int): Inception block kernel 수. Default: 6
        e_layers (int): TimesBlock 수. Default: 2
        dropout (float): Dropout rate. Default: 0.1

    Attributes:
        enc_embedding: Input → d_model embedding
        model: TimesBlock list
        predict_linear: seq_len → seq_len + pred_len extension
        projection: d_model → c_out projection

    Shape:
        - Input: [B, seq_len, c_in]
        - Output: [B, pred_len, c_out]

    Example:
        >>> model = TimesNet(seq_len=512, pred_len=16, c_in=30, c_out=30)
        >>> x = torch.randn(32, 512, 30)
        >>> output = model(x)  # [32, 16, 30]
    """

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
                 dropout=0.1):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.c_out = c_out

        # ============================================
        # Embedding Layer
        # - c_in → d_model로 차원 확장
        # - Linear projection (Conv1d가 아닌 Linear 사용)
        # ============================================
        self.enc_embedding = nn.Linear(c_in, d_model)

        # ============================================
        # TimesBlocks
        # - 각 block에서 FFT → 2D reshape → Inception → 1D reshape
        # - e_layers개의 block을 순차적으로 적용
        # ============================================
        self.layer = e_layers
        self.model = nn.ModuleList([
            TimesBlock(seq_len, pred_len, top_k, d_model, d_ff, num_kernels)
            for _ in range(e_layers)
        ])

        # ============================================
        # Time Extension Layer
        # - Forecasting을 위해 시간 축 확장
        # - seq_len → seq_len + pred_len으로 Linear projection
        # ============================================
        self.predict_linear = nn.Linear(self.seq_len, self.seq_len + self.pred_len)

        # ============================================
        # Output Projection
        # - d_model → c_out (원래 변수 수)로 projection
        # ============================================
        self.projection = nn.Linear(d_model, c_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_enc, x_mark_enc=None):
        """
        Forward pass

        Args:
            x_enc (Tensor): Input 시계열 [B, Seq_Len, N]
            x_mark_enc (Tensor, optional): Time feature (미사용)

        Returns:
            Tensor: Prediction [B, pred_len, c_out]
        """
        # x_enc shape: [B, Seq_Len, N]

        # ============================================
        # 1단계: Instance Normalization (RevIN style)
        # - 각 변수별로 mean/std 계산
        # - Non-stationary time series 처리
        # ============================================
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # ============================================
        # 2단계: Embedding
        # - c_in → d_model로 차원 변환
        # ============================================
        enc_out = self.enc_embedding(x_enc)  # [B, T, d_model]

        # ============================================
        # 3단계: Time Axis Extension
        # - Forecasting을 위해 prediction 구간 추가
        # - Linear projection으로 T → T + P 확장
        # - [B, T, d_model] → permute → [B, d_model, T]
        # - Linear(T, T+P) → permute back → [B, T+P, d_model]
        # ============================================
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        # Shape: [B, T+P, d_model]

        # ============================================
        # 4단계: TimesBlocks Processing
        # - 각 block에서:
        #   1. FFT로 top-K period 탐지
        #   2. 각 period에 따라 1D → 2D reshape
        #   3. Inception block으로 2D feature 추출
        #   4. 1D로 reshape하고 weighted sum
        # ============================================
        for i in range(self.layer):
            enc_out = self.model[i](enc_out)  # [B, T+P, d_model]

        # ============================================
        # 5단계: Output Projection
        # - d_model → c_out으로 projection
        # ============================================
        dec_out = self.projection(enc_out)  # [B, T+P, c_out]

        # ============================================
        # 6단계: Prediction 구간 추출
        # - 마지막 pred_len 구간만 선택
        # ============================================
        dec_out = dec_out[:, -self.pred_len:, :]  # [B, P, c_out]

        # ============================================
        # 7단계: De-normalization
        # - 원래 scale로 복원
        # ============================================
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


class TimesNetFeatureExtractor(nn.Module):
    """
    TimesNet Feature Extractor: TaXer/WaXer용 Feature Extractor

    TimesNet의 구조를 사용하되, 시간 축 확장 없이
    현재 sequence에서 rich periodic feature를 추출합니다.

    TimeXer의 exogenous variable로 사용되며,
    주기적 pattern 정보를 제공합니다.

    Args:
        seq_len (int): Input sequence 길이
        c_in (int): Input channel 수 (변수 개수)
        d_model (int): Embedding 차원
        d_ff (int): Inception block hidden 차원
        top_k (int): FFT에서 선택할 top-K period 수
        num_kernels (int): Inception block kernel 수
        e_layers (int): TimesBlock 수
        dropout (float): Dropout rate

    Shape:
        - Input: [B, seq_len, c_in]
        - Output: [B, seq_len, d_model]

    Example:
        >>> extractor = TimesNetFeatureExtractor(
        ...     seq_len=512, c_in=30, d_model=64,
        ...     d_ff=64, top_k=3, num_kernels=4, e_layers=2, dropout=0.1
        ... )
        >>> x = torch.randn(32, 512, 30)
        >>> features = extractor(x)  # [32, 512, 64]
    """

    def __init__(self, seq_len, c_in, d_model, d_ff, top_k, num_kernels, e_layers, dropout):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = 0  # Feature extractor이므로 prediction 없음

        # ============================================
        # Embedding Layer
        # - c_in → d_model로 차원 변환
        # ============================================
        self.enc_embedding = nn.Linear(c_in, d_model)

        # ============================================
        # TimesBlocks
        # - pred_len=0으로 설정하여 시간 축 확장 없음
        # - 순수하게 feature extraction만 수행
        # ============================================
        self.model = nn.ModuleList([
            TimesBlock(seq_len, 0, top_k, d_model, d_ff, num_kernels)
            for _ in range(e_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_enc):
        """
        Forward pass

        Args:
            x_enc (Tensor): Input 시계열 [B, Seq_Len, N_Vars]

        Returns:
            Tensor: Extracted features [B, Seq_Len, d_model]
                - 각 time step에 대한 rich periodic feature
                - TimeXer의 exogenous input으로 사용
        """
        # x_enc: [B, Seq_Len, N_Vars]

        # ============================================
        # 1단계: Instance Normalization
        # - 각 변수별로 mean/std 계산하여 normalize
        # ============================================
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # ============================================
        # 2단계: Embedding
        # - c_in → d_model로 차원 변환
        # ============================================
        enc_out = self.enc_embedding(x_enc)  # [B, T, d_model]

        # ============================================
        # 3단계: TimesBlocks Processing
        # - FFT → 2D reshape → Inception → 1D reshape
        # - 시간 축 확장 없이 feature extraction만 수행
        # - Rich periodic features 추출
        # ============================================
        for i in range(len(self.model)):
            enc_out = self.model[i](enc_out)  # [B, T, d_model]

        # Returns: [B, T, d_model] (Rich periodic features)
        # Note: De-normalization은 TaXer에서 수행
        return enc_out
