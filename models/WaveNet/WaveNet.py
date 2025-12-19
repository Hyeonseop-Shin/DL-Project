"""
WaveNet: Dilated Causal Convolution Network for Time Series Forecasting

이 module은 DeepMind의 WaveNet architecture를 시계열 예측에 적용한
모델을 구현합니다.

Architecture:
    Input → Input Projection → WaveNet Blocks (Dilated Conv) → Skip Aggregation → Output

주요 특징:
    - Dilated causal convolution: 지수적으로 증가하는 receptive field
    - Gated activation: Sigmoid gate로 정보 흐름 제어
    - Skip connection: 모든 layer의 feature를 집약
    - Causal: 미래 정보를 사용하지 않음 (autoregressive에 적합)

Dilation Pattern:
    Layer 0: dilation = 1  → receptive field = 3
    Layer 1: dilation = 2  → receptive field = 7
    Layer 2: dilation = 4  → receptive field = 15
    Layer 3: dilation = 8  → receptive field = 31
    ...
    총 receptive field = 2^layers * (kernel_size - 1) + 1

참고 논문:
    - WaveNet: A Generative Model for Raw Audio (DeepMind 2016)
    - https://arxiv.org/abs/1609.03499

클래스:
    WaveNetForecaster: WaveNet 기반 시계열 예측 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.Conv import CausalConv1d, WaveNetBlock


class WaveNetForecaster(nn.Module):
    """
    WaveNetForecaster: Dilated Causal Convolution 시계열 예측 모델

    지수적으로 증가하는 dilation으로 long-range dependency를 효율적으로 포착합니다.
    각 layer의 skip connection을 합산하여 최종 prediction을 생성합니다.

    Args:
        seq_len (int): Input sequence 길이. Default: 90
        pred_len (int): Prediction 길이. Default: 30
        c_in (int): Input channel 수 (변수 개수). Default: 7
        d_model (int): Hidden channel 수. Default: 64
        dropout (float): Dropout rate. Default: 0.1
        layers (int): WaveNet block 수. Default: 4
        kernel_size (int): Convolution kernel 크기. Default: 3

    Attributes:
        input_projection: Input channel → d_model projection
        blocks: WaveNet block list (dilated convolutions)
        final_conv1: Skip connection 후 1x1 convolution
        final_projection: d_model → pred_len * c_in projection

    Shape:
        - Input: [B, seq_len, n_vars]
        - Output: [B, pred_len, n_vars]

    Example:
        >>> model = WaveNetForecaster(seq_len=512, pred_len=16, c_in=30)
        >>> x = torch.randn(32, 512, 30)
        >>> output = model(x)  # [32, 16, 30]
    """

    def __init__(self,
                 seq_len=90,
                 pred_len=30,
                 c_in=7,
                 d_model=64,
                 dropout=0.1,
                 layers=4,
                 kernel_size=3):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len

        # ============================================
        # Input Projection
        # - c_in (변수 수) → d_model로 channel 확장
        # - 1x1 convolution으로 point-wise projection
        # ============================================
        self.input_projection = nn.Conv1d(c_in, d_model, kernel_size=1)

        # ============================================
        # WaveNet Blocks
        # - 각 block은 dilated causal convolution 사용
        # - Dilation: 1, 2, 4, 8, ... (2^i)
        # - Residual + Skip connection 구조
        # ============================================
        self.blocks = nn.ModuleList()
        self.skip_channels = d_model

        for i in range(layers):
            # Dilation이 지수적으로 증가: 2^0, 2^1, 2^2, ...
            dilation = 2 ** i
            self.blocks.append(
                WaveNetBlock(
                    residual_channels=d_model,
                    skip_channels=self.skip_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )

        # ============================================
        # Output Layers
        # - Skip connection들을 합산한 후 처리
        # - 최종적으로 pred_len * c_in 차원으로 projection
        # ============================================
        self.final_conv1 = nn.Conv1d(self.skip_channels, self.skip_channels, kernel_size=1)
        self.final_projection = nn.Linear(d_model, pred_len * c_in)

    def forward(self, x_enc, x_mark_enc=None):
        """
        Forward pass

        Args:
            x_enc (Tensor): Input 시계열 [B, T, N]
                - B: Batch size
                - T: Sequence length
                - N: Number of variables
            x_mark_enc (Tensor, optional): Time feature (미사용)

        Returns:
            Tensor: Prediction [B, pred_len, N]
        """
        # x_enc shape: [B, T, N]

        # ============================================
        # 1단계: Instance Normalization
        # - 각 변수별로 mean/std 계산
        # - Distribution shift 문제 완화
        # ============================================
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        B, T, N = x_enc.shape

        # ============================================
        # 2단계: Transpose for Conv1d
        # - Conv1d는 [B, C, T] 형태를 기대
        # - [B, T, N] → [B, N, T]
        # ============================================
        x = x_enc.transpose(1, 2)  # [B, N, T]

        # ============================================
        # 3단계: Input Projection + WaveNet Blocks
        # - Input channel을 d_model로 확장
        # - 각 block에서 dilated causal convolution 수행
        # - Skip connection들을 누적하여 합산
        # ============================================
        x = self.input_projection(x)  # [B, d_model, T]

        skip_connections = 0
        for block in self.blocks:
            # 각 block은 (residual_output, skip_output)을 반환
            x, skip = block(x)
            skip_connections += skip  # Skip connection 누적 합산

        # ============================================
        # 4단계: Output Generation
        # - Skip connection 합산 결과에 ReLU + 1x1 Conv
        # - 마지막 time step의 feature로 prediction 생성
        # ============================================
        x = F.relu(skip_connections)
        x = F.relu(self.final_conv1(x))  # [B, d_model, T]

        # 마지막 time step 사용 (causal하므로 최신 정보가 집약됨)
        x_last = x[:, :, -1]  # [B, d_model]

        # Final projection: d_model → pred_len * N
        dec_out = self.final_projection(x_last)  # [B, pred_len * N]
        dec_out = dec_out.view(B, self.pred_len, N)  # [B, pred_len, N]

        # ============================================
        # 5단계: De-normalization
        # - 원래 scale로 복원
        # ============================================
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out
