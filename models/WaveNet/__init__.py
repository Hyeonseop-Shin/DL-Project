"""
WaveNet Package

WaveNet 기반 시계열 예측 모델을 제공합니다.

주요 컴포넌트:
    - WaveNetForecaster: Dilated causal convolution 기반 예측 모델

핵심 아이디어:
    - Dilated causal convolution으로 지수적으로 증가하는 receptive field 확보
    - Gated activation으로 정보 흐름 제어
    - Skip connection으로 multi-scale feature aggregation

참고 논문:
    - WaveNet: A Generative Model for Raw Audio (DeepMind 2016)

사용 예시:
    >>> from models.WaveNet import WaveNetForecaster
    >>> model = WaveNetForecaster(seq_len=512, pred_len=16, c_in=30)
    >>> output = model(x)  # [B, pred_len, n_vars]
"""

from .WaveNet import WaveNetForecaster

__all__ = ['WaveNetForecaster']
