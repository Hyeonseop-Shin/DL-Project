"""
TimesNet Package

TimesNet 기반 시계열 예측 모델을 제공합니다.

주요 컴포넌트:
    - TimesNet: 2D temporal variation modeling 예측 모델
    - TimesNetFeatureExtractor: TaXer용 feature extractor

핵심 아이디어:
    - FFT로 시계열의 주요 period (주기) 탐지
    - 1D 시계열을 2D tensor로 reshape하여 Inception block 적용
    - Intra-period, inter-period variation을 2D convolution으로 포착

참고 논문:
    - TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis
      (ICLR 2023)

사용 예시:
    >>> from models.TimesNet import TimesNet, TimesNetFeatureExtractor
    >>> model = TimesNet(seq_len=512, pred_len=16, c_in=30)
    >>> output = model(x)  # [B, pred_len, n_vars]
"""

from .TimesNet import TimesNet, TimesNetFeatureExtractor

__all__ = ['TimesNet', 'TimesNetFeatureExtractor']
