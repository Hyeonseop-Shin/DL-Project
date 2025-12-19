"""
WaXer Package

WaXer (WaveNet + TimeXer) hybrid 시계열 예측 모델을 제공합니다.

주요 컴포넌트:
    - WaXer: TimeXer + WaveNet feature extraction

핵심 아이디어:
    - WaveNet의 dilated causal convolution으로 long-range temporal pattern 추출
    - 추출된 feature를 TimeXer의 exogenous variable로 사용
    - Cross-attention으로 endogenous/exogenous 정보 통합

사용 예시:
    >>> from models.WaXer import WaXer
    >>> model = WaXer(seq_len=512, pred_len=16, d_model=64)
    >>> output = model(x)  # [B, pred_len, n_vars]
"""

from .WaXer import WaXer

__all__ = ['WaXer']
