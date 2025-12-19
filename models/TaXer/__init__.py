"""
TaXer Package

TaXer (TimesNet + TimeXer) hybrid 시계열 예측 모델을 제공합니다.

주요 컴포넌트:
    - TaXer: TimeXer + TimesNet feature extraction

핵심 아이디어:
    - TimesNet의 FFT 기반 period 탐지로 주기적 pattern 발견
    - 2D Inception convolution으로 multi-scale temporal feature 추출
    - Cross-attention으로 endogenous/exogenous 정보 통합

사용 예시:
    >>> from models.TaXer import TaXer
    >>> model = TaXer(seq_len=512, pred_len=16, d_model=64)
    >>> output = model(x)  # [B, pred_len, n_vars]
"""

from .TaXer import TaXer

__all__ = ['TaXer']
