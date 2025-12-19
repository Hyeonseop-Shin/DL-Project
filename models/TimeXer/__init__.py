"""
TimeXer Package

TimeXer Transformer 시계열 예측 모델을 제공합니다.

주요 컴포넌트:
    - TimeXer: Exogenous variable을 활용하는 Transformer 모델

핵심 아이디어:
    - Patch embedding으로 시계열 분할
    - Global token으로 전체 sequence 정보 요약
    - Cross-attention으로 exogenous variable 통합

참고 논문:
    - TimeXer: Empowering Transformers for Time Series Forecasting with
      Exogenous Variables (NeurIPS 2024)

사용 예시:
    >>> from models.TimeXer import TimeXer
    >>> model = TimeXer(seq_len=512, pred_len=16, d_model=64)
    >>> output = model(x, ex_features)  # [B, pred_len, n_vars]
"""

from .TimeXer import TimeXer

__all__ = ['TimeXer']
