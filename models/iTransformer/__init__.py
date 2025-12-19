"""
iTransformer Package

iTransformer (Inverted Transformer) 시계열 예측 모델을 제공합니다.

주요 컴포넌트:
    - iTransformer: 변수를 token으로 처리하는 역전된 Transformer

핵심 아이디어:
    - 일반 Transformer: time step을 token으로 처리
    - iTransformer: 변수를 token으로 처리 (inverted)
    - Variable-wise self-attention으로 변수 간 상관관계 학습

참고 논문:
    - iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
      (ICLR 2024 Spotlight)

사용 예시:
    >>> from models.iTransformer import iTransformer
    >>> model = iTransformer(seq_len=512, pred_len=16, d_model=64)
    >>> output = model(x)  # [B, pred_len, n_vars]
"""

from .iTransformer import iTransformer

__all__ = ['iTransformer']
