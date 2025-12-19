"""
사용 가능한 모델:
    - iTransformer: Inverted Transformer (ICLR 2024)
        변수 간 관계를 모델링하는 역전된 Transformer

    - TimeXer: Time Series Transformer with Exogenous Variables (NeurIPS 2024)
        Exogenous variable을 처리하는 Patch 기반 Transformer

    - WaveNet: Dilated Causal Convolution Network (DeepMind 2016)
        지수적으로 증가하는 dilation으로 long-range dependency 포착

    - TimesNet: Temporal 2D-Variation Modeling (ICLR 2023)
        FFT 기반 period 탐지와 2D convolution으로 주기적 pattern 포착

    - WaXer: TimeXer + WaveNet Hybrid 모델 (본 프로젝트)
        WaveNet feature를 TimeXer의 exogenous input으로 활용

    - TaXer: TimeXer + TimesNet Hybrid 모델 (본 프로젝트)
        TimesNet feature를 TimeXer의 exogenous input으로 활용

Example:
    >>> from models import WaXer, TaXer
    >>> model = WaXer(seq_len=512, pred_len=16, d_model=32)
    >>> output = model(x_enc)  # [B, pred_len, n_vars]
"""

from .iTransformer.iTransformer import iTransformer
from .TimeXer.TimeXer import TimeXer
from .WaveNet.WaveNet import WaveNetForecaster as WaveNet
from .TimesNet.TimesNet import TimesNet, TimesNetFeatureExtractor
from .TaXer.TaXer import TaXer
from .WaXer.WaXer import WaXer


__all__ = [
    "iTransformer",
    "TimeXer",
    "WaveNet",
    "TimesNet",
    "TimesNetFeatureExtractor",
    "TaXer",
    "WaXer",
]
