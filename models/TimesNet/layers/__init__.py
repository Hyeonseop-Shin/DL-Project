"""
TimesNet Layer Components

TimesNet 모델의 핵심 layer들을 제공합니다.

주요 컴포넌트:
    - FFT_for_Period: FFT로 시계열의 주요 주기 탐지
    - Inception_Block: Multi-scale 2D convolution block
    - TimesBlock: 1D → 2D → Inception → 1D 변환 블록

핵심 아이디어:
    - FFT로 dominant frequency 탐지 → period 계산
    - 1D 시계열을 2D tensor로 reshape (period 기반)
    - 2D Inception block으로 intra/inter-period pattern 포착
"""

from .Inception import Inception_Block, TimesBlock, FFT_for_Period

__all__ = ['Inception_Block', 'TimesBlock', 'FFT_for_Period']
