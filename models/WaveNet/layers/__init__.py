"""
WaveNet Layer Components

WaveNet 모델의 핵심 layer들을 제공합니다.

주요 컴포넌트:
    - CausalConv1d: 미래 정보를 사용하지 않는 causal convolution
    - WaveNetBlock: Gated activation + skip connection block

핵심 아이디어:
    - Dilated causal convolution으로 넓은 receptive field 확보
    - Gated activation: tanh(filter) * sigmoid(gate)
    - Skip connection으로 multi-scale feature aggregation
"""

from .Conv import CausalConv1d, WaveNetBlock

__all__ = ['CausalConv1d', 'WaveNetBlock']
