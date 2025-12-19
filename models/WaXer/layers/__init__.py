"""
WaXer Layer Components

WaXer (TimeXer + WaveNet) 모델의 핵심 layer들을 제공합니다.

Embedding:
    - PositionalEmbedding: Sinusoidal positional encoding
    - EnEmbedding: Patch embedding + Global token

Attention:
    - FullAttention: Scaled dot-product attention
    - AttentionLayer: Multi-head attention wrapper

Encoder:
    - FlattenHead: Output projection head
    - EncoderLayer: Self-attention + Cross-attention layer
    - Encoder: Encoder layer stack

WaveNet:
    - CausalConv1d: Causal convolution
    - WaveNetBlock: Gated activation block
    - WaveNetFeatureExtractor: Feature extractor for exogenous input
"""

from .Embed import PositionalEmbedding, EnEmbedding
from .Attention import FullAttention, AttentionLayer
from .Encoder import FlattenHead, EncoderLayer, Encoder
from .WaveNet import CausalConv1d, WaveNetBlock, WaveNetFeatureExtractor

__all__ = [
    'PositionalEmbedding',
    'EnEmbedding',
    'FullAttention',
    'AttentionLayer',
    'FlattenHead',
    'EncoderLayer',
    'Encoder',
    'CausalConv1d',
    'WaveNetBlock',
    'WaveNetFeatureExtractor',
]
