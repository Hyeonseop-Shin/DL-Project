
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from .Attention import AttentionLayer

class EncoderLayer(nn.Module):
    def __init__(self, attention: AttentionLayer,
                 d_model=512,
                 d_ff=None,
                 dropout=0.1,
                 activation='relu'):
        super().__init__()

        d_ff = d_ff or 4 * d_model  # ref.to Transformer paper
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)   # feed forward
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)   # feed forward
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    
    def forward(self, x):
        new_x, attn = self.attention(
            queries=x, 
            keys=x, 
            values=x,
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)   # [batch, query_len, d_model]
        y = self.activation(self.conv1(y.transpose(-1, 1)))
        y = self.dropout(y)
        y = self.conv2(y).transpose(-1, 1)
        y = self.dropout(y)

        return self.norm2(x + y), attn
    

    # nn.Conv1d(c_in, c_out, kernel_size) makes weight W.shape = [c_out, c_in, kernel_size]
    # if kernel_size == 1, Consider it matmul(W, x), output shape would be [c_out, x[-1]]
    # that's the reason why the transpose needed


class Encoder(nn.Module):
    def __init__(self, attn_layers: List[EncoderLayer], 
                 norm_layer: nn.LayerNorm):
        super().__init__()

        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm_layer = norm_layer


    def forward(self, x):
        # x: [batch, query_len=variable + time_feature, d_model]

        attentions = list()
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x)
            attentions.append(attn)
        
        if self.norm_layer is not None:
            x = self.norm_layer(x)

        # x: [batch, query_len, d_model]
        return x, attentions