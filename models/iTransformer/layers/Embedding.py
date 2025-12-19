"""iTransformer Embedding Layers"""

import torch
import torch.nn as nn


class DataEmbedding_inverted(nn.Module):
    """Inverted Embedding: variables as tokens"""

    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        """
        Args:
            x: [B, T, N]
            x_mark: [B, T, F] (optional)
        Returns:
            embedded: [B, N (+F), d_model]
        """
        x = x.permute(0, 2, 1)  # [B, N, T]

        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat((x, x_mark.permute(0, 2, 1)), 1))

        return self.dropout(x)
