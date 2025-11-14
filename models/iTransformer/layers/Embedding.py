
import torch
import torch.nn as nn

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()

        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # x: [batch, time, variable]
        # x_mark: [batch, time, time_feature], time_feature: month, day, weekday, hour

        x = x.permute(0, 2, 1)  # [batch, variable, time]

        if x_mark == None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(
                torch.cat((x, x_mark.permute(0, 2, 1)), 1)) # concatenate them in horizontal
        # x: [batch, variable + time_feature, time]

        x = self.dropout(x)
        return x    
        