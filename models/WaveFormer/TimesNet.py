
### TimesNet (Long-Term Forecasting for Periodic Time Series)
### by Manyoung Han (2025/11/25)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

class Inception_Block(nn.Module):
    """
    2D Inception Block for processing 2D variations of time series.
    Standard Inception architecture with kernels [1, 3, 5, 7].
    """
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class TimesBlock(nn.Module):
    """
    Core module of TimesNet.
    1. FFT to find top-k periods.
    2. Reshape 1D time series to 2D based on period.
    3. Apply 2D Inception Block.
    4. Aggregate results based on amplitude weights.
    """
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ff, num_kernels):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        
        # 2D Convolutions (Inception Block)
        self.conv = nn.Sequential(
            Inception_Block(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(), # Gaussian Error Linear Unit
            Inception_Block(d_ff, d_model, num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        
        # 1. FFT to find Top-k periods
        period_list, period_weight = self.FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            
            # 2. Reshape 1D -> 2D
            # Generating padding for 2D data translation
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            
            # Reshape: [B, T, N] -> [B, N, Length // Period, Period]
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            
            # 3. 2D Conv
            out = self.conv(out)
            
            # [B, N, Length // Period, Period] -> [B, Length, N]
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            
            # return to zero-padding
            res.append(out[:, :(self.seq_len + self.pred_len), :])

        # 4. Aggregation (Weighted Sum)
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1) # softmax in k-direction
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1) # sum in k-direction
        
        # Residual connection
        res = res + x
        return res

    def FFT_for_Period(self, x, k=2):
        xf = torch.fft.rfft(x, dim=1)
        frequency_list = abs(xf).mean(0).mean(-1)  # batch mean, channel mean
        frequency_list[0] = 0 # Ignore DC component
        
        _, top_list = torch.topk(frequency_list, k)
        top_list = top_list.detach().cpu().numpy()
        
        period = x.shape[1] // top_list
        
        return period, abs(xf).mean(-1)[:, top_list] # channel mean


class TimesNet(nn.Module):
    def __init__(self,
        seq_len=90,      # Input Sequence Length
        pred_len=30,     # Prediction Length
        c_in=7,          # Number of Input Variables
        c_out=7,         # Number of Output Variables (usually same as c_in)
        d_model=64,      # Model Dimension
        d_ff=64,         # Hidden Size in Inception
        top_k=5,         # Number of Top Periods
        num_kernels=6,   # Number of Inception Kernels
        e_layers=2,      # Number of TimesBlocks
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.c_out = c_out
        
        # Embedding Layer
        self.enc_embedding = nn.Linear(c_in, d_model)
        
        # TimesBlocks
        self.layer = e_layers
        self.model = nn.ModuleList([
            TimesBlock(seq_len, pred_len, top_k, d_model, d_ff, num_kernels)
            for _ in range(e_layers)
        ])
        
        # For forecasting: Map seq_len -> seq_len + pred_len in time axis
        self.predict_linear = nn.Linear(self.seq_len, self.seq_len + self.pred_len)
        
        # Final Projection
        self.projection = nn.Linear(d_model, c_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_enc, x_mark_enc=None):
        # x_enc shape: [B, Seq_Len, N]
        # 1. Normalization (RevIN style inside)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # 2. Embedding : [B, T, N] -> [B, T, d_model]
        enc_out = self.enc_embedding(x_enc)
        
        # 3. Extend Time Axis for Forecasting
        # [B, T, d_model] -> [B, d_model, T] -> Linear -> [B, d_model, T+Pred] -> [B, T+Pred, d_model]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 4. Process TimesBlocks
        # The blocks process the extended sequence (T + Pred)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i], enc_out)

        # 5. Final Projection [B, T+Pred, d_model] -> [B, T+Pred, N]
        dec_out = self.projection(enc_out)
        
        # 6. Slice the Prediction Part (Last pred_len)
        dec_out = dec_out[:, -self.pred_len:, :]
        
        # 7. De-Normalization : stdev, means shape: [B, 1, N] -> Broadcast to [B, pred_len, N]
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def layer_norm(self, layer, enc_out):
        # Residual connection + LayerNorm style wrapper
        # Note: TimesBlock internally has residual, but usually we apply Norm after block
        # In official code, they apply TimesBlock then simply pass. 
        return layer(enc_out)