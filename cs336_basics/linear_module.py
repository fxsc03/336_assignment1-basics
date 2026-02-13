from math import sqrt
import torch
from torch import nn

class LinearModule(nn.Module):
    # 这里的in_features和out_features都是
    def __init__(self, in_features, out_features, device = None, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        # out*in 计算的时候再转职，对连续访存友好
        self.W = nn.Parameter(torch.empty(self.out_features, self.in_features, device = self.device, dtype = self.dtype))

        std = sqrt(2) / (self.out_features + self.in_features) ** 0.5
        torch.nn.init.trunc_normal_(self.W, mean = 0.0, std = std, a = -3 * std, b = 3 * std)      



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T