from turtle import forward
import torch 
from torch import nn

class RMSNorm(nn.Module):
    """
     input:
        x: (batch_size, seq_len, d_model) 输入的稠密向量
    output:
        x_norm: (batch_size, seq_len, d_model) 归一化后的稠密向量
    """
    def __init__(self, d_model: int, eps:float = 1e-5, device = None, dtype = None):
        super().__init__()
        # dim
        self.d_model = d_model
        self.eps = eps
        self.device = device 
        self.dtype = dtype

        self.W = nn.Parameter(torch.ones(self.d_model, device = self.device, dtype = self.dtype))

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        x = x.float() # 全部转成float32归一化

        # dim = -1 只在d_model维度做计算
        RMS = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = (x / RMS) * self.W
        return out.to(original_dtype)