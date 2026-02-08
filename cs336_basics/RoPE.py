from functools import total_ordering
import torch
from torch import nn

# 好处是不会破坏原本的向量
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        freqs = 1.0/(self.theta ** (torch.arange(0, self.d_k, 2).float() / self.d_k))
        pos = torch.arange(self.max_seq_len, device = device)
        # max_seq , d_k/2 每一行对应每个token的旋转角度
        sinusoids = torch.outer(pos, freqs)

        self.register_buffer("cos_cache", sinusoids.cos(), persistent = False)
        self.register_buffer("sin_cache", sinusoids.sin(), persistent = False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:

        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        # 把x切成偶列和奇列
        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]

        out_even = x_even * cos - x_odd * sin
        out_odd  = x_even * sin + x_odd * cos

        out = torch.stack([out_even, out_odd], dim = -1)
        out = out.flatten(-2)

        return out
