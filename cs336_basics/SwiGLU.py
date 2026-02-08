from turtle import forward
import torch
from torch import nn

class SwiGLU(nn.Module):

    "FFN = W2 * (SiLU(W1 x) ⊙ W3 x)"
    # 输入维度，输出维度
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.W1 = nn.Linear(d_model, d_ff, bias = False)
        self.W2 = nn.Linear(d_model, d_ff, bias = False)
        self.W3 = nn.Linear(d_ff, d_model, bias = False)

    def silu(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        return self.W3(self.silu(self.W1(x)) * self.W2(x))