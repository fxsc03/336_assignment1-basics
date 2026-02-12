import math
import torch 
from torch import nn

class scaleddotproductattention(nn.Module):
    """
    ScaledDotProductAttention 是缩放点积注意力，它通过将输入的稠密向量与输入的稠密向量进行点积来得到输出。
    公式是：
    out = softmax(QK^T / sqrt(d_k))V
    Args:
        Q: (batch_size, seq_len, d_k) 查询向量
        K: (batch_size, seq_len, d_k) 键向量
        V: (batch_size, seq_len, d_k) 值向量
        mask: (batch_size, seq_len, seq_len) 掩码
    output:
        out: (batch_size, seq_len, d_k) 输出的稠密向量
    """
    def __init__(self) -> None:
        super().__init__()
    def forward(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask: torch.Tensor | None = None):
        # d_k是最后一维的大小
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        tmp = torch.softmax(scores, dim = -1)
        final_ans = torch.matmul(tmp, V)
        return final_ans
            


