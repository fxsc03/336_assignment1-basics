import math
import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

    def attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # seq_len, head_dim
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            # ✅ 修正点 1：屏蔽 mask 为 1 (True) 的部分，也就是屏蔽"未来"
            # triu 生成的上三角是 1，我们要把它填成 -inf
            scores = scores.masked_fill(mask == 1 , float('-inf'))
            
        tmp = torch.softmax(scores, dim = -1)
        return torch.matmul(tmp, V)
        
    def forward(self, x, wq, wk, wv, wo) -> torch.Tensor:
        # x: (batch, seq, d_model)
        
        q = x @ wq.T
        k = x @ wk.T
        v = x @ wv.T
        
        batch_size, seq_len, d_model = x.shape
        
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, num_heads, head_dim]
        # 注意：这里必须赋值 (你原来写的对)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # ✅ 修正点 2：transpose 必须赋值给变量！
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # mask 矩阵：上三角为 1 (未来)，下三角为 0 (历史)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        # 扩展维度以匹配 (Batch, Heads, Seq, Seq)
        mask = mask.unsqueeze(0).unsqueeze(0) 
    
        out = self.attention(q, k, v, mask)
        
        # ✅ 修正点 3：transpose 必须赋值
        out = out.transpose(1, 2)
        # contiguous() 是必须的，否则 view 会报错
        out = out.contiguous().view(batch_size, seq_len, d_model)
        
        out = out @ wo.T
        return out