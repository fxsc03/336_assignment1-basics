import torch
from torch import nn
from .RoPE import RoPE
from .RMSnorm import RMSNorm
from .multi_head_self_attention import MultiHeadAttention_rope
from .SwiGLU import SwiGLU


class TransformerBlock(nn.Module):
    """
    TransformerBlock 是Transformer块，它把包含多头注意力机制的一些组件包装在一起，形成一个完整的Transformer块。
    Args:
        d_model (int): 输入的维度，也就是d_model
        n_heads (int): 头的数量
        d_ff (int): 前馈神经网络的维度
        max_seq_len (int): 最大序列长度
        theta (float): 底数超参数
        device: 设备
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, theta: float, device=None):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device

        # 在内部创建注意力权重 (d_model, d_model)
        self.attn_q_proj_weight = nn.Parameter(torch.empty(d_model, d_model))
        self.attn_k_proj_weight = nn.Parameter(torch.empty(d_model, d_model))
        self.attn_v_proj_weight = nn.Parameter(torch.empty(d_model, d_model))
        self.attn_o_proj_weight = nn.Parameter(torch.empty(d_model, d_model))

        self.rms_norm1 = RMSNorm(d_model, eps=1e-5, device=device)
        self.rms_norm2 = RMSNorm(d_model, eps=1e-5, device=device)
        self.swiglu = SwiGLU(d_model, d_ff)
        self.causal_multi_head_attention = MultiHeadAttention_rope(d_model, n_heads, max_seq_len, theta, device)

    def set_para(self, block_weights: dict):
        """从参考实现的 state_dict 片段加载本层权重。"""
        with torch.no_grad():
            self.attn_q_proj_weight.copy_(block_weights["attn.q_proj.weight"])
            self.attn_k_proj_weight.copy_(block_weights["attn.k_proj.weight"])
            self.attn_v_proj_weight.copy_(block_weights["attn.v_proj.weight"])
            self.attn_o_proj_weight.copy_(block_weights["attn.output_proj.weight"])
            self.rms_norm1.W.copy_(block_weights["ln1.weight"])
            self.rms_norm2.W.copy_(block_weights["ln2.weight"])
            # 参考实现 ffn 权重形状与 PyTorch Linear 约定不同，w1 需转置
            self.swiglu.W1.weight.copy_(block_weights["ffn.w1.weight"])
            self.swiglu.W2.weight.copy_(block_weights["ffn.w3.weight"])
            self.swiglu.W3.weight.copy_(block_weights["ffn.w2.weight"])

    def forward(self,in_features:torch.Tensor):
        token_positions = torch.arange(in_features.shape[1],device=in_features.device)
        x1 = self.rms_norm1(in_features)
        x1 = self.causal_multi_head_attention(x1,self.attn_q_proj_weight,self.attn_k_proj_weight,self.attn_v_proj_weight,self.attn_o_proj_weight,token_positions)
        x1 = x1 + in_features
        x2 = self.rms_norm2(x1)
        x2 = self.swiglu(x2)
        out = x2 + x1
        return out
