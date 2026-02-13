import torch 
from torch import nn
from .transformer_block import TransformerBlock
from .embedding_module import EmbeddingModel
from .RMSnorm import RMSNorm
from .linear_module import LinearModule

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device=None):
        super().__init__()
        self.device = device
        self.num_layers = num_layers
        
        # 1. Embedding
        # 假设你的 EmbeddingModel 已经写好
        self.embdedding = EmbeddingModel(vocab_size, d_model)
        
        # 2. 多层 Block (关键修改!)
        # 使用 ModuleList 创建 num_layers 个独立的 Block
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=num_heads, # 修正变量名
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device
                # 去掉了具体的权重参数
            ) for _ in range(num_layers)
        ])
        
        # 3. Final Norm
        self.norm = RMSNorm(d_model=d_model, device=device)
        
        # 4. LM Head
        self.linear = LinearModule(d_model, vocab_size) # 你的 LinearModule

    def forward(self, x):
        x = self.embdedding(x)
        
        # 循环经过每一层
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        x = self.linear(x)
        return x


        
