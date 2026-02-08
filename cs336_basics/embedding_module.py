from math import sqrt
import torch
from torch import nn

class EmbeddingModel(nn.Module):
    # token数， embedding字典维度
    def __init__(self, num_embeddings, embeddings_dim, device = None, dtype = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embeddings_dim =  embeddings_dim
        self.device = device
        self.dtype = dtype
        
        self.W = nn.Parameter(torch.empty(self.num_embeddings, self.embeddings_dim, device = self.device, dtype = self.dtype))

        # std = sqrt(2) / (self.out_features + self.in_features) ** 0.5
        # torch.nn.init.trunc_normal_(self.W, mean = 0, std = std, a = -3 * std, b = 3 * std)      
        std = sqrt(2) / (self.num_embeddings + self.embeddings_dim)
        torch.nn.init.trunc_normal_(self.W, mean = 0.0, std = std, a = -3 * std, b = 3 * std)

        

    def forward(self, token_ids:torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]