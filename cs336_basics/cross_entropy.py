import torch
from .softmax import softmax

# 交叉熵损失函数
def CrossEntropyLoss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits: (batch, ..., vocab_size)  预测值
    targets: (batch, ...)             真实标签索引
    """
    # max_logits类似于归一化，用来防止e的指数过大
    max_logits, _ = logits.max(dim = -1, keepdim = True)

    log_sum_exp = max_logits + torch.log(torch.sum(torch.exp(logits - max_logits), dim = -1, keepdim = True))
    
    target_logits = logits.gather(dim = -1, index = targets.unsqueeze(-1))

    loss = log_sum_exp - target_logits

    return loss.mean()