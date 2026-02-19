import torch

def GradientClipping(parameters, max_norm):
    """
    实现梯度裁剪：原地（in-place）修改参数的梯度。
    说白了就是梯度的归一化
    """
    # 1. 收集所有有梯度的参数
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return

    # 2. 计算全局 L2 范数 (所有参数梯度的平方和再开方)
    # PyTorch 默认使用 float64 累加防止溢出
    total_norm = torch.norm(
        torch.stack([torch.norm(g, 2) for g in grads]), 
        2
    )

    # 3. 计算缩放系数
    eps = 1e-6
    # 只有当 total_norm > max_norm 时才进行缩放
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + eps)
        
        # 4. 原地修改每个梯度
        for g in grads:
            g.detach().mul_(clip_coef)