import math # 必须导入 math
import torch
from torch import optim

class AdamW(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None): # 补全标准接口
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                
                # 初始化懒加载
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                
                m = state['m']
                v = state['v']
                state['step'] += 1
                t = state['step']
                grad = p.grad

                # 计算新的 m 和 v
                m_new = beta1 * m + (1 - beta1) * grad
                v_new = beta2 * v + (1 - beta2) * (grad ** 2)

                # 错误修正 1: torch.sqrt 不能处理标量 float，必须用 math.sqrt
                # 且为了清晰，提取出 bias_correction
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                
                # 错误修正 2 (最致命): 必须使用原地操作 (In-place) 修改 p.data
                # 你原来的写法 p = p - ... 只是修改了局部变量，模型参数没有变！
                denom = v_new.sqrt().add_(eps)
                p.data.addcdiv_(m_new, denom, value=-step_size) # 等价于 p -= step_size * m / denom

                # 错误修正 3: 权重衰减也必须原地修改
                # 你原来的写法 p = p - ... 无效
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                # 更新状态
                state['m'] = m_new
                state['v'] = v_new
        
        return loss