import torch
import numpy as np
import numpy.typing as npt

def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    
    # 1. 确定数据集总长度
    n = len(dataset)
    
    # 2. 随机采样 batch_size 个起始位置
    # 这里的上限是 n - context_length - 1，确保 y 序列不会越界
    # 给出 batch_size 个 0 ~ n - context_length的起始位置
    ix = torch.randint(0, n - context_length, (batch_size,))
    
    # 3. 构造输入 x 和 目标 y
    # 我们遍历每个随机索引，切片出长度为 context_length 的片段
    x_list = [dataset[i : i + context_length] for i in ix]
    y_list = [dataset[i + 1 : i + context_length + 1] for i in ix]
    
    # 4. 转换为 PyTorch 张量
    # 使用 np.stack 组合成 (batch_size, context_length) 的矩阵，再转为 Tensor
    x = torch.from_numpy(np.stack(x_list)).to(torch.long)
    y = torch.from_numpy(np.stack(y_list)).to(torch.long)
    
    # 5. 移动到目标设备 (CPU/GPU)
    return x.to(device), y.to(device)