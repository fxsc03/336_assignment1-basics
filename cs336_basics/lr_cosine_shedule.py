import math

def get_lr_cosine_schedule(
    it, 
    max_learning_rate, 
    min_learning_rate, 
    warmup_iters, 
    cosine_cycle_iters) -> float:
    # it是当前迭代步数,warmup_iters是预热的迭代总步数

    if it <= warmup_iters:
        return max_learning_rate * (it / warmup_iters)

    if it >= cosine_cycle_iters:
        return min_learning_rate
    
    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)


    
