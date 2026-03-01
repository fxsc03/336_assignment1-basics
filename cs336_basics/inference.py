"""
从训练好的 Transformer LM 解码生成文本。
支持：prompt 续写、最大生成长度、temperature 缩放、top-p (nucleus) 采样，遇 <|endoftext|> 停止。
"""
import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """对 logits 做 temperature 缩放后 softmax 得到概率。temperature > 1 更随机，< 1 更确定。"""
    if temperature <= 0:
        raise ValueError("temperature 必须为正数")
    return torch.softmax(logits / temperature, dim=-1)


def apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Top-p (nucleus) 采样：保留累积概率达到 top_p 的最小 token 集合，其余置 0 后重新归一化。
    probs: (..., vocab_size)
    top_p: 阈值，例如 0.9 表示从累积概率达到 90% 的 token 中采样。
    """
    if top_p >= 1.0 or top_p <= 0:
        return probs
    probs_sorted, indices = torch.sort(probs, dim=-1, descending=True)
    cumsum = torch.cumsum(probs_sorted, dim=-1)
    mask = cumsum > top_p
    mask[..., 0] = False
    probs_sorted = probs_sorted.masked_fill(mask, 0.0)
    probs_sorted = probs_sorted / (probs_sorted.sum(dim=-1, keepdim=True) + 1e-10)
    probs_out = torch.zeros_like(probs, dtype=probs.dtype, device=probs.device)
    probs_out.scatter_(-1, indices, probs_sorted)
    return probs_out


def decode(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    endoftext_token_id: int | None = None,
    context_length: int | None = None,
) -> torch.Tensor:
    """
    根据 prompt 自回归生成，直到遇到 <|endoftext|> 或达到 max_new_tokens。

    Args:
        model: TransformerLM，forward 输入 (batch, seq) 输出 (batch, seq, vocab_size)
        prompt_ids: 一维 long tensor，prompt 的 token id 序列（或 (1, seq)）
        max_new_tokens: 最多新生成的 token 数
        temperature: 对 next-token logits 的 temperature，>1 更随机，<1 更确定
        top_p: nucleus 采样阈值，1.0 表示不做 top-p
        endoftext_token_id: 遇到该 token 时停止生成；None 则不因该 token 停止
        context_length: 模型最大上下文长度；若提供则只取 prompt 最后 context_length 个 token 作为上下文

    Returns:
        (1, seq_len) 的 long tensor，为 prompt + 生成的全部 token（含停止符若遇到）。
    """
    model.eval()
    device = next(model.parameters()).device
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    prompt_ids = prompt_ids.to(device)
    if context_length is not None and prompt_ids.size(1) > context_length:
        prompt_ids = prompt_ids[:, -context_length:]
    generated = prompt_ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if context_length is not None and generated.size(1) > context_length:
                model_input = generated[:, -context_length:]
            else:
                model_input = generated
            logits = model(model_input)
            next_logits = logits[:, -1, :]
            probs = apply_temperature(next_logits, temperature)
            if top_p < 1.0:
                probs = apply_top_p(probs, top_p)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if endoftext_token_id is not None and (next_token == endoftext_token_id).all().item():
                break
    return generated


def load_model_from_checkpoint(
    checkpoint_path: str,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: float,
    rope_theta: float = 10000.0,
    device: str | None = None,
):
    """从 train.py 保存的 checkpoint 加载模型（仅模型权重，不加载 optimizer）。"""
    from cs336_basics.transformer_lm import TransformerLM
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=int(d_ff),
        rope_theta=rope_theta,
        device=device,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model


def main():
    p = argparse.ArgumentParser(description="从 checkpoint 加载模型并对 prompt 做解码生成")
    p.add_argument("--checkpoint", type=str, required=True, help="checkpoint 路径（.pt）")
    p.add_argument("--vocab_path", type=str, required=True, help="vocab.json 路径")
    p.add_argument("--merges_path", type=str, required=True, help="merges.txt 路径")
    p.add_argument("--prompt", type=str, default="Once upon a time", help="文本 prompt")
    p.add_argument("--max_new_tokens", type=int, default=100, help="最多生成 token 数")
    p.add_argument("--temperature", type=float, default=0.8, help="softmax temperature")
    p.add_argument("--top_p", type=float, default=0.9, help="nucleus 采样 top_p 阈值")
    p.add_argument("--vocab_size", type=int, default=1024)
    p.add_argument("--context_length", type=int, default=128)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--rope_theta", type=float, default=10000.0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--special_tokens", type=str, nargs="*", default=["<|endoftext|>"], help="特殊 token，用于解析 endoftext id")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(
        args.checkpoint,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )

    from cs336_basics.Tokenizer import Tokenizer
    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path, specail_tokens=args.special_tokens)
    prompt_ids = tokenizer.encode(args.prompt)
    prompt_t = torch.tensor(prompt_ids, dtype=torch.long, device=device)

    # 解析 <|endoftext|> 的 id
    eot_id = None
    eot_bytes = "<|endoftext|>".encode("utf-8")
    for tid, b in tokenizer.vocab.items():
        if b == eot_bytes:
            eot_id = tid
            break

    out = decode(
        model,
        prompt_t,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        endoftext_token_id=eot_id,
        context_length=args.context_length,
    )
    out_ids = out[0].cpu().tolist()
    text = tokenizer.decode(out_ids)
    print(text)


if __name__ == "__main__":
    main()
