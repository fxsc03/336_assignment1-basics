"""
主训练脚本：将模型、优化器、数据加载、检查点与日志整合为完整训练流程。
支持命令行配置超参数、memmap 大数据集、检查点保存与恢复、控制台及可选 wandb 日志。
"""
import argparse
import os
import random
import sys
from pathlib import Path

# 支持直接运行 train.py（uv run train.py 或 python train.py）
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0,   str(_root))

import numpy as np
import torch

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamW import AdamW
from cs336_basics.dataloader import run_get_batch
from cs336_basics.cross_entropy import CrossEntropyLoss
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.lr_cosine_shedule import get_lr_cosine_schedule
from cs336_basics.gradient_clipping import GradientClipping


# 默认路径（可被命令行覆盖）
_DEFAULT_TRAIN_PATH = "/home/fxs/LLM1.30/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
_DEFAULT_VAL_PATH = "/home/fxs/LLM1.30/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"


def parse_args():
    p = argparse.ArgumentParser(description="Train Transformer LM with configurable hyperparameters.")
    # 数据
    p.add_argument("--train_data_path", type=str, default=_DEFAULT_TRAIN_PATH, help="训练集 token ids 文件路径（用于 np.memmap）")
    p.add_argument("--val_data_path", type=str, default=_DEFAULT_VAL_PATH, help="验证集 token ids 文件路径（用于 np.memmap）")
    p.add_argument("--memmap_dtype", type=str, default="uint16", choices=["uint16", "int32"], help="memmap 数组 dtype，省内存用 uint16")
    # 模型
    p.add_argument("--vocab_size", type=int, default=50257, help="词表大小")
    p.add_argument("--context_length", type=int, default=128, help="上下文长度（序列长度）")
    p.add_argument("--d_model", type=int, default=256, help="模型维度")
    p.add_argument("--num_layers", type=int, default=4, help="Transformer 层数")
    p.add_argument("--num_heads", type=int, default=4, help="注意力头数")
    p.add_argument("--d_ff", type=int, default=512, help="FFN 中间维度")
    p.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE 的 theta 参数")
    # 优化器与学习率
    p.add_argument("--lr_max", type=float, default=1e-3, help="最大学习率（cosine 峰值）")
    p.add_argument("--lr_min", type=float, default=1e-5, help="最小学习率（cosine 谷值）")
    p.add_argument("--warmup_iters", type=int, default=1000, help="线性 warmup 步数")
    p.add_argument("--weight_decay", type=float, default=0.01, help="AdamW 权重衰减")
    # 训练
    p.add_argument("--batch_size", type=int, default=32, help="每步 batch size")
    p.add_argument("--max_iters", type=int, default=10000, help="最大训练步数")
    p.add_argument("--grad_clip_max_norm", type=float, default=1.0, help="梯度 L2 范数上限")
    # 检查点与日志
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="检查点保存目录（用户指定路径）")
    p.add_argument("--resume", type=str, default=None, help="从中恢复的检查点路径（可选）")
    p.add_argument("--checkpoint_interval", type=int, default=5000, help="每 N 步保存一次检查点")
    p.add_argument("--log_interval", type=int, default=100, help="每 N 步打印/记录一次训练指标")
    p.add_argument("--eval_interval", type=int, default=500, help="每 N 步在验证集上评估一次")
    p.add_argument("--eval_iters", type=int, default=100, help="验证时使用的批次数（用于估计 val loss）")
    # 设备与随机种子
    p.add_argument("--device", type=str, default=None, help="设备，如 cuda:0 或 cpu，默认自动选择")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    # Weights & Biases（可选）
    p.add_argument("--wandb", action="store_true", help="是否启用 Weights and Biases 日志")
    p.add_argument("--wandb_project", type=str, default="cs336-basics", help="W&B 项目名")
    p.add_argument("--wandb_run_name", type=str, default=None, help="W&B 运行名称（可选）")
    # 文本数据（.txt）需提供 tokenizer，用于将文本转为 token id
    p.add_argument("--tokenizer_vocab_path", type=str, default=None, help="词表 JSON 路径（.txt 数据时必填）")
    p.add_argument("--tokenizer_merges_path", type=str, default=None, help="BPE merges 文件路径（.txt 数据时必填）")
    p.add_argument("--tokenizer_special_tokens", type=str, default=None, nargs="*", help="特殊 token 列表（可选）")
    return p.parse_args()


def _is_text_path(path: str) -> bool:
    return path.endswith(".txt")


def load_memmap_data(path: str, dtype: str) -> np.ndarray:
    """使用 np.memmap 内存高效加载一维 token id 数组（二进制文件）。"""
    dtype_np = np.dtype(dtype)
    try:
        return np.memmap(path, mode="r", dtype=dtype_np)
    except ValueError as e:
        if "multiple of the data-type size" in str(e):
            raise ValueError(
                f"文件 {path} 不是合法的二进制 token id 数组（或 dtype 与文件不匹配）。"
                "若为 .txt 文本，请提供 --tokenizer_vocab_path 和 --tokenizer_merges_path。"
            ) from e
        raise


def load_text_data(path: str, tokenizer, dtype: str, log_progress: bool = True) -> np.ndarray:
    """从 .txt 文件读取文本并用 tokenizer 编码为一维 token id 数组。"""
    dtype_np = np.dtype(dtype)
    chunk_size = 1 << 20  # 1MB 一块，避免单次读入过大
    ids_list = []
    total_mb = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            ids_list.extend(tokenizer.encode(chunk))
            total_mb += 1
            if log_progress and total_mb % 50 == 0:
                print(f"  Tokenizing... {total_mb} MB read, {len(ids_list)} tokens", flush=True)
    if log_progress:
        print(f"  Done: {len(ids_list)} tokens from {path}", flush=True)
    return np.array(ids_list, dtype=dtype_np)


def load_data(path: str, dtype: str, tokenizer=None) -> np.ndarray:
    """根据路径类型加载数据：.txt 用 tokenizer 编码，否则用 memmap 读二进制。"""
    if _is_text_path(path):
        if tokenizer is None:
            raise ValueError(
                f"数据路径为 .txt 文件 ({path})，需提供 tokenizer。"
                "请传入 --tokenizer_vocab_path 和 --tokenizer_merges_path。"
            )
        return load_text_data(path, tokenizer, dtype)
    return load_memmap_data(path, dtype)


def estimate_loss(
    model: torch.nn.Module,
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int,
) -> float:
    """在 data 上取 num_batches 个 batch 估计平均 loss。"""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = run_get_batch(data, batch_size, context_length, device)
            logits = model(x)
            # (B, T, V) -> (B*T, V) 与 (B*T,)
            loss = CrossEntropyLoss(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
            total_loss += loss.item()
    model.train()
    return total_loss / num_batches


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 若为 .txt 则需 tokenizer
    tokenizer = None
    if _is_text_path(args.train_data_path) or _is_text_path(args.val_data_path):
        if not args.tokenizer_vocab_path or not args.tokenizer_merges_path:
            raise ValueError(
                "数据路径为 .txt 时需提供 --tokenizer_vocab_path 和 --tokenizer_merges_path。"
                "请先在语料上训练 BPE 并保存 vocab/merges，或改用预处理的二进制 token id 文件。"
            )
        from cs336_basics.Tokenizer import Tokenizer
        special = args.tokenizer_special_tokens or None
        tokenizer = Tokenizer.from_files(
            args.tokenizer_vocab_path,
            args.tokenizer_merges_path,
            specail_tokens=special,
        )

    # 加载数据：.txt 用 tokenizer 编码，否则 memmap 读二进制
    train_data = load_data(args.train_data_path, args.memmap_dtype, tokenizer)
    val_data = load_data(args.val_data_path, args.memmap_dtype, tokenizer)
    n_train = len(train_data)
    n_val = len(val_data)
    if n_train < args.context_length + 1 or n_val < args.context_length + 1:
        raise ValueError(
            f"数据长度必须 > context_length+1: train={n_train}, val={n_val}, context_length={args.context_length}"
        )

    # 模型与优化器
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        weight_decay=args.weight_decay,
    )
    start_iter = 0
    if args.resume:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"从检查点恢复: {args.resume}, iteration={start_iter}")

    # 可选 wandb
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
            )
        except ImportError:
            print("未安装 wandb，忽略 --wandb")
            args.wandb = False

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"开始训练: max_iters={args.max_iters}, device={device}, train_samples≈{n_train}, val_samples≈{n_val}")

    for it in range(start_iter, args.max_iters):
        # 学习率（cosine + warmup）
        lr = get_lr_cosine_schedule(
            it,
            args.lr_max,
            args.lr_min,
            args.warmup_iters,
            args.max_iters,
        )
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = run_get_batch(train_data, args.batch_size, args.context_length, device)
        logits = model(x)
        loss = CrossEntropyLoss(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        GradientClipping(model.parameters(), args.grad_clip_max_norm)
        optimizer.step()

        # 周期性日志（控制台 + 可选 wandb）
        if (it + 1) % args.log_interval == 0:
            log_msg = f"iter {it+1}/{args.max_iters}  train_loss={loss.item():.4f}  lr={lr:.2e}"
            print(log_msg)
            if args.wandb:
                try:
                    import wandb
                    wandb.log({"train_loss": loss.item(), "lr": lr, "iter": it + 1}, step=it + 1)
                except Exception:
                    pass

        # 周期性验证
        if (it + 1) % args.eval_interval == 0:
            val_loss = estimate_loss(
                model, val_data, args.batch_size, args.context_length,
                device, args.eval_iters,
            )
            print(f"iter {it+1}  val_loss={val_loss:.4f}")
            if args.wandb:
                try:
                    import wandb
                    wandb.log({"val_loss": val_loss, "iter": it + 1}, step=it + 1)
                except Exception:
                    pass

        # 周期性保存检查点到用户指定目录
        if (it + 1) % args.checkpoint_interval == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_iter_{it+1}.pt")
            save_checkpoint(model, optimizer, it + 1, ckpt_path)
            print(f"已保存检查点: {ckpt_path}")

    # 训练结束保存最终检查点
    final_path = os.path.join(args.checkpoint_dir, "checkpoint_final.pt")
    save_checkpoint(model, optimizer, args.max_iters, final_path)
    print(f"训练完成，最终检查点: {final_path}")
    if args.wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
