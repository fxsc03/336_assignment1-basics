"""
在语料上训练 BPE tokenizer，并保存 vocab.json 和 merges.txt（格式与 Tokenizer.from_files 兼容）。
用法示例：
  uv run train_bpe_tokenizer.py --input /path/to/train.txt --output_dir ./tokenizer
  或
  python -m cs336_basics.train_bpe_tokenizer --input data/TinyStoriesV2-GPT4-train.txt --output_dir ./tokenizer
"""
import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from cs336_basics.BPE_Tokenizer_Training import run_train_bpe


def main():
    p = argparse.ArgumentParser(description="训练 BPE 并保存 vocab.json、merges.txt")
    p.add_argument("--input", type=str, default="/home/fxs/LLM1.30/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt", help="训练语料路径（.txt）")
    p.add_argument("--output_dir", type=str, default="./tokenizer", help="输出目录，将生成 vocab.json 和 merges.txt")
    p.add_argument("--vocab_size", type=int, default=8192, help="词表大小（含特殊 token）")
    p.add_argument("--special_tokens", type=str, nargs="*", default=["<|endoftext|>"], help="特殊 token 列表")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.txt"

    print(f"训练 BPE: input={args.input}, vocab_size={args.vocab_size}")
    vocab, merges = run_train_bpe(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens or [],
    )

    # vocab.json：key 为字符串 id，value 为 token 的 UTF-8 解码字符串
    vocab_dict = {str(k): v.decode("utf-8", errors="replace") for k, v in vocab.items()}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

    # merges.txt：每行 "b1,b2,...,bm c1,c2,...,cn"（字节值，与 Tokenizer.from_files 兼容）
    with open(merges_path, "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            part1 = ",".join(str(b) for b in p1)
            part2 = ",".join(str(b) for b in p2)
            f.write(f"{part1} {part2}\n")

    print(f"已保存: {vocab_path}, {merges_path}")
    print(f"词表大小: {len(vocab)}, merges 数量: {len(merges)}")


if __name__ == "__main__":
    main()
