"""
把 .txt 语料用已有 tokenizer 预编码为 .bin（token id 数组），之后训练时用 --train_data_path xxx.bin 即可，
无需再传 tokenizer，加载会很快（memmap）。

用法（在 cs336_basics 目录下）：
  uv run pretokenize.py --txt_path /path/to/train.txt --bin_path /path/to/train.bin \\
    --vocab_path ./tokenizer_test/vocab.json --merges_path ./tokenizer_test/merges.txt
"""
import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
from cs336_basics.Tokenizer import Tokenizer


def main():
    p = argparse.ArgumentParser(description="将 .txt 用 tokenizer 编码并保存为 .bin")
    p.add_argument("--txt_path", type=str, required=True, help="输入 .txt 路径")
    p.add_argument("--bin_path", type=str, required=True, help="输出 .bin 路径（token id 数组，uint16）")
    p.add_argument("--vocab_path", type=str, required=True, help="vocab.json 路径")
    p.add_argument("--merges_path", type=str, required=True, help="merges.txt 路径")
    p.add_argument("--chunk_mb", type=int, default=10, help="每块读取的 MB 数，用于显示进度")
    args = p.parse_args()

    print("加载 tokenizer...")
    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path, specail_tokens=None)
    chunk_size = args.chunk_mb * (1 << 20)
    ids_list = []
    total_mb = 0
    print(f"编码 {args.txt_path} -> {args.bin_path} ...")
    with open(args.txt_path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            ids_list.extend(tokenizer.encode(chunk))
            total_mb += args.chunk_mb
            print(f"  {total_mb} MB read, {len(ids_list)} tokens", flush=True)
    arr = np.array(ids_list, dtype=np.uint16)
    arr.tofile(args.bin_path)
    print(f"已写入 {args.bin_path}，共 {len(arr)} tokens")


if __name__ == "__main__":
    main()
