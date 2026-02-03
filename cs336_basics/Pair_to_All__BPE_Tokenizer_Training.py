import concurrent
from curses import pair_content
import os
import collections
from typing import List, Tuple, Dict, Set
from collections import Counter  # ✅ 必须是从 collections 导入
import json
# import regex    
from collections import defaultdict
import pickle


def merge_token_sequence(token_seq: Tuple, best_pair: Tuple, new_token: bytes) -> Tuple:
    new_token_seq = []
    n = len(token_seq)
    i = 0
    while i < n:
        if i < n-1 and (token_seq[i], token_seq[i+1]) == best_pair:
            new_token_seq.append(new_token)
            i += 2
        else:
            new_token_seq.append(token_seq[i])
            i += 1
    return tuple(new_token_seq)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for it in special_tokens:
        it_unicode = it.encode("utf-8")
        vocab[next_id] = it_unicode
        next_id += 1


   # ==========================
    # 2. 读取与预分词
    # ==========================
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    # 修复点：先检查 special_tokens 是否为空
    if special_tokens:
        # 只有有特殊符号时，才去切分
        special_pattern = "(" + "|".join(map(regex.escape, special_tokens)) + ")"
        chunks = regex.split(special_pattern, text)
    else:
        # 如果没有特殊符号，整段文本就是一个大块
        chunks = [text]

    vocab_counts = Counter()

    for chunk in chunks:
        # 如果是特殊符号，直接跳过（不参与BPE合并）
        if chunk in special_tokens:
            continue
            
        # 过滤掉空的 chunk (regex.split 有时会产生空字符串)
        if not chunk:
            continue

        raw_words = regex.findall(PAT, chunk)
        
        for word in raw_words:
            word_bytes = word.encode("utf-8")
            tokens = tuple(bytes([b]) for b in word_bytes)
            vocab_counts[tokens] += 1


    merges: list[Tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        # 先从头统计一遍pair_counts的最新情况
        pair_counts = defaultdict(int)
        for tokens, count in vocab_counts.items():
            for i in range(len(tokens) - 1):
                now_token = (tokens[i], tokens[i+1])
                pair_counts[now_token] += count
        if not pair_counts:
            break
        
        # 找到当前的pair
        max_pair = max(pair_counts, key = lambda p: (pair_counts[p], p))

        # 更新vocab表,merges表
        new_token = max_pair[0] + max_pair[1]
        vocab[next_id] = new_token
        next_id += 1
        merges.append(max_pair)

        #执行实际的合并(token替换)
        new_vocab_counts = Counter()
        
        for tokens, count in vocab_counts.items():
            if max_pair[0] in tokens and max_pair[1] in tokens:
                new_token_seq = merge_token_sequence(tokens, max_pair, new_token)
                new_vocab_counts[new_token_seq] = count
            else:
                new_vocab_counts[tokens] = count
        
        vocab_counts = new_vocab_counts

  # 保存词汇表到文件 (使用 pickle)
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    
    # 保存合并操作记录到文件 (使用 pickle)
    with open("merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    return vocab, merges # 返回最终的词汇表和合并记录
        

# ... (你的函数定义) ...

if __name__ == "__main__":
    import os

    # === 1. 创建测试数据 ===
    # 我们构造一个简单的语料，故意让 'u' 和 'g' 经常在一起出现
    # 预期：BPE 应该最早发现 ('u', 'g') 并合并它们
    test_content = "hug " * 10 + "pug " * 5 + "bus " * 5
    test_file_path = "debug_corpus.txt"

    print(f">>> 生成测试文件: {test_file_path}")
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write(test_content)

    # === 2. 运行 BPE ===
    # 我们设置 vocab_size = 256 (基础) + 3 (新词) = 259
    # 看看它会学出什么新词
    print(">>> 开始训练...")
    try:
        vocab, merges = run_train_bpe(
            input_path=test_file_path,
            vocab_size=256 + 3,
            special_tokens=["<|endoftext|>"]
            )

        # === 3. 打印结果 ===
        print("\n=== 训练结果 ===")
        print(f"生成的 Merges ({len(merges)}个):")
        for p1, p2 in merges:
            print(f"  Merge: {p1} + {p2}")
            
        # 验证一下有没有合并 'u' 和 'g'
        # 注意：bytes 显示为 b'u', b'g'
        if (b'u', b'g') in merges:
            print("\n✅ 成功检测到高频对 (u, g)！")
        else:
            print("\n❌ 未检测到 (u, g)，请检查逻辑。")

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()

    # === 4. (可选) 清理文件 ===
    os.remove(test_file_path)
    print(f"\n已删除测试文件: {test_file_path}")