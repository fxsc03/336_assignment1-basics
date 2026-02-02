from hashlib import pbkdf2_hmac
import os
import collections
from typing import List, Tuple, Dict, Set
import re
import json
# from xxlimited import Null


def gpt2_bytes_to_unicode():
    # 建立一个“0-255 所有字节”到“Unicode 字符”的 1 对 1 映射表。
    # 先初始化可见的
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    now = 0 #当前指针
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256+now)
            now+=1
    # 把cs转换成unicode字符
    cs = [chr(tmp) for tmp  in cs]
    
    return dict(zip(bs, cs))

# print(list(gpt2_bytes_to_unicode()))

# 统计token序列中所有相邻unicode字符对的频率,返回一个字典
def get_stats(token_sequences: List[List[str]]) -> collections.Counter:
    mp = collections.Counter()
    for seq in token_sequences:
        for i in range(len(seq)-1):
            mp[(seq[i], seq[i+1])] += 1
    return mp

# tmp_seq = [
#     ['h', 'e', 'l', 'l', 'o'],
#     ['h', 'e', 'l', 'p']
# ]

# print(get_stats(tmp_seq))

def merge_pair_in_sequences(token_sequences: List[List[str]], pair:Tuple[str,str]) ->List[List[str]]:
    new_token = pair[0] + pair[1]
    new_token_sequences = []
    for seq in token_sequences:
        # for i in range(len(seq)):
        tmp_token_sequnences = []
        i = 0
        while i < len(seq):
            if(i< len(seq)-1 and pair==(seq[i],seq[i+1])):
                tmp_token_sequnences.append(new_token)
                i+=2
            else:
                tmp_token_sequnences.append(seq[i])
                i+=1
        new_token_sequences.append(tmp_token_sequnences)
    return new_token_sequences

# tmp_seq = [
#     ['h', 'e', 'l', 'l', 'o'],
#     ['h', 'e', 'l', 'p']
# ]    
# tmp_pair = ('h', 'e')

# print(merge_pair_in_sequences(tmp_seq, tmp_pair))

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int, # size表示我最终需要的参数有多少个
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    参数 (Args):
    input_path (str | os.PathLike): BPE 分词器训练数据的路径。
    vocab_size (int): 分词器词汇表中的条目总数（包含特殊 token）。
    special_tokens (list[str]): 要添加到分词器词汇表中的特殊 token 字符串列表。
    这些字符串永远不会被拆分成多个 token，而是始终保持为一个单独的 token。
    注意：如果这些特殊 token 出现在 input_path（训练语料）中，它们会被当作任何其他普通字符串一样处理（即在训练阶段不享有特殊待遇，只是最后会被强制加到词表中）。
    返回值 (Returns):
    tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: 返回一个包含 vocab 和 merges 的元组：
    vocab: 训练好的分词器词汇表。这是一个从 int（词汇表中的 token ID）到 bytes（token 的字节内容）的映射。
    merges: BPE 合并规则。列表中的每一项都是一个字节元组 (<token1>, <token2>)，代表 <token1> 和 <token2> 被合并在了一起。 合并规则是按照创建顺序排列的。
    """

    byte_to_unicode_map = gpt2_bytes_to_unicode();
    token_str_to_bytes = {v: bytes([k]) for k, v in byte_to_unicode_map.items()}

    #注意原始数据的unicode是字节流。因此需要用bytes转换
    unicode_to_byte_map = {}
    for k, v in byte_to_unicode_map.items():
        tmp_k = v
        tmp_v = bytes([k])
        unicode_to_byte_map[tmp_k] = tmp_v
    
    # 初始化vocab表
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    current_next_id: int =256

    # 用一个集合来高效检查特殊符号的字节表示是否已存在于词汇表中，用列表也能查重，但时间复杂度是O(n)，集合是O(1)
    existing_byte_values: Set[bytes] = set(vocab.values())

    # 添加特殊符号到词汇表
    for st_str in special_tokens:
        if len(vocab) >= vocab_size: # 如果词汇表满了，就不再添加
            break
        st_bytes = st_str.encode("utf-8") # 将特殊符号字符串转为字节串
        if st_bytes not in existing_byte_values: # 只有当这个字节串不在现有词汇中时才添加（避免重复，例如特殊符号 "a" 和基础字节 b'a'）
            vocab[current_next_id] = st_bytes # 将新的字节串添加到词汇表中
            existing_byte_values.add(st_bytes) # 记录这个新的字节值
            current_next_id += 1 # 更新下一个token ID


    try:
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        text = "" # 如果文件不存在，视为空文本处理

    # 使用正则表达式对text进行预分词
    raw_words: List[str] = re.findall(r'\s*\S+', text)

    unicode_sequences: List[List[str]] = []

    for word_str in raw_words:
        tmp_word:bytes = word_str.encode("utf-8")
        if not tmp_word:
            continue
        unicode_sequences.append([byte_to_unicode_map[val] for val in tmp_word])
    
    merges: List[Tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        if not unicode_sequences:
            break
        word_mp = get_stats(unicode_sequences)
        if not word_mp:
            break
        # 找到频率最高的unicode字符对
        best_pair: Tuple[str, str] = max(word_mp, key=lambda x: word_mp[x])

        # 这是我合并后的字符
        now_token_str: str = best_pair[0] + best_pair[1]
        p1_bytes = token_str_to_bytes[best_pair[0]]
        p2_bytes = token_str_to_bytes[best_pair[1]]
        now_token_bytes: bytes = p1_bytes + p2_bytes
        token_str_to_bytes[now_token_str] = now_token_bytes
        vocab[current_next_id] = now_token_bytes

        merges.append((p1_bytes, p2_bytes))

        unicode_sequences = merge_pair_in_sequences(unicode_sequences, (best_pair[0], best_pair[1]))
        current_next_id += 1


    # 保存vocab.json词表，merges.txt合并表
    with open("vocab.json", "w", encoding="utf-8") as f:
        vocab_dict = {token_id: token_bytes.decode("utf-8", errors="replace") 
                     for token_id, token_bytes in vocab.items()}
        json.dump(vocab_dict, f, ensure_ascii=False, indent=4)
    
    # 保存合并操作记录到文件
    with open("merges.txt", "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            # 将字节对转换为可读的字符串表示
            p1_str = p1.decode("utf-8", errors="replace")
            p2_str = p2.decode("utf-8", errors="replace")
            f.write(f"{p1_str} {p2_str}\n")


    return vocab, merges

