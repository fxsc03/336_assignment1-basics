import os
from collections import Counter
from collections import defaultdict
import regex as re
import multiprocessing
# from .pretokenization_example import find_chunk_boundaries

import time


# 预编译正则表达式
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def pre_tokenization(chunk, special_tokens)->list[str]:
    if not special_tokens:
        # 如果没有特殊 token，直接正则切词，不要用 re.split
        return [m.group() for m in PAT.finditer(chunk)]
    
    sorted_special = sorted(special_tokens, key=len, reverse=True)
    special_pat = "(" + "|".join(re.escape(t) for t in sorted_special) + ")"
    parts = re.split(special_pat, chunk)
    
    result = []
    for part in parts:
        if not part: continue
        if part in special_tokens:
            result.append(part)
        else:
            matches = PAT.finditer(part)
            for m in matches:
                result.append(m.group())
    return result


        

# --------------------2.6 Encoding & Decoding-----------------------# 
import json

class Tokenizer:

    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab.copy() # 后续可能把special_tokens加入
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.byte_to_id = {v: k for k, v in self.vocab.items()}
        self.bpe_ranks = {pair: i for i, pair in enumerate(self.merges)}  # 查找表，否则直接merge遍历单词太慢


    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):

        with open(vocab_filepath, "r", encoding = 'utf-8')as f:
            raw_vocab = json.load(f)
            # 转成 int: bytes
            vocab = {int(k): v.encode('utf-8') if isinstance(v, str) else bytes([v]) for k, v in raw_vocab.items()}
            
        merges = []
        with open(merges_filepath, "r", encoding = "utf-8") as f:
            for line in f:
                line = line.strip()
                parts = line.split()
                if len(parts) == 2:
                    p0 = bytes(map(int, parts[0].split(',')))
                    p1 = bytes(map(int, parts[1].split(',')))
                    merges.append((p0, p1)) 
        
        return cls(vocab, merges, special_tokens)



    def decode(self, ids):
        # 逻辑简单: 翻译，串起来，decode
        byte_data = b"".join(self.vocab[id] for id in ids)
        return byte_data.decode('utf-8', errors = 'replace')

    def encode(self, text):
        if not text:
            return []
        
        # 预分词
        words = pre_tokenization(text, self.special_tokens)
        

        final_ids = []

        for word in words:
            # 整个单词就是一个Token，则直接查
            word_bytes = word.encode('utf-8')
            if word_bytes in self.byte_to_id:
                final_ids.append(self.byte_to_id[word_bytes])
                continue
            
            # 拆成字节
            tokens = [bytes([b]) for b in word_bytes]
            

            while len(tokens) > 1:
                # 找出rank 最小的
                pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]

                best_pair = min(pairs, key=lambda p: self.bpe_ranks.get(p, float('inf')))
                
                if best_pair not in self.bpe_ranks:
                    break

                new_tokens = []
                i = 0
                p0, p1 = best_pair
                target = p0 + p1
                while i < len(tokens):
                    if i < len(tokens) - 1 and tokens[i] == p0 and tokens[i+1] == p1:
                        new_tokens.append(target)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens

            # 将最终合并出的 tokens 转为 ID
            for t in tokens:
                final_ids.append(self.byte_to_id[t])

        return final_ids


    def encode_iterable(self, iterable):
        for text in iterable:
            yield from self.encode(text)