from curses import raw, resetty
from re import S, split
import numpy as np
from typing import Dict, List, Set, Tuple, Iterable, Iterator
import regex as re
import json
import tiktoken


# 预编译正则表达式
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


# 预分词切割text
def pre_tokenization(text, special_tokens) -> List[str]:

    result: List[str] = []

    # 没有special_tokens直接切分然后返回
    if not special_tokens:
        for m in PAT.finditer(text):
            result.append(m.group())
        return result
    
    # 排序防止special_token被切错了
    sorted_special_tokens = sorted(special_tokens, key = len, reverse = True)
    parttern_string = "(" + "|".join(re.escape(t) for t in sorted_special_tokens) + ")"
    chunks = re.split(parttern_string, text)

    # 注意现在的chunks是整个文本,chunk是句子

    for chunk in chunks:
        if not chunk:
            continue
        if chunk in special_tokens:
            result.append(chunk)
            continue
        split_chunk = PAT.finditer(chunk)
        for s in split_chunk:
            result.append(s.group())
    
    return result

# text = "Hello, world! I'm ready."
# special_tokens = []

# output = pre_tokenization(text, special_tokens)
# print(output)

    


class Tokenizer:

    def __init__(self, vocab, merges, special_tokens = None):
        # id ->bytes
        self.vocab = vocab.copy()
        self.merges = merges 
        self.special_tokens: List[str] = special_tokens or []
        
        # 把special_tokens中的token加到vocab中
        next_id = 0
        if self.vocab:       
            next_id = max(self.vocab.keys()) + 1
        if special_tokens :
            for st in special_tokens:
                st_bytes = st.encode("utf-8")
                if st_bytes not in self.vocab.values():
                    self.vocab[next_id] = st_bytes
                    next_id += 1
        
        # 方便encode的时候查表
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}

        self.bpe_ranks = {pair: i for i, pair in enumerate(self.merges)}
        
    
    @classmethod
    def from_files(cls, vocab_filepath, merge_filepath, specail_tokens: List[str] | None = None):


        vocab: dict[int, bytes] = []
        with open(vocab_filepath, "r", encoding = "utf-8")as f:
            raw_vocab = json.load(f)
        
        for k, v in raw_vocab.items():
            new_key = int(k)

            # v可能是str，可能是num，做不同转换
            if isinstance(v, str):
                new_value = v.encode("utf-8")
            else:
                new_value = bytes[v]
            vocab[new_key] = new_value
        
        merges = []
        with open(merge_filepath, "r", encoding = "utf-8") as f:
            for line in f:
                line = line.strip()
                # 跳过空行或注释
                if not line or line.startswith("#"):
                    continue
                
                parts = line.split()

                if len(parts) == 2:
                    # 解析 "117,118" -> [117, 118] -> b'uv'
                    p0 = bytes(map(int, parts[0].split(',')))
                    p1 = bytes(map(int, parts[1].split(',')))
                    merges.append((p0, p1))
            
        return cls(vocab, merges, specail_tokens)



    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        
        # 现在里面有单词和special_token
        chunks = pre_tokenization(text, self.special_tokens)

        final_ids:List[int] = []
        for chunk in chunks:
            # 先处理特殊字符
            if chunk in self.special_tokens:
                chunk_bytes = chunk.encode("utf-8")
                if chunk_bytes in self.bytes_to_id:
                    final_ids.append(self.bytes_to_id[chunk_bytes])
                else:
                    raise ValueError(f"特殊符号 '{chunk}' 被识别到了，但在词表中找不到对应的 ID\n")
                continue
            
            word_bytes = chunk.encode("utf-8")

            # 把每一个字母拿出来用于查merge表合并
            parts = [bytes([b]) for b in word_bytes]
            
            while True:
                # 每次根据vocab表查一对合
                if len(parts) < 2:
                    break
                
                # 先遍历找出能够合并的临对，也就是bpe_ranks中最小的
                pairs = [(parts[i], parts[i+1]) for i in range(len(parts)-1)]
                best_pairs = min(pairs, key =lambda pair: self.bpe_ranks.get(pair, float('inf')) )

                if best_pairs not in self.bpe_ranks:
                    break

                # 开始merge
                new_parts = []
                i = 0
                p0, p1 = best_pairs
                target = p0 + p1
                while i < len(parts):
                    if i< len(parts) -1 and parts[i] == p0 and parts[i+1] == p1:
                        new_parts.append(target)
                        i += 2
                    else:
                        new_parts.append(parts[i])
                        i += 1
                parts = new_parts

            # 更新final_list 
            for part in parts:
                if part in self.bytes_to_id:
                    final_ids.append(self.bytes_to_id[part])

        return final_ids


    # streaming流式处理
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    # bytes->unicode->str
    def decode(self, ids: List[int]) -> str:
        byte_data = b"".join(self.vocab[i] for i in ids)
        text = byte_data.decode("utf-8", errors = 'replace')
        return text
        

# # ==========================================
# # 简单的手动测试脚本
# # ==========================================
# if __name__ == "__main__":
#     # 1. 伪造数据
#     # 我们手动构造一个 vocab，假装这些是训练好的
#     # 注意：这里特意把中文 "你" (b'\xe4\xbd\xa0') 拆成了三个碎片，测试拼接能力
#     fake_vocab = {
#         0: b'H',
#         1: b'el',
#         2: b'lo',
#         3: b', ',
#         4: b'World',
#         5: b'!',
#         6: b'\xe4', # '你' 的第1个字节
#         7: b'\xbd', # '你' 的第2个字节
#         8: b'\xa0'  # '你' 的第3个字节
#     }
    
#     # decode 不需要 merges，给个空列表就行
#     fake_merges = [] 
    
#     # 2. 实例化你的 Tokenizer
#     # 注意：这里会触发你的 __init__，确保你之前的 __init__ 代码是好的
#     tokenizer = Tokenizer(fake_vocab, fake_merges, special_tokens=None)
    
#     # 3. 准备测试 ID 序列
#     # 对应: H + el + lo + , + World + ! + (你)
#     test_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    
#     # 4. 运行解码
#     print(f"输入 ID: {test_ids}")
#     try:
#         decoded_text = tokenizer.decode(test_ids)
#         print(f"解码结果: {decoded_text}")
        
#         # 5. 验证正确性
#         expected_text = "Hello, World!你"
#         if decoded_text == expected_text:
#             print("✅ 测试通过！")
#         else:
#             print(f"❌ 测试失败。\n预期: {expected_text}\n实际: {decoded_text}")
            
#     except Exception as e:
#         print(f"❌ 运行报错: {e}")


# ==========================================
# 深度逻辑测试脚本
# ==========================================
if __name__ == "__main__":
    print("🚀 开始测试 Encode 模块...")

    # 1. 构造一个微型词表
    # 包含了基础字母、部分合并词、以及特殊符号
    fake_vocab = {
        # --- 基础字节 ---
        0: b'u', 1: b'n', 2: b'i', 3: b'g', 4: b'h', 5: b't', 
        6: b'a', 7: b'b',
        # --- BPE 合并产生的词 ---
        8: b'un',   # u + n
        9: b'ni',   # n + i (用来测试优先级的干扰项)
        10: b'uni', # un + i
        11: b'gh',  # g + h
        12: b'ght', # gh + t
        # --- 特殊符号 ---
        13: b'<|END|>'
    }

    # 2. 构造合并规则 (注意顺序！下标越小优先级越高)
    # 逻辑陷阱：我们同时有 (u, n) 和 (n, i)。
    # 对于单词 "uni"：
    # - 如果先合并 (u, n)，变成 un, i -> 再合并 (un, i) -> uni (正确路径)
    # - 如果先合并 (n, i)，变成 u, ni -> 无法合并成 uni (因为没有 u+ni 的规则)
    fake_merges = [
        (b'u', b'n'),   # Rank 0 (最高优先级)
        (b'g', b'h'),   # Rank 1
        (b'un', b'i'),  # Rank 2
        (b'gh', b't'),  # Rank 3
        (b'n', b'i'),   # Rank 4 (优先级低，陷阱！)
    ]
    
    special_tokens = ["<|END|>"]

    # 3. 初始化
    tokenizer = Tokenizer(fake_vocab, fake_merges, special_tokens)
    
    # 4. 测试案例
    # 目标文本: "unight<|END|>"
    # 预期拆解:
    #   "unight" -> b'u', b'n', b'i', b'g', b'h', b't'
    #   Step 1: (u, n) 合并 -> [un, i, g, h, t]
    #   Step 2: (g, h) 合并 -> [un, i, gh, t]
    #   Step 3: (un, i) 合并 -> [uni, gh, t]
    #   Step 4: (gh, t) 合并 -> [uni, ght]
    #   Step 5: <|END|> 直接查表
    # 最终 ID: [10 (uni), 12 (ght), 13 (<|END|>)]
    text = "unight<|END|>"
    
    try:
        print(f"\n测试文本: '{text}'")
        ids = tokenizer.encode(text)
        print(f"Encode 结果: {ids}")
        
        # 验证 Encode
        expected_ids = [10, 12, 13]
        if ids == expected_ids:
            print("✅ Encode 逻辑正确！(优先级处理完美)")
        else:
            print(f"❌ Encode 失败。\n预期: {expected_ids}\n实际: {ids}")
            # 如果你输出了 [0, 9, ... ] 说明优先级搞错了，先合并了 ni
            
        # 验证 Decode (Round Trip)
        decoded_text = tokenizer.decode(ids)
        print(f"Decode 结果: '{decoded_text}'")
        
        if decoded_text == text:
            print("✅ Decode 还原无损！")
        else:
            print(f"❌ Decode 还原失败。\n预期: '{text}'\n实际: '{decoded_text}'")

    except Exception as e:
        import traceback
        print(f"❌ 程序崩溃: {e}")
        traceback.print_exc()