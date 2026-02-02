import json
import time

from .adapters import run_train_bpe
from .common import FIXTURES_PATH, gpt2_bytes_to_unicode


def test_train_bpe_speed():
    """
    Ensure that BPE training is relatively efficient by measuring training
    time on this small dataset and throwing an error if it takes more than 1.5 seconds.
    This is a pretty generous upper-bound, it takes 0.38 seconds with the
    reference implementation on my laptop. In contrast, the toy implementation
    takes around 3 seconds.
    """
    input_path = FIXTURES_PATH / "corpus.en"
    start_time = time.time()
    _, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    assert end_time - start_time < 1.5


def test_train_bpe():
    input_path = FIXTURES_PATH / "corpus.en"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

    # Path to the reference tokenizer vocab and merges
    reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"
    reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"

    # Compare the learned merges to the expected output merges
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(reference_merges_path, encoding="utf-8") as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]
    assert merges == reference_merges

    # Compare the vocab to the expected output vocab
    with open(reference_vocab_path, encoding="utf-8") as f:
        gpt2_reference_vocab = json.load(f)
        reference_vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }
    # Rather than checking that the vocabs exactly match (since they could
    # have been constructed differently, we'll make sure that the vocab keys and values match)
    assert set(vocab.keys()) == set(reference_vocab.keys())
    assert set(vocab.values()) == set(reference_vocab.values())


def test_train_bpe_special_tokens(snapshot):
    """
    Ensure that the special tokens are added to the vocabulary and not
    merged with other tokens.
    """
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )

    # Check that the special token is not in the vocab
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    for word_bytes in vocabs_without_specials:
        assert b"<|" not in word_bytes

    snapshot.assert_match(
        {
            "vocab_keys": set(vocab.keys()),
            "vocab_values": set(vocab.values()),
            "merges": merges,
        },
    )

# done
# 做一个小数据集的验证
def test_toy_example(tmp_path):
    """
    这是一个自定义的极简测试。
    目的是验证：
    1. 代码能不能跑通（不报错）。
    2. 能不能正确合并最高频的字符对。
    """
    # 1. 创建一个临时的小文件
    # 这里的文本设计很有讲究："ab" 连续出现了 4 次，是频率最高的
    # 预期 BPE 第一步一定会合并 'a' 和 'b'
    toy_text = "ab ab ab ab c d e f"
    
    p = tmp_path / "toy_corpus.txt"
    p.write_text(toy_text, encoding="utf-8")

    # 2. 运行你的 BPE 函数
    # vocab_size 设置为 258：
    # 256 (基础字符) + 2 (我们需要合并两次看看效果)
    print("\n>>> 开始运行 Toy Test...")
    vocab, merges = run_train_bpe(
        input_path=p,
        vocab_size=256 + 2, 
        special_tokens=[],
    )

    # 3. 打印结果帮你看清发生了什么 (配合 -s 参数)
    print("\n>>> [Debug] Merges 列表:", merges)
    
    # 4. 简单的断言验证
    # 验证是否产生了合并
    assert len(merges) > 0
    
    # 验证第一个合并的是否是 'a' 和 'b' (对应字节)
    # 注意：ord('a')=97, ord('b')=98
    first_merge = merges[0]
    expected_merge = (b'a', b'b')
    
    if first_merge == expected_merge:
        print(">>> [Pass] 成功识别出最高频组合 (a, b)！")
    else:
        print(f">>> [Fail] 预期合并 (a, b)，实际合并了 {first_merge}")
    
    assert first_merge == expected_merge