fxs@node3 ~/L/a/cs336_basics (main)> head -n 10000 /home/fxs/LLM1.30/assignmen
t1-basics/data/TinyStoriesV2-GPT4-train.txt > /tmp/small.txt
                                        uv run train_bpe_tokenizer.py --input 
/tmp/small.txt --vocab_size 1024 --output_dir ./tokenizer_test


cd /home/fxs/LLM1.30/assignment1-basics/cs336_basics

# 训练集
uv run pretokenize.py \
  --txt_path /home/fxs/LLM1.30/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt \
  --bin_path ./data/train_tokens.bin \
  --vocab_path ./tokenizer_test/vocab.json \
  --merges_path ./tokenizer_test/merges.txt

# 验证集
uv run pretokenize.py \
  --txt_path /home/fxs/LLM1.30/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt \
  --bin_path ./data/val_tokens.bin \
  --vocab_path ./tokenizer_test/vocab.json \
  --merges_path ./tokenizer_test/merges.txt




  uv run train.py \
  --train_data_path ./data/train_tokens.bin \
  --val_data_path ./data/val_tokens.bin \
  --vocab_size 1024 \
  --max_iters 500 \
  --batch_size 16 \
  --context_length 64


