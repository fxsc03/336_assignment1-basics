import numpy as np
from typing import Dict, List, Set, Tuple, Iterable, Iterator
import json
import tiktoken


class implementing_bpe_tokenizer:


    def __init__(self, vocab, merges, special_tokens = None):
        self.vocab = vocab
        self.merges = merges
        f
    
    @classmethod
    def from_files(cls, vocab_filepath, merge_filepath, specail_tokens: List[str] | None = None):


    def encode(self, text: str) -> List[int]:

    def encode_iterable(self, iterable, Iterable[str]) -> Iterator[int]:
        