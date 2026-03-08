"""
etok — Entropy Tokenizer
========================
Pure-Python BPE tokenizer with novel algorithmic features.
Vocab files are compatible with the etok C binary.

Quick start:
    from etok import Tokenizer, magic_split, rotate_compare

    tok = Tokenizer(vocab_size=4096)
    tok.train(open("corpus.txt").read())
    tok.save("vocab.json")

    ids  = tok.encode("hello world")
    text = tok.decode(ids)

    tok2 = Tokenizer.load("vocab.json")
"""

from .tokenizer import (
    Tokenizer,
    magic_split,
    rotate_compare,
    common_string,
    char_entropy,
    token_entropy,
)

__version__ = "0.1.0"
__all__ = [
    "Tokenizer",
    "magic_split",
    "rotate_compare",
    "common_string",
    "char_entropy",
    "token_entropy",
]
