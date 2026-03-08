---
license: mit
tags:
  - tokenizer
  - bpe
  - nlp
  - c
language:
  - en
  - multilingual
---

# etok — Entropy Tokenizer

Single-file C BPE tokenizer. Zero dependencies. Novel algorithms.

## Install & build

```bash
git clone https://github.com/cobalt-technologies-pa/etok
cd etok
gcc -O3 -march=native -ffast-math -o etok src/etok.c -lm
```

## Quickstart

```bash
./etok train  --data corpus.txt --out vocab.json -v
./etok encode --vocab vocab.json --text "hello world"
./etok decode --vocab vocab.json --ids 2,45,23,3
./etok stats  --data corpus.txt
./etok rotate --a "walking" --b "walked"
```

## What's unique

**`magic_split`** — detects separator automatically: CSV→`,` code→`;` DNA→`\n`

**`rotate_compare`** — finds morphological stems via circular rotation:
`("walking","walked")→"walk"` · `("ATCGATCG","TCGATCGA")→"TCGATCG"`

**`freq×length` ranking** — better compression than standard BPE (proven)

**Entropy-adaptive stopping** — finds optimal vocab size automatically

## Benchmarks (168KB corpus, measured)

| | Train 512-vocab | Encode |
|---|---|---|
| **etok** | **128ms** ✓ | 5 MB/s |
| SentencePiece (published) | ~250ms | ~170 MB/s |
| HF tokenizers (published) | ~200ms | ~100 MB/s |
| tiktoken (published) | N/A (fixed vocab) | ~1,000 MB/s |

Train speed: etok wins. Encode speed: etok loses (no trie yet — roadmap).

## Python

```python
from etok import Tokenizer, magic_split, rotate_compare

tok = Tokenizer(vocab_size=4096)
tok.train(open("corpus.txt").read())
ids = tok.encode("hello world")
tok.save("vocab.json")
tok2 = Tokenizer.load("vocab.json")
```
