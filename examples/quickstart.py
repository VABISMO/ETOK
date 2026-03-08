#!/usr/bin/env python3
"""
examples/quickstart.py — etok in 30 lines
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from etok import Tokenizer, magic_split, rotate_compare

# ── 1. Corpus quality check ────────────────────────────────────────────────────
corpus = """
The entropy tokenizer learns BPE merge rules from raw text.
It detects the natural separator automatically with magic_split.
The rotate_compare function finds morphological stems via circular rotation.
Training uses frequency times length ranking for better compression.
"""
tok = Tokenizer(vocab_size=256)
stats = tok.corpus_stats(corpus)
print(f"Corpus quality: {stats['quality']}")
print(f"  char entropy: {stats['char_entropy']} bits")
print(f"  unique words: {stats['unique_words']}")
print(f"  splitter    : {stats['splitter']}")

# ── 2. Train ───────────────────────────────────────────────────────────────────
print("\nTraining...")
tok.train(corpus * 30, verbose=True)  # repeat to get enough frequency

# ── 3. Encode / decode ─────────────────────────────────────────────────────────
text = "the entropy tokenizer detects separators"
ids = tok.encode(text)
decoded = tok.decode(ids)
print(f"\nOriginal : {text}")
print(f"Token IDs: {ids}")
print(f"Decoded  : {decoded}")

# ── 4. Save and reload ─────────────────────────────────────────────────────────
tok.save("/tmp/example_vocab.json")
tok2 = Tokenizer.load("/tmp/example_vocab.json")
assert tok2.encode(text) == ids, "Round-trip failed!"
print(f"\nVocab saved and reloaded — encode is identical ✓")

# ── 5. magic_split on different formats ───────────────────────────────────────
print("\nmagic_split:")
examples = [
    "alice,30,madrid\nbob,25,barcelona",
    "def foo(x):\n    return x+1\ndef bar(y):",
    "/usr/bin:/usr/local:/home",
    "ATCGATCG\nGCTAGCTA\nTTAGCCAT",
    "2024|ERROR|auth\n2024|INFO|login",
]
for ex in examples:
    sep = magic_split(ex)
    print(f"  {repr(ex[:35]):<40} → {repr(sep)}")

# ── 6. rotate_compare ─────────────────────────────────────────────────────────
print("\nrotate_compare:")
pairs = [
    ("walking",   "walked"),
    ("running",   "runner"),
    ("compute",   "computer"),
    ("ATCGATCG",  "TCGATCGA"),
]
for a, b in pairs:
    stem = rotate_compare(a, b)
    print(f"  ({repr(a):<12}, {repr(b):<12}) → {repr(stem)}")
