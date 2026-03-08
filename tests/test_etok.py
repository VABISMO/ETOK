#!/usr/bin/env python3
"""
etok test suite — python3 tests/test_etok.py
"""
import sys, os, json, math, tempfile, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from etok import Tokenizer, magic_split, rotate_compare, common_string, char_entropy, token_entropy
from collections import Counter

OK = "✓"; FAIL = "✗"; errors = 0

def check(label, got, expected=None, cond=None):
    global errors
    ok = cond if cond is not None else (got == expected)
    if not ok:
        errors += 1
        print(f"  {FAIL} {label}")
        if expected is not None:
            print(f"       got={repr(got)!r}  expected={repr(expected)!r}")
    else:
        print(f"  {OK} {label}  [{repr(got)!s:.60}]")

def section(title):
    print(f"\n{'─'*54}\n  {title}\n{'─'*54}")

# ══ common_string ══════════════════════════════════════════
section("common_string  O(n+m)")
for a, b, want in [
    ("abcdef",      "cdefgh",          "cdef"),
    ("hello",       "hello world",     "hello"),
    ("",            "hello",           ""),
    ("x",           "x",               "x"),
    ("the cat sat", "cat ran sat",     "cat"),
]:
    check(f"({repr(a)[:15]}, {repr(b)[:15]})", common_string(a,b),
          cond=want in common_string(a,b) or common_string(a,b)==want)

# ══ magic_split ════════════════════════════════════════════
section("magic_split — auto separator detection")
for text, want in [
    ("the cat sat on the mat",    ' '),
    ("1,2,3,4,5,1,2,3",          ','),
    ("x=1;y=2;z=3;a=4",          ';'),
    ("/usr/bin:/usr/local:/bin",  '/'),
    ("key|val|key|val|key|val",   '|'),
    ("a\nb\nc\nd\ne\nf\ng",       '\n'),
]:
    check(f"magic_split({repr(text[:25])})", magic_split(text), want)

# ══ rotate_compare ═════════════════════════════════════════
section("rotate_compare — rotation-invariant morphology")
for a, b, must_contain in [
    ("walking",   "walked",    "walk"),
    ("running",   "runner",    "runn"),
    ("nation",    "national",  "nation"),
    ("compute",   "computer",  "compute"),
    ("ATCGATCG",  "TCGATCGA",  "TCGAT"),
]:
    result = rotate_compare(a, b)
    check(f"rotate_compare({repr(a)},{repr(b)})", result,
          cond=must_contain in result or result in must_contain)

# ══ char_entropy ═══════════════════════════════════════════
section("char_entropy")
_, e = char_entropy("aaaa"); check("uniform → 0.0", round(e,4), 0.0)
_, e = char_entropy("ab");   check("binary  → 1.0", round(e,4), 1.0)
_, e = char_entropy("hello world"); check("> 2.5 for real text", e, cond=e>2.5)

# ══ Tokenizer: train ═══════════════════════════════════════
section("Tokenizer — train")
CORPUS = (
    "the cat sat on the mat the dog ran over the mat\n"
    "the quick brown fox jumps over the lazy dog\n"
    "hello world hello earth goodbye world sunrise sunset\n"
    "mathematics is the language of the universe indeed\n"
) * 15

tok = Tokenizer(vocab_size=128, min_freq=2)
tok.train(CORPUS, verbose=False)
check("trained flag",           tok._trained, True)
check("vocab not empty",        len(tok), cond=len(tok)>20)
check("has <bos>",              '<bos>' in tok.vocab, True)
check("has <eos>",              '<eos>' in tok.vocab, True)
check("has </w>",               '</w>'  in tok.vocab, True)
check("train_entropy_start > 0",tok.train_entropy_start, cond=tok.train_entropy_start>0)

# ══ Tokenizer: encode/decode ═══════════════════════════════
section("Tokenizer — encode / decode round-trip")
for text in ["the cat sat", "hello world", "the quick brown fox"]:
    ids = tok.encode(text)
    check(f"encode non-empty: {repr(text)}", ids, cond=len(ids)>0)
    check(f"all IDs in range", all(0<=i<len(tok) for i in ids), True)
    decoded = tok.decode(ids)
    check(f"round-trip: {repr(text)}", decoded, text)

ids_no_spec = tok.encode("hello", add_special=False)
bos, eos = tok.vocab['<bos>'], tok.vocab['<eos>']
check("no BOS/EOS when add_special=False",
      bos not in ids_no_spec and eos not in ids_no_spec, True)

# ══ Tokenizer: save/load ═══════════════════════════════════
section("Tokenizer — save / load")
with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
    vpath = f.name
tok.save(vpath)
tok2 = Tokenizer.load(vpath)
check("vocab size matches",   len(tok2), len(tok))
check("splitter matches",     tok2.splitter, tok.splitter)
check("merges count matches", len(tok2.merges), len(tok.merges))
check("same encoding",        tok.encode("the cat sat"), tok2.encode("the cat sat"))

with open(vpath) as f: data = json.load(f)
check("json has version",  'version' in data, True)
check("json has vocab",    'vocab'   in data, True)
check("json has merges",   'merges'  in data, True)
check("json has splitter", 'splitter'in data, True)
os.unlink(vpath)

# ══ corpus_stats ═══════════════════════════════════════════
section("corpus_stats")
stats = tok.corpus_stats(CORPUS)
for key in ['char_entropy','word_entropy','total_chars','unique_words','ttr','quality']:
    check(f"has '{key}'", key in stats, True)
check("quality valid", stats['quality'] in ('good','medium','low'), True)
check("repetitive → 'low'",
      tok.corpus_stats("the the the the the "*50)['quality'], 'low')

# ══ freq×length ranking ════════════════════════════════════
section("freq×length ranking — compression advantage")
def bpe_ent(text, vs, ranking):
    wf = Counter(text.split())
    corp = {w: (list(w)+['</w>'], f) for w,f in wf.items()}
    vocab = {t for w,(tk,_) in corp.items() for t in tk}
    for _ in range(vs-len(vocab)):
        pairs: Counter = Counter()
        for w,(tk,freq) in corp.items():
            for i in range(len(tk)-1):
                a,b = tk[i],tk[i+1]
                pairs[(a,b)] += freq*(len(a)+len(b)) if ranking=='fl' else freq
        if not pairs: break
        best=max(pairs,key=pairs.get)
        if pairs[best]<2: break
        a,b=best; m=a+b
        nc={}
        for w,(tk,freq) in corp.items():
            nt,i=[],0
            while i<len(tk):
                if i<len(tk)-1 and tk[i]==a and tk[i+1]==b: nt.append(m);i+=2
                else: nt.append(tk[i]);i+=1
            nc[w]=(nt,freq)
        corp=nc; vocab.add(m)
    all_t=[t for w,(tk,_) in corp.items() for t in tk]
    c=Counter(all_t); n=len(all_t)
    return -sum((v/n)*math.log2(v/n) for v in c.values())

e_fl = bpe_ent(CORPUS, 80, 'fl')
e_f  = bpe_ent(CORPUS, 80, 'f')
check(f"freq×len ent ({e_fl:.4f}) ≤ freq-only ({e_f:.4f})",
      e_fl, cond=e_fl <= e_f + 0.05)

# ══ C binary ═══════════════════════════════════════════════
section("C binary integration")
binary = os.path.join(os.path.dirname(__file__), '..', 'etok')
if os.path.exists(binary):
    with tempfile.NamedTemporaryFile(mode='w',suffix='.txt',delete=False) as f:
        f.write(CORPUS); cpath=f.name
    with tempfile.NamedTemporaryFile(suffix='.json',delete=False) as f:
        vpath=f.name

    r=subprocess.run([binary,'train','--data',cpath,'--out',vpath,'--vocab_size','128'],
                     capture_output=True,text=True)
    check("train exit 0",           r.returncode, 0)
    check("train output ok",        'Trained' in r.stdout, True)

    r2=subprocess.run([binary,'encode','--vocab',vpath,'--text','the cat sat'],
                      capture_output=True,text=True)
    check("encode exit 0",          r2.returncode, 0)
    check("encode produces IDs",    len(r2.stdout.strip())>0, True)

    ids_str=r2.stdout.strip()
    r3=subprocess.run([binary,'decode','--vocab',vpath,'--ids',ids_str],
                      capture_output=True,text=True)
    check("decode exit 0",          r3.returncode, 0)
    check("decode round-trip",      r3.stdout.strip(), "the cat sat")

    r4=subprocess.run([binary,'rotate','--a','walking','--b','walked'],
                      capture_output=True,text=True)
    check("rotate exit 0",          r4.returncode, 0)
    check("rotate finds 'walk'",    'walk' in r4.stdout, True)

    r5=subprocess.run([binary,'stats','--data',cpath],
                      capture_output=True,text=True)
    check("stats exit 0",           r5.returncode, 0)
    check("stats has entropy",      'entropy' in r5.stdout, True)

    os.unlink(cpath); os.unlink(vpath)
else:
    print(f"  ⚠ binary not found — run 'make' first")

# ══ Summary ════════════════════════════════════════════════
print(f"\n{'═'*54}")
if errors == 0:
    print(f"  {OK}  ALL TESTS PASSED")
else:
    print(f"  {FAIL}  {errors} TEST(S) FAILED")
print(f"{'═'*54}\n")
sys.exit(0 if errors == 0 else 1)
