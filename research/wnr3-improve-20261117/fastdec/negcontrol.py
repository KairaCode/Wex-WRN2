#!/usr/bin/env python3
"""Negative controls for kernels 3 and 4: remove one check at a time from a copy of wex_fast_decode.h (inside one
function only), build the harness with TEST_DEFAULT_ONLY (reference vs the default entry point: kernel 4 in the
AVX-512 build, kernel 3 in the AVX2 build) and run `mutate`; the mutation test must report a MISMATCH.
Usage: python3 negcontrol.py CASES SEED REC...   (writes out/neg_*.h, out/neg_*, prints one line per control;
NEG_ONLY=TEXT runs the controls whose name contains TEXT)
Expected: the two "end-pointer check at block start" controls are NOT caught: that check only keeps reads inside the
slack (memory safety); the end-of-stream check still gives the same verdict, as for kernel 1 in the original
negative_control.sh, which does not test it either."""
import subprocess, sys, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "/workspace/Wex-WRN2/c"
MODEL = "<TMP>/wnr3-data/recs/model.bin"
cases, seed, recs = sys.argv[1], sys.argv[2], sys.argv[3:]
base = open(os.path.join(HERE, "wex_fast_decode.h")).read()

def in_function(text, fname, old, new):
    i = text.index("WEXF_INLINE int " + fname + "(")
    j = text.index("\n}\n", i)
    body = text[i:j]
    assert body.count(old) >= 1, (fname, old)
    return text[:i] + body.replace(old, new) + text[j:]

controls = [
    ("k4 weights: end-pointer check at block start", "z", "wexf_weights_t16",
     "if (_mm512_cmpgt_epu32_mask(p[g], e[g])) return -1;", ";"),
    ("k4 weights: end-pointer check at stream end", "z", "wexf_weights_t16",
     "if (pp[k] != ee[k] || st[k] != x0) return -1;", "if (st[k] != x0) return -1;"),
    ("k4 weights: final state (stream hash) check", "z", "wexf_weights_t16",
     "if (pp[k] != ee[k] || st[k] != x0) return -1;", "if (pp[k] != ee[k]) return -1;"),
    ("k4 weights: canonical-zero check", "z", "wexf_weights_t16",
     "if (J->cz && _mm512_test_epi32_mask(bad, bad)) return -1;", ";"),
    ("k4 weights: ceil underflow check", "z", "wexf_weights_t16",
     "if (wexf_ceil_fused(out[k] + (size_t)b * 16u, sc[k] + b)) return -1;", "(void)wexf_ceil_fused(out[k] + (size_t)b * 16u, sc[k] + b);"),
    ("k4 scales: end-pointer check at stream end", "z", "wexf_scales_t16",
     "if (pp[k] != ee[k] || st[k] !=", "if (st[k] !="),
    ("k4 scales: final state (stream hash) check", "z", "wexf_scales_t16",
     "if (pp[k] != ee[k] || st[k] != WEX_NR_L + (wex_nr3_h32_final(hh[k], 16u + count) & WEX_NR3_HASH_MASK)) return -1;",
     "if (pp[k] != ee[k]) return -1;"),
    ("k3 weights: end-pointer check at block start", "h", "wexf_weights_t8",
     "if (_mm256_movemask_epi8(_mm256_cmpgt_epi32(p[g], e[g]))) return -1;", ";"),
    ("k3 weights: end-pointer check at stream end", "h", "wexf_weights_t8",
     "if (pp[k] != ee[k] || st[k] != x0) return -1;", "if (st[k] != x0) return -1;"),
    ("k3 weights: final state (stream hash) check", "h", "wexf_weights_t8",
     "if (pp[k] != ee[k] || st[k] != x0) return -1;", "if (pp[k] != ee[k]) return -1;"),
    ("k3 weights: canonical-zero check", "h", "wexf_weights_t8",
     "if (J->cz && !_mm256_testz_si256(bad, bad)) return -1;", ";"),
    ("k3 weights: ceil underflow check", "h", "wexf_weights_t8",
     "if (wexf_ceil_fused(out[k] + (size_t)b * 16u, sc[k] + b)) return -1;", "(void)wexf_ceil_fused(out[k] + (size_t)b * 16u, sc[k] + b);"),
]
flags = {"z": "-march=x86-64-v4 -mavx512vbmi2", "h": "-march=haswell"}
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
caught = 0
only = os.environ.get("NEG_ONLY", "")
tried = 0
for n, (name, build, fn, old, new) in enumerate(controls):
    if only not in name:
        continue
    tried += 1
    hdr = os.path.join(HERE, "out", f"neg_{n}.h")
    open(hdr, "w").write(in_function(base, fn, old, new))
    test_c = os.path.join(HERE, "out", f"neg_{n}.c")
    open(test_c, "w").write(open(os.path.join(HERE, "test_fast_decode.c")).read().replace(
        '#include "wex_fast_decode.h"', f'#include "neg_{n}.h"'))
    exe = os.path.join(HERE, "out", f"neg_{n}")
    subprocess.run(f"gcc -std=c11 -O3 {flags[build]} -fopenmp -DTEST_DEFAULT_ONLY -I{SRC} -I{HERE}/out {test_c} -o {exe}",
                   shell=True, check=True)
    r = subprocess.run([exe, "mutate", MODEL, cases, seed] + recs, capture_output=True, text=True)
    hit = "MISMATCH" in r.stdout
    caught += hit
    line = [l for l in r.stdout.splitlines() if "MISMATCH" in l or l.startswith("mutate:")]
    print(f"{'CAUGHT' if hit else 'MISSED'} | {name} | {line[0] if line else r.stdout[-200:]}", flush=True)
print(f"negative controls: {caught} of {tried} caught")
