"""Mutation check of the WNR3 CPU tests: each mutant edits one rule in a scratch copy of the codec headers,
rebuilds tests/test_wex_format.c (and, for store mutants, test_expert_store_wex) and must FAIL.
usage: python -I mutants.py WORKTREE_C_DIR SCRATCH_DIR   (needs MSYS2 at C:\\msys64)"""
import os
import shutil
import subprocess
import sys

SRC, SCRATCH = sys.argv[1], sys.argv[2]
NR = 'wex_neural_rans.h'
FMT = 'wex_format.h'
MUTANTS = [
    ('scale-stream hash not checked', NR,
     'if (ptr != end || state != (nr3 ? wex_nr3_stream_x0(ident, scales_out[m] + first, count) : WEX_NR_L)) return -1;',
     'if (ptr != end) return -1;'),
    ('weight-stream hash not checked', NR,
     'if (ptr != end || state != (nr3 ? wex_nr3_stream_x0(ident, weights_out[m] + first / 2u, count / 2u) : WEX_NR_L))',
     'if (ptr != end)'),
    ('CZ nibble 8 accepted', NR, '|| (cz && symbol == 8u)) return -1;', ') return -1;'),
    ('ceil scale 0 accepted', NR, 'if (!scales[m][b]) return -1;', ''),
    ('ceil inverse skipped', NR,
     'return nr3 ? wex_nr3_ceil_inverse(layout, v->flags & WEX_NR3_FLAG_CEIL_MASK, scales_out, weights_out) : 0;',
     'return 0;'),
    ('reserved flag bits accepted', NR, ': (record[27] & ~(WEX_NR3_FLAG_CEIL_MASK | WEX_NR3_FLAG_CZ)) != 0) return -1;',
     ': 0) return -1;'),
    ('rule g ties take the last pair', NR, 'if (histogram[i] + histogram[i + 1u] > top)', 'if (histogram[i] + histogram[i + 1u] >= top)'),
    ('even block at scale 255 transformed', NR, 'if (scales[m][b] == 255u) { kept++; continue; }', ''),
    ('ceil gate ignored', NR, 'if (!fives) {', 'if (1) {'),
    ('canonical zero only on low nibbles', NR, "if ((weights[i] >> 4) == 8u) { weights[i] &= 0x0fu; mapped++; }", ''),
    ('owner not hashed (both sides)', NR, 'ident[0] = model->id; ident[1] = owner;', 'ident[0] = model->id; ident[1] = 0;'),
    ('MurmurHash3 c1 wrong', NR, 'k *= UINT32_C(0xcc9e2d51);', 'k *= UINT32_C(0xcc9e2d53);'),
    ('hash length omits the ident', NR, 'data, n, 16u) & WEX_NR3_HASH_MASK);', 'data, n, 0u) & WEX_NR3_HASH_MASK);'),
    ('flag 32 allowed with 16', FMT, '(hdr->flags & (WEX_HEADER_NEURAL_RANS | WEX_HEADER_NEURAL_RANS2))) ||', '0) ||'),
    ('codec 5 accepted without flag 32', FMT,
     'uint32_t need = entry->flags == WEX_ENTRY_NEURAL_RANS3 ? WEX_HEADER_NEURAL_RANS3 : WEX_HEADER_NEURAL_RANS2;',
     'uint32_t need = WEX_HEADER_NEURAL_RANS2 | WEX_HEADER_NEURAL_RANS3;'),
    ('codec 5 parsed as WNR2', FMT, 'return e && e->flags == WEX_ENTRY_NEURAL_RANS3 ? WEX_NR3_MAGIC : WEX_NR2_MAGIC;',
     'return WEX_NR2_MAGIC;'),
    ('owner check always passes', FMT, 'return record && wex_read_u32_le(record + 28u) == (layer << 16 | expert);',
     'return 1;'),
]
BASH = r'C:\msys64\usr\bin\bash.exe'


def run(work):
    cmd = ('export PATH=/mingw64/bin:/usr/bin:$PATH; cd "%s" && gcc -O2 -w tests/test_wex_format.c -o tf.exe && '
           './tf.exe >/dev/null 2>&1 && make -s tests/test_expert_store_wex.exe >/dev/null 2>&1 && '
           'cd tests && ../tests/test_expert_store_wex.exe >/dev/null 2>&1' % work.replace('\\', '/'))
    return subprocess.run([BASH, '-lc', cmd]).returncode


killed = 0
for name, fname, old, new in MUTANTS:
    work = os.path.join(SCRATCH, 'mut')
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(SRC, work, ignore=shutil.ignore_patterns('*.o', '*.exe', '*.d', 'build*'))
    path = os.path.join(work, fname)
    text = open(path, encoding='utf-8', newline='').read()
    assert text.count(old) == 1, (name, text.count(old))
    open(path, 'w', encoding='utf-8', newline='').write(text.replace(old, new))
    rc = run(work)
    killed += rc != 0
    print(f"{'KILLED' if rc else 'SURVIVED'}  {name}", flush=True)
work = os.path.join(SCRATCH, 'mut')
shutil.rmtree(work, ignore_errors=True)
shutil.copytree(SRC, work, ignore=shutil.ignore_patterns('*.o', '*.exe', '*.d', 'build*'))
print('unmutated:', 'PASS' if run(work) == 0 else 'FAIL')
print(f'{killed}/{len(MUTANTS)} mutants killed')
