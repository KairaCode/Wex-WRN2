# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Host C++ copies of the CUDA WNR3 decoder, for the harness fuzz_wnr3_cuda.cpp.

    python -I wex4/fuzz/cuda_host.py C_DIR OUT_DIR

Writes OUT_DIR/wex_cuda.cuh (from C_DIR/wex_cuda.cuh) and OUT_DIR/tools/wex_cuda_bridge.cu (from
C_DIR/tools/wex_cuda_bridge.cu), so that the bridge's own `#include "../wex_cuda.cuh"` finds the copy. Most of
the CUDA source compiles as host C++ once cuda_host.h has defined the qualifiers, the built-in variables and the
intrinsics. Four constructs cannot be reached by a macro, and only those are rewritten, each one by an exact
pattern whose number of matches is checked:

  - a kernel launch `K<<<G, B, S, Q>>>(ARGS)` becomes `wex_shim_launch(K, G, B, S, Q)(ARGS)` (what nvcc's front
    end does with it, with cuda_host.h's launcher instead of the CUDA runtime's);
  - a `__shared__ T NAME[N];` becomes a reference to an array of the same type that cuda_host.h allocates for
    each block, and `extern __shared__ T NAME[];` a pointer to the block's dynamic shared memory;
  - the three inline PTX statements (prmt with the sign-replicate selector, cp.async.ca.shared.global of 4
    bytes, cp.async.commit_group with cp.async.wait_group) become calls of the functions that emulate them;
  - `#include <cuda_runtime.h>` becomes `#include "cuda_host.h"`.

The bridge is copied unchanged. Every other byte of both files is the same as in C_DIR; the script stops with
exit status 1 when a pattern does not match the expected number of times, or when a launch, a __shared__
declaration or an asm statement is left. Standard library only."""
import os
import re
import sys

PRMT = 'asm("prmt.b32 %0, %1, 0, 0xbb99;" : "=r"(r) : "r"(x));'
CP_ASYNC = ('asm volatile("{\\n .reg .pred p;\\n setp.ne.b32 p, %2, 0;\\n @p cp.async.ca.shared.global [%0], [%1], '
            '4;\\n}\\n"\n                 :: "r"(destination), "l"(word), "r"((int)predicate) : "memory");')
WAIT = 'asm volatile("cp.async.commit_group;\\ncp.async.wait_group " #n ";\\n" ::: "memory")'

# (what, pattern, replacement, expected matches); patterns of kind 'text' match exactly
RULES = [
    ('include', 'text', '#include <cuda_runtime.h>', '#include "cuda_host.h"', 1),
    ('launch', 're', r'\b([A-Za-z_]\w*(?:<[^<>();]*>)?)<<<(.*?)>>>\(', r'wex_shim_launch(\1, \2)(', 13),
    ('dynamic shared memory', 're', r'extern __shared__ ([A-Za-z_]\w*) ([A-Za-z_]\w*)\[\];',
     r'\1 *const \2 = (\1 *)wex_shim_dynamic_shared();', 1),
    ('shared memory', 're', r'__shared__ ([A-Za-z_][\w ]*?) ([A-Za-z_]\w*)\[([^;\]]+)\];',
     r'\1 (&\2)[\3] = *reinterpret_cast<\1 (*)[\3]>(wex_shim_shared(sizeof(\1[\3]), __LINE__));', 3),
    ('inline PTX prmt', 'text', PRMT, 'r = wex_shim_prmt(x, 0u, 0xbb99u);', 1),
    ('inline PTX cp.async', 'text', CP_ASYNC, 'wex_shim_cp_async_ca4(destination, word, (int)predicate);', 1),
    ('inline PTX commit and wait', 'text', WAIT, 'wex_shim_cp_async_commit_wait(n)', 1),
]
LEFT = [('launch', '<<<'), ('__shared__', '__shared__'), ('asm statement', re.compile(r'\basm\s*(volatile\s*)?\('))]


def fail(message):
    print(f'cuda_host.py: {message}', file=sys.stderr)
    sys.exit(1)


def left_over(text):
    found = []
    for what, needle in LEFT:
        n = len(needle.findall(text)) if hasattr(needle, 'findall') else text.count(needle)
        if n:
            found.append(f'{n} {what}')
    return found


def main():
    if len(sys.argv) != 3:
        fail('usage: python -I wex4/fuzz/cuda_host.py C_DIR OUT_DIR')
    source, out = sys.argv[1], sys.argv[2]
    with open(os.path.join(source, 'wex_cuda.cuh'), 'rb') as f:
        header = f.read().decode('utf-8')
    with open(os.path.join(source, 'tools', 'wex_cuda_bridge.cu'), 'rb') as f:
        bridge = f.read()
    done = []
    for what, kind, pattern, replacement, expected in RULES:
        if kind == 'text':
            n = header.count(pattern)
            header = header.replace(pattern, replacement)
        else:
            header, n = re.subn(pattern, replacement, header)
        if n != expected:
            fail(f'c/wex_cuda.cuh: {n} matches of the rule "{what}", {expected} expected; the source changed, '
                 'so cuda_host.py and cuda_host.h have to be checked against it')
        done.append(f'{n} {what}')
    rest = left_over(header)
    if rest:
        fail('c/wex_cuda.cuh: left after the rewrite: ' + ', '.join(rest))
    rest = left_over(bridge.decode('utf-8'))
    if rest:
        fail('c/tools/wex_cuda_bridge.cu: the bridge is copied unchanged, but it has ' + ', '.join(rest))
    os.makedirs(os.path.join(out, 'tools'), exist_ok=True)
    with open(os.path.join(out, 'wex_cuda.cuh'), 'wb') as f:
        f.write(header.encode('utf-8'))
    with open(os.path.join(out, 'tools', 'wex_cuda_bridge.cu'), 'wb') as f:
        f.write(bridge)
    print('host C++ copy of c/wex_cuda.cuh: rewrote ' + ', '.join(done) + '; c/tools/wex_cuda_bridge.cu copied unchanged')


if __name__ == '__main__':
    main()
