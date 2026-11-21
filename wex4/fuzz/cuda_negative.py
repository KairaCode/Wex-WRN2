# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Negative control of the harness fuzz_wnr3_cuda.cpp: one change at a time in the host copy of c/wex_cuda.cuh or
c/tools/wex_cuda_bridge.cu, a build of the harness on that copy, and a replay of the seeds, which must fail. A
harness that still passes with a broken decoder would prove nothing about the real one.

    python -I wex4/fuzz/cuda_negative.py [--gpu]

Run it after `bash wex4/fuzz/run.sh --replay`, which writes the seeds to wex4/fuzz/out/seeds. CXX names the C++
compiler (default g++), WEX_NEG_JOBS the number of builds at a time (default: the CPU count, at most 4). Each
change is an exact text that must occur once in the copy; when the source no longer has it, the script stops
with exit status 1 and names the control, so that the list is checked again. Everything goes to
wex4/fuzz/out/negative/ (git-ignored). Last line on success: "negative control: N of N broken copies noticed".

With --gpu (on a machine with an NVIDIA GPU and its driver, after `bash wex4/fuzz/run.sh --replay` or `--gpu`,
for the seeds) the change goes into the source of the PTX instead (cuda_gpu.h), and the harness built with
-DWEX_FUZZ_GPU from the unchanged host copy replays the seeds with 4 mutator inputs each: a control is noticed
when the replay fails on an ORACLE FAILED line of the GPU path (the emulated path runs the unchanged copy).
The three controls on the bridge do not apply (the bridge does not run on the GPU). Three controls break a
barrier or a wait (GPU_RACES): on a GPU they leave a race that may or may not change a byte, so they are
reported and not required; every other control must be noticed. CUDA_CLANG names the clang that writes the PTX
(default clang++). Last line: "negative control on the GPU: N of M broken copies noticed (...)".
Standard library only."""
import concurrent.futures
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'wex4', 'fuzz', 'out')
EXE = '.exe' if sys.platform == 'win32' else ''

CUH, BRIDGE = 'wex_cuda.cuh', os.path.join('tools', 'wex_cuda_bridge.cu')
SCALE_END = ('if (s.pos != end) atomicOr(error, WEX_CUDA_ERR_STRUCT);\n'
             '    else if (state != expect) atomicOr(error, plan.nr3 ? WEX_CUDA_ERR_HASH : WEX_CUDA_ERR_STRUCT);\n}\n\n/* Each')
STATE_CHECK = 'if (state < WEX_NR_L || state >= (WEX_NR_L << 8)) { atomicOr(error, WEX_CUDA_ERR_STRUCT); return; }\n'
# (what is broken, file, exact text, replacement)
CONTROLS = [
    ('canonical zero: nibble 8 is not refused', CUH,
     'if (plan.cz && (wex_cuda_has_nibble8', 'if (false && (wex_cuda_has_nibble8'),
    ('ceil form: the scale is not decremented', CUH,
     'scales[block] = (unsigned char)(scale - 1u);', ';'),
    ('ceil form: no error for a coded scale 0 in a block of maximum 5', CUH,
     'if (!scale) { atomicOr(error, WEX_CUDA_ERR_CEIL); return; }', ''),
    ('ceil form: the weights are not doubled back', CUH,
     'for (uint32_t w = 0; w < 4u; w++) word[w] = wex_cuda_ceil_dbl8(word[w]);',
     'for (uint32_t w = 0; w < 0u; w++) word[w] = wex_cuda_ceil_dbl8(word[w]);'),
    ('ceil form: the inverse runs on every matrix, whatever its flag', CUH,
     'const bool ceil_matrix = plan.nr3 && ((plan.ceil_mask >> m) & 1u);', 'const bool ceil_matrix = plan.nr3;'),
    ('hash: the tail of a scale stream is not folded', CUH,
     'if (held) hash = wex_nr3_h32_tail(hash, pending);', ';'),
    ('hash: the vector path of the scale kernel folds 7 words of 8', CUH,
     'for (uint32_t k = 0; k < 8u; k++) hash = wex_nr3_h32_block(hash, w[k]);',
     'for (uint32_t k = 0; k < 7u; k++) hash = wex_nr3_h32_block(hash, w[k]);'),
    ('hash: the length of a weight stream counted twice', CUH,
     'wex_nr3_h32_final(hash, 16u + blocks * 16u)', 'wex_nr3_h32_final(hash, 16u + blocks * 32u)'),
    ('initial state: no lower bound in the weight kernel', CUH,
     STATE_CHECK + '    int median', STATE_CHECK.replace('state < WEX_NR_L || ', '') + '    int median'),
    ('initial state: no upper bound in the scale kernel', CUH,
     STATE_CHECK + '    bool vector', STATE_CHECK.replace(' || state >= (WEX_NR_L << 8)', '') + '    bool vector'),
    ('stream length: a scale stream may end before its last byte', CUH,
     SCALE_END, SCALE_END.replace('if (s.pos != end) atomicOr(error, WEX_CUDA_ERR_STRUCT);\n    else ', '')),
    ('context: the block maximum starts at 8 instead of "none"', CUH,
     'maximum = 0, word[4]', 'maximum = 8, word[4]'),
    ('renormalization: two bytes only below 2^14', CUH,
     'state < (1u << 15) ? 16u', 'state < (1u << 14) ? 16u'),
    ('lookup 1: the frequency of symbol 15 is not cut at 2^14', CUH,
     'frequency = min(min(hi & 0xffffu, hi >> 16), WEX_NR2_TOTAL) - start;',
     'frequency = min(hi & 0xffffu, hi >> 16) - start;'),
    ('lookup 0: no exact walk in a bucket that spans symbols', CUH,
     'if (slot >= start + frequency) {   /* exact walk inside a mixed bucket */', 'if (false) {'),
    ('barrier: the weight kernel reads its CDF rows before every lane stored its part', CUH,
     'rows[i] = __ldg(source + i);\n        __syncthreads();', 'rows[i] = __ldg(source + i);'),
    ('barrier: the scale kernel reads its tables before every lane stored its part', CUH,
     '((uint4 *)symbol_table)[i] = __ldg(source + i);\n    __syncthreads();',
     '((uint4 *)symbol_table)[i] = __ldg(source + i);'),
    ('payload ring: cp.async waits for one group fewer', CUH,
     'WEX_CUDA_NR2_WAIT(1);', 'WEX_CUDA_NR2_WAIT(2);'),
    ('payload ring: refilled by 8 words per 32 symbols', CUH,
     's.requested = max(s.requested, min(s.requested + 16u, target));',
     's.requested = max(s.requested, min(s.requested + 8u, target));'),
    ('commit kernel: copies whatever the error word says', CUH,
     'if (*(const volatile uint32_t *)error) return;', ';'),
    ('bridge: kernel B does not wait for the weight payload', BRIDGE,
     'RUN_OK(cudaStreamWaitEvent(work,workspace.data,0));', ';'),
    ('bridge: the host batch copies back a record refused on the device', BRIDGE,
     'if(host.errors[j]) { status[j]=-1; wex_cuda_nr3_report(host.errors[j]); continue; }',
     'if(host.errors[j]) { status[j]=-1; wex_cuda_nr3_report(host.errors[j]);'
     ' (void)wex_cuda_copy_back(entries[j],plans[(size_t)j],lane[r].device_raw,dst_host[j],lane[r].aux); continue; }'),
    ('bridge: a batch record refused on the device gets status 0', BRIDGE,
     'host.errors[i] ? -1 : 0;', 'host.errors[i] ? 0 : 0;'),
]
# --gpu: the controls that leave a race (a missing barrier or wait) rather than a wrong result on every run. A
# block of these kernels is one warp of 32 lanes, and a copy may land before its wait: whether a GPU shows the
# race is not fixed, so these are reported and not required there.
GPU_RACES = (
    'barrier: the weight kernel reads its CDF rows before every lane stored its part',
    'barrier: the scale kernel reads its tables before every lane stored its part',
    'payload ring: cp.async waits for one group fewer',
)


def fail(message):
    print(f'cuda_negative.py: {message}', file=sys.stderr)
    sys.exit(1)


def seeds():
    files = []
    for folder in ('wnr3', 'file', 'testdata'):
        path = os.path.join(OUT, 'seeds', folder)
        if not os.path.isdir(path):
            fail(f'{path} is missing: run bash wex4/fuzz/run.sh --replay first')
        files += [os.path.join(path, name) for name in sorted(os.listdir(path))]
    return files


def build(index, change, gpu=False):
    """The harness built on a host copy with one change (None: unchanged), with the GPU path when gpu is set;
    returns (executable, error text)."""
    folder = os.path.join(OUT, 'negative-gpu' if gpu else 'negative', str(index))
    shutil.rmtree(folder, ignore_errors=True)
    host = os.path.join(folder, 'wex_cuda_host')
    r = subprocess.run([sys.executable, '-I', os.path.join(ROOT, 'wex4', 'fuzz', 'cuda_host.py'),
                        os.path.join(ROOT, 'c'), host], capture_output=True, text=True)
    if r.returncode:
        return None, r.stderr
    if change:
        path = os.path.join(host, change[1])
        with open(path, 'rb') as f:
            text = f.read().decode('utf-8')
        if text.count(change[2]) != 1:
            return None, f'its text occurs {text.count(change[2])} times in {change[1]}, 1 expected'
        with open(path, 'wb') as f:
            f.write(text.replace(change[2], change[3]).encode('utf-8'))
    exe = os.path.join(folder, 'fuzz_wnr3_cuda' + EXE)
    command = [os.environ.get('CXX') or 'g++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Wno-unused-parameter',
               '-Wno-unused-function', '-Wno-unknown-pragmas', '-DWEX_FUZZ_STANDALONE', '-pthread',
               '-I', folder, '-I', os.path.join(ROOT, 'wex4', 'research', 'fast-cpu-decode'), '-I', os.path.join(ROOT, 'c'),
               '-I', os.path.join(ROOT, 'wex4', 'fuzz', 'engine'), '-I', os.path.join(ROOT, 'wex4', 'fuzz'),
               os.path.join(ROOT, 'wex4', 'fuzz', 'fuzz_wnr3_cuda.cpp'), '-o', exe] + (['-DWEX_FUZZ_GPU'] if gpu else [])
    r = subprocess.run(command, capture_output=True, text=True, cwd=folder)
    return (exe, '') if not r.returncode else (None, r.stdout + r.stderr)


def ptx(index, change):
    """--gpu: sm86.ptx and sm75.ptx from c/wex_cuda.cuh with one change (None: unchanged), as run.sh --gpu builds
    them; returns (folder, error text)."""
    folder = os.path.join(OUT, 'negative-gpu', f'ptx{index}')
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)
    with open(os.path.join(ROOT, 'c', CUH), 'rb') as f:
        text = f.read().decode('utf-8')
    if change:
        if text.count(change[2]) != 1:
            return None, f'its text occurs {text.count(change[2])} times in c/{CUH}, 1 expected'
        text = text.replace(change[2], change[3])
    with open(os.path.join(folder, CUH), 'wb') as f:   # found before c/wex_cuda.cuh
        f.write(text.encode('utf-8'))
    with open(os.path.join(folder, 'cuda_runtime.h'), 'w') as f:
        f.write('#include "cuda_device.h"\n')
    for arch in ('86', '75'):
        r = subprocess.run([os.environ.get('CUDA_CLANG') or 'clang++', '-x', 'cuda', '--cuda-device-only',
                            f'--cuda-gpu-arch=sm_{arch}', '-nocudainc', '-nocudalib', '-O2', '-S', '-I', folder,
                            '-I', os.path.join(ROOT, 'c'), '-I', os.path.join(ROOT, 'wex4', 'fuzz'),
                            os.path.join(ROOT, 'wex4', 'fuzz', 'cuda_device.cu'), '-o', os.path.join(folder, f'sm{arch}.ptx')],
                           capture_output=True, text=True)
        if r.returncode:
            return None, r.stdout + r.stderr
    return folder, ''


def replay(exe, files, ptx_folder=None):
    """(passed, the line that says why not)"""
    env = dict(os.environ, WEX_FUZZ_GPU_PTX=ptx_folder) if ptx_folder else None
    r = subprocess.run([exe] + files, capture_output=True, text=True, cwd=ROOT, env=env)
    lines = (r.stdout + r.stderr).splitlines()
    why = [l for l in lines if 'ORACLE FAILED' in l] or lines[-1:] or ['(no output)']
    for prefix in (ROOT + os.sep, ROOT.replace(os.sep, '/') + '/'):   # the paths __FILE__ printed, made relative
        why[0] = why[0].replace(prefix, '')
    return r.returncode == 0, why[0].replace('\\', '/')


def one(index, change, files):
    exe, error = build(index, change)
    if not exe:
        return 'error', error.strip().splitlines()[-1] if error.strip() else 'build failed'
    passed, why = replay(exe, files)
    shutil.rmtree(os.path.join(OUT, 'negative', str(index)), ignore_errors=True)
    return ('passed' if passed else 'failed'), why


def one_gpu(index, change, exe, files):
    folder, error = ptx(index, change)
    if not folder:
        return 'error', error.strip().splitlines()[-1] if error.strip() else 'PTX build failed'
    passed, why = replay(exe, files, folder)
    shutil.rmtree(folder, ignore_errors=True)
    if passed:
        return 'passed', why
    return ('failed' if why.startswith('ORACLE FAILED: GPU') else 'failed elsewhere'), why


def main_gpu(files, jobs):
    exe, error = build('harness', None, gpu=True)
    if not exe:
        fail(f'the harness with -DWEX_FUZZ_GPU cannot be built: {error.strip()[-400:]}')
    work = [(0, None)] + [(i, c) for i, c in enumerate(CONTROLS, 1) if c[1] == CUH]
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(lambda item: one_gpu(item[0], item[1], exe, files), work))
    state, why = results[0]
    if state != 'passed':
        fail(f'the unchanged PTX does not pass: {why}')
    noticed = required = required_noticed = 0
    broken = []
    for (index, change), (state, why) in zip(work[1:], results[1:]):
        race = change[0] in GPU_RACES
        required += not race
        if state == 'error':
            broken.append(change[0])
            print(f'control {index} "{change[0]}": cannot be built ({why})')
        elif state == 'failed':
            noticed += 1
            required_noticed += not race
            print(f'control {index} "{change[0]}": noticed on the GPU: {why}')
        else:
            print(f'control {index} "{change[0]}": NOT noticed on the GPU' + (f' ({why})' if state != 'passed' else '') +
                  (' (a race: not required)' if race else ''))
    if broken:
        fail('the source changed under ' + '; '.join(broken) + ': check the list of controls against it')
    print(f'negative control on the GPU: {noticed} of {len(work) - 1} broken copies noticed ({required_noticed} of the '
          f'{required} that are not races; the {len(CONTROLS) - len(work) + 1} controls of the bridge do not apply)')
    sys.exit(0 if required_noticed == required else 1)


def main():
    if sys.argv[1:] not in ([], ['--gpu']):
        fail('usage: python -I wex4/fuzz/cuda_negative.py [--gpu]')
    files = seeds()
    jobs = int(os.environ.get('WEX_NEG_JOBS') or min(4, os.cpu_count() or 1))
    if sys.argv[1:] == ['--gpu']:
        main_gpu(files, jobs)
    work = [(0, None)] + list(enumerate(CONTROLS, 1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(lambda item: one(item[0], item[1], files), work))
    state, why = results[0]
    if state != 'passed':
        fail(f'the unchanged copy does not pass: {why}')
    noticed, broken = 0, []
    for (index, change), (state, why) in zip(work[1:], results[1:]):
        if state == 'error':
            broken.append(change[0])
            print(f'control {index} "{change[0]}": cannot be built ({why})')
        elif state == 'failed':
            noticed += 1
            print(f'control {index} "{change[0]}": noticed: {why}')
        else:
            print(f'control {index} "{change[0]}": NOT noticed')
    if broken:
        fail('the source changed under ' + '; '.join(broken) + ': check the list of controls against it')
    print(f'negative control: {noticed} of {len(CONTROLS)} broken copies noticed')
    sys.exit(0 if noticed == len(CONTROLS) else 1)


if __name__ == '__main__':
    main()
