"""GPU mutation check: each mutant removes one WNR3 check from wex_cuda.cuh in a fresh copy of the worktree
src, builds the helper + test_wex_nr3_gpu.exe (build_helper.cmd SRC OUT) and records the harness result,
which must FAIL. Running the harnesses needs heavy.lock (run_gpu_mutants.sh).
usage: python -I gpu_mutants.py WORKTREE_C_DIR OUT_ROOT"""
import os, shutil, subprocess, sys
SRC, ROOT = sys.argv[1], sys.argv[2]
CUH = 'wex_cuda.cuh'
MUTANTS = [
    ('m1_weight_hash', "    else if (state != expect) atomicOr(error, plan.nr3 ? WEX_CUDA_ERR_HASH : WEX_CUDA_ERR_STRUCT);\n}\n\n/* The WNM2 blob",
     "    else if (!plan.nr3 && state != expect) atomicOr(error, WEX_CUDA_ERR_STRUCT);\n}\n\n/* The WNM2 blob"),
    ('m2_scale_hash', "    else if (state != expect) atomicOr(error, plan.nr3 ? WEX_CUDA_ERR_HASH : WEX_CUDA_ERR_STRUCT);\n}\n\n/* Each halfword",
     "    else if (!plan.nr3 && state != expect) atomicOr(error, WEX_CUDA_ERR_STRUCT);\n}\n\n/* Each halfword"),
    ('m3_cz_check', 'atomicOr(error, WEX_CUDA_ERR_CZ); return;', '(void)0;'),
    ('m4_ceil_zero', 'if (!scale) { atomicOr(error, WEX_CUDA_ERR_CEIL); return; }', ''),
]
env = dict(os.environ, PATH=r'C:\Program Files (x86)\Microsoft Visual Studio\Installer;' + os.environ['PATH'])
for name, old, new in MUTANTS:
    work, out = os.path.join(ROOT, name, 'src'), os.path.join(ROOT, name, 'build')
    shutil.rmtree(os.path.join(ROOT, name), ignore_errors=True)
    shutil.copytree(SRC, work, ignore=shutil.ignore_patterns('*.o', '*.exe', '*.d'))
    os.makedirs(out)
    p = os.path.join(work, CUH)
    t = open(p, encoding='utf-8', newline='').read()
    t = t.replace('\r\n', '\n')
    assert t.count(old) == 1, (name, t.count(old))
    open(p, 'w', encoding='utf-8', newline='\n').write(t.replace(old, new))
    rc = subprocess.run(['cmd', '/c', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build_helper.cmd'), work, out],
                        env=env, capture_output=True, text=True).returncode
    print(name, 'built' if rc == 0 else f'BUILD FAILED {rc}', flush=True)
