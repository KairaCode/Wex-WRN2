"""Mutants for the review fixes (2026-11-10): each must make test_v4_wex_nr3 (engine store)
or test_expert_store_wex (standalone store) fail. Run inside a fresh workspace copy:
python -I fix_mutants.py SRC_C_DIR"""
import os, subprocess, sys
src = sys.argv[1]
M = [
 ('deepseek_v4.c', 'coli_v4_wex_crc_done(f->store, slot, status != -1);', 'coli_v4_wex_crc_done(f->store, slot, status == 0);', 'tests/test_v4_wex_nr3.exe'),
 ('wex_format.h', '!(record[27] & WEX_NR3_FLAG_CZ) == !(header_flags & WEX_HEADER_CANONICAL_ZERO)', '1', 'tests/test_v4_wex_nr3.exe'),
 ('wex_decode.c', 'if (masked && atoi(masked) != 0) {', 'if (0) {', 'tests/test_v4_wex_nr3.exe'),
 ('expert_store_wex.c', 'if (coli_wex_container_refused(header_flags, "wex")) {', 'if (0) {', 'tests/test_expert_store_wex.exe'),
 ('deepseek_v4.c', 'wex_nr3_owner_ok(entry, slot->slab + read_offset, (uint32_t)record->layer,\n                                 (uint32_t)record->expert, state->wex_flags) &&',
  '1 &&', 'tests/test_v4_wex_nr3.exe'),
 ('deepseek_v4.c', '        if (slot >= 0) coli_v4_wex_crc_done(f->store, slot, 0);\n        return -1;', '        if (slot >= 0) coli_v4_wex_crc_done(f->store, slot, 1);\n        return -1;', 'tests/test_v4_wex_nr3.exe'),
]
env = dict(os.environ)
for i, (f, old, new, test) in enumerate(M):
    p = os.path.join(src, f)
    raw = open(p, 'rb').read()
    crlf = b'\r\n' in raw
    s = raw.decode().replace('\r\n', '\n')
    assert s.count(old) == 1, (f, old)
    m = s.replace(old, new)
    open(p, 'wb').write((m.replace('\n', '\r\n') if crlf else m).encode())
    try:
        b = subprocess.run(['make', '-j8', test], cwd=src, capture_output=True, text=True)
        if b.returncode:
            print(f'mutant {i} ({f}): BUILD FAILED', b.stderr[-400:]); continue
        r = subprocess.run([os.path.join(src, test)], cwd=src, capture_output=True, text=True, timeout=600)
        print(f'mutant {i} ({f}: {old[:50]!r}): rc={r.returncode} ->', 'KILLED' if r.returncode else 'SURVIVED', flush=True)
    finally:
        open(p, 'wb').write(raw)
