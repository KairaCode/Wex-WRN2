"""Relative read speed of two drives with identical fresh test files: random
4.69 MiB records (a Qwen3.8 expert), unbuffered, THREADS workers; then both
drives at once with records alternated (the striping a mirrored container gets).
Both files are written the same way just before, so drive caches favour both
alike: compare the drives with each other, not with cold shard reads.

usage: python -I drive_probe.py FILE_A FILE_B [gib=8] [records=600] [threads=8] [OUT.json]
Deletes both test files at the end."""
import json
import os
import random
import sys
import time
from pathlib import Path

A, B = Path(sys.argv[1]), Path(sys.argv[2])
GIB = int(sys.argv[3]) if len(sys.argv) > 3 else 8
RECORDS = int(sys.argv[4]) if len(sys.argv) > 4 else 600
THREADS = int(sys.argv[5]) if len(sys.argv) > 5 else 8
OUT = Path(sys.argv[6]) if len(sys.argv) > 6 else None
RECORD = 4915200          # 1200 pages

sys.argv = [sys.argv[0], '.', '1', str(THREADS)]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_q38_experts as probe   # noqa: E402


def write(path):
    block = os.urandom(64 << 20)
    started = time.perf_counter()
    with open(path, 'wb', buffering=0) as f:
        for _ in range(GIB * 16):
            f.write(block)
        os.fsync(f.fileno())
    return round(GIB / (time.perf_counter() - started), 3)


def side(paths, rng):
    slots = (GIB << 30) // RECORD
    plan = []
    for i in range(RECORDS):
        path = paths[i % len(paths)]
        plan.append((i, [(str(path), rng.randrange(slots) * RECORD, RECORD)]))
    probe.MODEL = Path('')
    seconds, moved, _ = probe.run_side(plan, True)
    return {'gib_s': round(moved / seconds / 2**30, 3), 'ms_per_record_wall': round(1000 * seconds / RECORDS, 3)}


def main():
    report = {'a': str(A), 'b': str(B), 'gib': GIB, 'records': RECORDS, 'threads': THREADS}
    try:
        report['write_gib_s'] = {'a': write(A), 'b': write(B)}
        rng = random.Random(5)
        report['a_only'] = side([A], rng)
        report['b_only'] = side([B], rng)
        report['striped'] = side([A, B], rng)
    finally:
        for p in (A, B):
            p.unlink(missing_ok=True)
    text = json.dumps(report, indent=1)
    print(text)
    if OUT:
        OUT.write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
