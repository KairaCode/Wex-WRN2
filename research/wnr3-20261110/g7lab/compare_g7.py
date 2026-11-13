"""G7-lab verdict (run_g7_lab.sh). usage: python -I compare_g7.py DUMPDIR A1 B1 A2 B2 M1

PASS  : every arm's run is valid, A1 == A2 byte for byte, and every B/M arm == A1 byte for byte
        (first differing logits row reported otherwise), with equal texts per prompt.
INCONCLUSIVE: A1 != A2 (the engine is not deterministic in this configuration).
Validity per log: WEX1 container active, 40 dense layer uploads, decode on the GPU pipeline, no
cpu-union fallback, no 'DeepGEMM routed experts active' (the masked dg_moe path is never taken),
V4_LAB_CANON_ZERO ON exactly in the B/M arms with experts > 0."""
import json
import re
import sys
from pathlib import Path

VOCAB = 129280
RUNS = Path(__file__).resolve().parent / 'runs'


def check_log(name, canon):
    log = (RUNS / f'g7-{name}.stderr.log').read_text('utf-8', 'replace')
    counts = [int(x) for x in re.findall(r'v4_lab_canon_zero: experts=(\d+)', log)]
    problems = []
    if 'Container v1 active: U:\\kaira_wex\\v41-reap\\v4_flash_experts.wex' not in log:
        problems.append('WEX1 container not active')
    if len(re.findall(r'v4_gpu layer=\d+ uploaded=', log)) < 40:
        problems.append('fewer than 40 dense uploads')
    if 'v41_moe decode=gpu-pipeline' not in log:
        problems.append('decode not on the GPU pipeline')
    if 'v41_moe decode=cpu-union fallback' in log:
        problems.append('cpu-union fallback in decode')
    if 'DeepGEMM routed experts active' in log:
        problems.append('masked/contiguous dg_moe route path taken')
    on = 'v4_lab_canon_zero: ON' in log
    if on != canon or (canon and not counts):
        problems.append(f'canon switch state {on}, counts {counts[-1:]}')
    return problems, (counts[-1] if counts else 0)


def texts(name):
    return {r['id']: r.get('text') for r in map(json.loads, open(RUNS / f'g7-{name}.jsonl', encoding='utf-8'))
            if 'id' in r}


def first_diff(a, b):
    row = VOCAB * 4
    n = min(len(a), len(b)) // row
    for i in range(n):
        if a[i * row:(i + 1) * row] != b[i * row:(i + 1) * row]:
            return i
    return None if len(a) == len(b) else n


def main():
    dump, arms = Path(sys.argv[1]), sys.argv[2:]
    data, ok = {}, True
    for a in arms:
        problems, experts = check_log(a, a[0] in 'BM')
        data[a] = (dump / f'g7-{a}.logits').read_bytes()
        print(f'{a}: {len(data[a]) // (VOCAB * 4)} logits rows, canon experts {experts}, '
              f'{"VALID" if not problems else "INVALID: " + "; ".join(problems)}')
        ok &= not problems
    base = arms[0]
    same_aa = all(data[a] == data[base] for a in arms if a[0] == 'A')
    print(f'A vs A byte-identical: {same_aa}')
    verdict = 'PASS' if ok and same_aa else ('INCONCLUSIVE' if ok else 'INVALID')
    for a in arms:
        if a[0] == 'A':
            continue
        d = first_diff(data[base], data[a])
        same_text = texts(a) == texts(base)
        print(f'{a} vs {base}: {"byte-identical" if d is None else f"FIRST DIFFERENCE at logits row {d}"}, '
              f'texts equal: {same_text}')
        if d is not None or not same_text:
            verdict = 'FAIL' if verdict == 'PASS' else verdict
    print('VERDICT', verdict)


if __name__ == '__main__':
    main()
