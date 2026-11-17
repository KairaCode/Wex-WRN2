# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Run nvwnr.c on the slabs of analyze.py, cross-validated over two folds of layers, and collect the results.

  python -I run_proto.py NVWNR_EXE DATA_DIR SLAB_DIR OUT.json [RUNS] [--only CHECKPOINT[,...]] [--configs NAME[,...]]
                         [--in-sample]

Fold A = the experts of the 1st, 3rd, 5th and 7th chosen layer of fetch.py's pair, fold B = the others (the
folds of wnr3_real.py). Every configuration is run twice: model trained on A, tested on B, and the other way
round; a checkpoint's size is the sum over its 16 experts of the record each got from the other fold's model.
RUNS (default 1) timed decodes per record after one that is not timed; the decode speed of a configuration is
raw bytes over the summed median times of both runs. --in-sample trains once on all 16 experts and tests the same
16 (the result is stored under "<configuration> (in-sample)"). Standard library only.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fetch  # noqa: E402

CONFIGS = {   # name -> (formats it applies to, nvwnr options)
    'mx-wnr3': ('mxfp4', ['--wctx', 'srel7', '--sctx', 'o0', '--ceil']),
    'mx-wnr3-cz': ('mxfp4', ['--wctx', 'srel7', '--sctx', 'o0', '--ceil', '--cz']),
    'nv-exp7-o0': ('nvfp4', ['--wctx', 'exp7', '--sctx', 'o0']),
    'nv-exp7-o0-cz': ('nvfp4', ['--wctx', 'exp7', '--sctx', 'o0', '--cz']),
    'nv-exp7-o1-cz': ('nvfp4', ['--wctx', 'exp7', '--sctx', 'o1', '--cz']),
    'nv-exp7-o1p-cz': ('nvfp4', ['--wctx', 'exp7', '--sctx', 'o1p', '--cz']),
    'nv-code12-o1-cz': ('nvfp4', ['--wctx', 'code12', '--sctx', 'o1', '--cz']),
    'nv-exp7-o1-cz-ceil': ('nvfp4', ['--wctx', 'exp7', '--sctx', 'o1', '--cz', '--ceil']),
    'nv-exp7-o1': ('nvfp4', ['--wctx', 'exp7', '--sctx', 'o1']),
    'nv-code6-o1-cz': ('nvfp4', ['--wctx', 'code6', '--sctx', 'o1', '--cz']),
    'nv-code24-o1-cz': ('nvfp4', ['--wctx', 'code24', '--sctx', 'o1', '--cz']),
    'nv-code12-o1': ('nvfp4', ['--wctx', 'code12', '--sctx', 'o1']),
}


def main(argv):
    exe, data, slabs, out = argv[1:5]
    runs = int(argv[5]) if len(argv) > 5 and not argv[5].startswith('--') else 1
    only = argv[argv.index('--only') + 1].split(',') if '--only' in argv else None
    pick = argv[argv.index('--configs') + 1].split(',') if '--configs' in argv else list(CONFIGS)
    log = json.load(open(os.path.join(data, 'fetch-log.json')))
    res = json.load(open(out)) if os.path.exists(out) else {}
    for pair, ckpts in fetch.PAIRS.items():
        layers = sorted({L for L, _ in log['pairs'][pair]['chosen']})
        fold = {L: 'A' if i % 2 == 0 else 'B' for i, L in enumerate(layers)}
        for c in ckpts:
            if (only and c not in only) or not os.path.isdir(os.path.join(slabs, c)):
                continue
            fmt = fetch.CHECKPOINTS[c][2]
            files = {f: fold[int(f[1:].split('_')[0])] for f in os.listdir(os.path.join(slabs, c)) if f.endswith('.nvs')}
            for name in pick:
                cfmt, opts = CONFIGS[name]
                if cfmt != fmt:
                    continue
                rows, ok = [], True
                insample = '--in-sample' in argv
                for train in (('all',) if insample else ('A', 'B')):
                    tr = [os.path.join(slabs, c, f) for f, g in sorted(files.items()) if insample or g == train]
                    te = [os.path.join(slabs, c, f) for f, g in sorted(files.items()) if insample or g != train]
                    p = subprocess.run([exe, str(runs)] + opts + ['--train'] + tr + ['--test'] + te,
                                       capture_output=True, text=True)
                    ok = ok and p.returncode == 0
                    for line in p.stdout.splitlines():
                        if line.startswith('json '):
                            r = json.loads(line[5:])
                            r['file'] = os.path.basename(r['file'])
                            r['trained_on'] = train
                            rows.append(r)
                    if p.returncode:
                        print(p.stderr, file=sys.stderr)
                raw = sum(r['raw'] for r in rows)
                summ = {k: sum(r[k] for r in rows) for k in ('raw', 'used', 'total', 'scale_payload', 'weight_payload',
                                                              'lengths', 'extra')}
                summ.update(ratio_used=summ['used'] / raw, ratio_total=summ['total'] / raw,
                            round_trips=sum(r['round_trip'] for r in rows), records=len(rows),
                            decode_mb_s=raw / sum(r['decode_ms_median'] for r in rows) / 1000.0,
                            options=opts, runs=runs, exit_ok=ok)
                name_out = name + (' (in-sample)' if insample else '')
                res.setdefault(c, {})[name_out] = {'summary': summ, 'rows': rows}
                print(c, name_out, json.dumps(summ), flush=True)
                with open(out, 'w', encoding='utf-8') as f:
                    json.dump(res, f, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
