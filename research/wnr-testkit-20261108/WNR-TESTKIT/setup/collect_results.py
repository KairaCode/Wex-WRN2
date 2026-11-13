"""Merge every results/*/summary.json (and the run_all.ps1 run log) into results/ALL_RESULTS.json.
Workflow outputs are merged too: results/wf-*/team*_summary.json (into 'summaries'; DONE when it has no status
field) and results/wf-*/<member>/member_result.json (into 'workflow_members').

Usage: python -I setup\\collect_results.py [--root <kit root>] [--results <results dir>] [--runs <runs.jsonl>]
Prints a table (one line per test) and 'RESULT collect_results DONE <counts>'.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kitenv  # noqa: E402


def load_json(path):
    try:
        with open(path, encoding='utf-8-sig') as fh:
            return json.load(fh)
    except Exception as e:  # noqa: BLE001
        return {'_error': f'cannot parse {os.path.basename(path)}: {type(e).__name__}: {e}'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=kitenv.default_root(__file__))
    ap.add_argument('--results', default=None)
    ap.add_argument('--runs', default=None)
    ap.add_argument('--out', default=None, help='output file (default <results>/ALL_RESULTS.json)')
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    res = os.path.abspath(a.results or os.path.join(root, 'results'))
    runs_path = a.runs or os.path.join(res, '_runlogs', 'runs.jsonl')
    runs = []
    if os.path.exists(runs_path):
        for line in open(runs_path, encoding='utf-8-sig', errors='replace'):
            line = line.strip().lstrip('﻿')
            if line:
                try:
                    runs.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    runs.append({'_unparsed': line[:500]})
    summaries = {}
    wf_members = {}
    if os.path.isdir(res):
        for dp, dns, fns in os.walk(res):
            reld = os.path.relpath(dp, res)
            depth = reld.count(os.sep)
            dns[:] = [d for d in dns if not d.startswith('_')] if depth < 2 else []
            if os.path.abspath(dp) == res:
                continue
            top = reld.replace('\\', '/').split('/')[0]
            for fn in fns:
                is_wf_team = top.startswith('wf-') and fn.startswith('team') and fn.endswith('_summary.json')
                if fn == 'summary.json' or is_wf_team:
                    rel = os.path.relpath(os.path.join(dp, fn), res).replace('\\', '/')
                    s = load_json(os.path.join(dp, fn))
                    if is_wf_team and isinstance(s, dict) and 'status' not in s and 'result' not in s:
                        s = dict(s, status='DONE', _note='workflow critic summary (no status field; DONE by convention)')
                    summaries[rel] = s
                elif top.startswith('wf-') and fn == 'member_result.json':
                    rel = os.path.relpath(os.path.join(dp, fn), res).replace('\\', '/')
                    wf_members[rel] = load_json(os.path.join(dp, fn))
    counts = {}
    rows = []
    for rel, s in sorted(summaries.items()):
        st = str(s.get('status', s.get('result', '?'))) if isinstance(s, dict) else '?'
        counts[st] = counts.get(st, 0) + 1
        rows.append((rel, st, s.get('seconds', '') if isinstance(s, dict) else ''))
    env = None
    ce = summaries.get('check_env/summary.json')
    if isinstance(ce, dict):
        env = ce.get('environment')
    if env is None:
        env = kitenv.env_info(with_gpu=True)
    allr = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'), 'results_dir': res, 'environment': env,
            'status_counts': counts, 'runs': runs, 'summaries': summaries, 'workflow_members': wf_members}
    outp = a.out or os.path.join(res, 'ALL_RESULTS.json')
    kitenv.write_json(outp, allr)
    print(f'{"summary":60s} {"status":8s} seconds')
    for rel, st, sec in rows:
        print(f'{rel:60s} {st:8s} {sec}')
    bad = [r for r in runs if isinstance(r, dict) and (r.get('timed_out') or (r.get('exit_code') not in (0, None)))]
    for r in bad:
        print(f"run problem: {r.get('name')} exit={r.get('exit_code')} timed_out={r.get('timed_out')} "
              f"result={r.get('result_line')}")
    print(f'wrote {outp}')
    print(f'RESULT collect_results DONE summaries={len(summaries)} wf_members={len(wf_members)} runs={len(runs)} ' +
          ' '.join(f'{k}={v}' for k, v in sorted(counts.items())))
    return 0


if __name__ == '__main__':
    sys.exit(main())
