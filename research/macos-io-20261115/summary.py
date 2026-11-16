# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""The tables of README.md from the JSON files of bench/wex_io.py (Python 3 stdlib only).

  python3 -I summary.py run1-wex_io.json run2-wex_io.json run3-wex_io.json
  python3 -I summary.py r5-run1-wex_io.json r5-run2-wex_io.json r5-run3-wex_io.json   (the earlier set)

1. Per read path and file: MB/s median (min to max) over the passes, against row 1, median ms per record, and the
   shares that mincore found (cached_after, ahead_cached).
2. The order of the passes. wex_io.py times the paths that passed their check in the order of its table, rotated by
   one row from round to round (a round is one timed pass of every path, --runs of them per file; round r, counted
   from 0, starts at row r of that list), and every pass of a round reads the same records (seed r + 1). So the first
   pass of a round is the only one whose records no earlier pass of that round had read. Printed: per round, the first
   pass, its MB/s and where it ranks in its row; then the MB/s of all passes by their position in the round (1 =
   first), over all files, and which rows held each position.
3. Each row again without the first pass of each round (the passes at positions 2 and later only).
4. Per row and file: in how many rounds the row read faster than row 1 in the same round (the two sit at
   different positions of that round; with --runs equal to the number of rows every row takes every position once).
5. Over all files: each row's median record time (median, min and max over its passes), how many passes had what
   share in the cache (cached_before, cached_after, ahead_cached), and how long a cold pass lasted."""
import json
import statistics
import sys


def timed_order(res):
    return [opt for opt, verdict in res['check'].items() if verdict.endswith('match their SHA-256')]


def fmt(v):
    return f'{statistics.median(v):.0f} ({min(v):.0f} to {max(v):.0f})'


def main():
    files = [(path, json.load(open(path))) for path in sys.argv[1:]]
    print('== 1. per read path: MB/s median (min to max), against row 1, ms per record (median), cached_after max, '
          'ahead_cached min to max')
    for path, res in files:
        s1 = res['summary'].get('path=sync direct=0', {}).get('mb_s_median')
        print(f'-- {path}: {res["runs"]} rounds, {res["records_per_pass"]} records per pass, drop_caches {res["drop_caches"]}')
        for opt, s in res['summary'].items():
            passes = res['passes'][opt]
            ahead = [p['ahead_cached'] for p in passes if p.get('ahead')]
            print(f'{s["label"]:46s} {fmt([p["mb_s"] for p in passes]):24s} {s["mb_s_median"] / s1:5.2f}x '
                  f'{s["record_ms_p50_median"]:6.1f} ms  after {max(p["cached_after"] for p in passes):.4f}'
                  + (f'  ahead {min(ahead):.4f} to {max(ahead):.4f}' if ahead else ''))
    print('== 2. the first pass of each round')
    by_pos, rows_at, firsts, slowest, two = {}, {}, 0, 0, 0
    for path, res in files:
        todo = timed_order(res)
        label = {opt: s['label'] for opt, s in res['summary'].items()}
        for run in range(res['runs']):
            order = todo[run % len(todo):] + todo[:run % len(todo)]
            for pos, opt in enumerate(order, 1):
                by_pos.setdefault(pos, []).append(res['passes'][opt][run]['mb_s'])
                rows_at.setdefault(pos, set()).add(todo.index(opt) + 1)
            first = order[0]
            mine = res['passes'][first][run]['mb_s']
            others = [p['mb_s'] for k, p in enumerate(res['passes'][first]) if k != run]
            rank = 1 + sum(o < mine for o in others)
            firsts += 1
            slowest += rank == 1
            two += rank <= 2
            print(f'{path} round {run + 1}: {label[first]}: {mine:.0f} MB/s, {res["passes"][first][run]["record_ms_p50"]:.1f} ms '
                  f'per record; the other {len(others)} passes of that row {fmt(others)}; rank {rank} of {len(others) + 1} '
                  f'(1 = slowest)')
    print(f'the first pass was the slowest of its row in {slowest} of {firsts} rounds, one of the two slowest in {two}')
    for pos in sorted(by_pos):
        print(f'position {pos:2d}: {len(by_pos[pos])} passes, MB/s {fmt(by_pos[pos])}; rows {sorted(rows_at[pos])}')
    print('== 3. per read path without the first pass of each round: MB/s median (min to max), against row 1')
    for path, res in files:
        todo = timed_order(res)
        first = {todo[run % len(todo)]: [] for run in range(res['runs'])}
        for run in range(res['runs']):
            first[todo[run % len(todo)]].append(run)
        later = {opt: [p['mb_s'] for k, p in enumerate(res['passes'][opt]) if k not in first.get(opt, [])] for opt in todo}
        base = statistics.median(later['path=sync direct=0'])
        print(f'-- {path}')
        for opt in todo:
            print(f'{res["summary"][opt]["label"]:46s} {len(later[opt])} passes {fmt(later[opt]):24s} '
                  f'{statistics.median(later[opt]) / base:5.2f}x')
    print('== 4. per read path: in how many rounds it was faster than row 1 of the same round')
    for path, res in files:
        todo = timed_order(res)
        base = [p['mb_s'] for p in res['passes']['path=sync direct=0']]
        print(f'-- {path}')
        for opt in todo[1:]:
            won = sum(p['mb_s'] > b for p, b in zip(res['passes'][opt], base))
            print(f'{res["summary"][opt]["label"]:46s} {won} of {len(base)} rounds')
    print('== 5. over all files')
    ms, count, seconds = {}, {}, []
    for _, res in files:
        for opt, passes in res['passes'].items():
            for p in passes:
                ms.setdefault(res['summary'][opt]['label'], []).append(p['record_ms_p50'])
                kind = 'warm' if p['warm'] else 'cold'
                if not p['warm']:
                    seconds.append(p['seconds'])
                for key, what, want in (('before', kind, 1.0 if p['warm'] else 0.0),
                                        ('after', 'F_NOCACHE' if p['direct'] else 'page cache', 0.0 if p['direct'] else 1.0),
                                        ('ahead_cached', 'announcing', 1.0)):
                    if key != 'ahead_cached' or p['ahead']:
                        got = p['cached_' + key] if key != 'ahead_cached' else p['ahead_cached']
                        n = count.setdefault((key, what, want), [0, 0])
                        n[0] += got == want
                        n[1] += 1
    for label, v in ms.items():
        print(f'{label:46s} ms per record over {len(v)} passes: {statistics.median(v):.1f} ({min(v):.1f} to {max(v):.1f})')
    for (key, what, want), (hit, all_) in count.items():
        print(f'{what} passes with {key} {want * 100:.0f}%: {hit} of {all_}')
    print(f'cold passes: {len(seconds)}, {min(seconds):.2f} to {max(seconds):.2f} s each')


if __name__ == '__main__':
    main()
