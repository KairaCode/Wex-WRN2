"""Compare the two arms of q38_ab_temp.sh: verify-q38samp-A-new.json (the installed build) against
verify-q38samp-B-new.json (e245f4c), both written by verify_gateway_t.py next to itself.

  python -I q38_ab_compare.py [A.json B.json]

Per request: tokens generated, TTFT, decode tok/s, B/A, and whether the generated text is the same (the engine
never seeds rand(): an equivalent sampler gives the same text at temperature 0.6). Reports only: exit code 0 always."""
import json
import os
import sys

D = os.path.dirname(os.path.abspath(__file__))   # the harness's LAB: it writes verify-<VERIFY_TAG>new.json there
IDS = ('p1-258', 'p2-2k', 'p3-turn1', 'p3-turn2', 'p3-turn3')


def first_diff(a, b):
    """Index of the first character that differs (the shorter length when one is a prefix), None when equal."""
    if a == b: return None
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


assert first_diff('abc', 'abc') is None and first_diff('abc', 'abd') == 2 and first_diff('ab', 'abc') == 2


def load(arm, path):
    try:
        return json.load(open(path, encoding='utf-8'))
    except (OSError, ValueError) as e:
        print('!! arm %s: no result (%s): %r' % (arm, path, e))
        return None


def num(r, key, fmt):
    v = r.get(key)
    return fmt % v if isinstance(v, (int, float)) else 'n/a'


def main():
    paths = sys.argv[1:3] if len(sys.argv) > 2 else [os.path.join(D, 'verify-q38samp-%s-new.json' % a) for a in 'AB']
    A, B = load('A', paths[0]), load('B', paths[1])
    for arm, rec in (('A', A), ('B', B)):
        if not rec: continue
        pk = rec.get('peaks') or {}
        print('%s  engine %s' % (arm, rec.get('engine')))
        print('   temperature %s | --ram %s | load (ready) %s s | peak VRAM %s MiB (%s before the run) | wall %s s'
              % (rec.get('temperature'), rec.get('ram_gb'), num(rec, 'ready_s', '%.1f'), pk.get('vram'),
                 rec.get('vram_start'), num(rec, 'wall', '%.0f')))
        bad = rec.get('error') or rec.get('kill') or [r['id'] for r in rec.get('requests', []) if r.get('error')]
        print('   %serror %s | kill %s | exit %s | alive at end %s | requests %d'
              % ('!! ' if bad else '', rec.get('error'), rec.get('kill'), rec.get('exit'), rec.get('alive_at_end'),
                 len(rec.get('requests', []))))
        for l in rec.get('stderr_marks', []):
            if any(k in l for k in ('RSS after load', 'warmstart done', 'prefix store', '[serve] context')):
                print('   | ' + l[:230])
    if not A or not B: return
    for key, what in (('temperature', 'temperature'), ('ram_gb', '--ram')):
        if A.get(key) != B.get(key):
            print('!! the arms ran with a different %s (A %s, B %s): not a fair A/B' % (what, A.get(key), B.get(key)))
    if not A.get('temperature'): print('!! temperature %s: the sampler cost does not show at 0' % A.get('temperature'))
    if A.get('engine') == B.get('engine'): print('!! both arms ran the same engine file')

    ra = {r.get('id'): r for r in A.get('requests', [])}
    rb = {r.get('id'): r for r in B.get('requests', [])}
    print('\n%-9s %9s %15s %17s %9s  %s' % ('request', 'tokens', 'TTFT s', 'decode tok/s', 'B/A', 'text'))
    print('%-9s %9s %15s %17s' % ('', 'A / B', 'A / B', 'A / B'))
    diffs = []
    for rid in IDS:
        a, b = ra.get(rid), rb.get(rid)
        if not a or not b:
            print('%-9s !! missing in arm %s' % (rid, ' and '.join(n for n, r in (('A', a), ('B', b)) if not r)))
            continue
        ta, tb = a.get('decode_tok_s'), b.get('decode_tok_s')
        ratio = '%.1f%%' % (100 * tb / ta) if ta and tb else 'n/a'
        at = first_diff(a.get('text') or '', b.get('text') or '')
        if at is None and first_diff(a.get('reasoning') or '', b.get('reasoning') or '') is not None: at = -1
        if at is not None: diffs.append((rid, at, a, b))
        print('%-9s %9s %15s %17s %9s  %s' % (
            rid, '%s / %s' % (a.get('completion_tokens'), b.get('completion_tokens')),
            '%s / %s' % (num(a, 'ttft_s', '%.1f'), num(b, 'ttft_s', '%.1f')),
            '%s / %s' % (num(a, 'decode_tok_s', '%.2f'), num(b, 'decode_tok_s', '%.2f')), ratio,
            'identical' if at is None else '!!!! DIFFERENT !!!!'))
        for arm, r in (('A', a), ('B', b)):
            if r.get('error'): print('%-9s !! arm %s error: %s' % ('', arm, r['error']))
    if not diffs:
        print('\ntext: identical in every request compared (same rand() stream, same tokens drawn)')
        return
    print('\n' + '!' * 100)
    print('!!!! TEXT DIFFERS between A and B in %d request(s): the samplers did not draw the same tokens' % len(diffs))
    print('!!!! only the first one is evidence: later requests share the rand() stream and the chat history')
    for rid, at, a, b in diffs:
        if at < 0:
            print('!!!! %s: the text is equal, the reasoning differs' % rid)
            continue
        print('!!!! %s: first difference at character %d (A %d chars, B %d chars)'
              % (rid, at, len(a.get('text') or ''), len(b.get('text') or '')))
        for arm, r in (('A', a), ('B', b)):
            print('     %s: ...%r' % (arm, (r.get('text') or '')[max(0, at - 60):at + 60]))
    print('!' * 100)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:   # noqa: BLE001 -- a report, never a gate
        print('!! compare stopped: %r' % e)
    sys.exit(0)
