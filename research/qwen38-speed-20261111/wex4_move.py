"""Move the real qwen38 expert containers (WEX1, U: + C: mirror) to WEX4, in place, under lab's heavy.lock.
Order: mirror first, engine test on the mirror alone, then the primary, engine test on both.
Any failure: downgrade what was upgraded and stop. Usage: python -I wex4_move.py TOOLS_DIR"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = sys.argv[1]
KAIRA = r'<KAIRA>'
LOCK = KAIRA + r'\lab\v41-gpu-20261107\heavy.lock'
EXE = KAIRA + r'\app\colibri-q38mx\qwen38.exe'
U = r'U:\kaira_wex\qwen38-mxfp4\q38_experts.wex'
C = r'C:\kaira_wex\qwen38-mxfp4\q38_experts.wex'
WEX1_SIZE = 64224956416
PY = r'C:\Python314\python.exe'


def say(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)


def tool(script, *args):
    r = subprocess.run([PY, '-I', os.path.join(TOOLS, script), *args], capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    say(script, ' '.join(args), '-> rc', r.returncode); print(out[-1500:], flush=True)
    return r.returncode, out


def busy():
    t = subprocess.run('tasklist', capture_output=True, text=True).stdout.lower()
    return [n for n in ('qwen38.exe', 'qwen36.exe', 'deepseek_v4.exe', 'kaira.exe') if n in t]


def engine_test(tag, wex, mirror):
    """One harness run (5 requests, temperature 0, no lock handling: we hold it). Returns (ok, why)."""
    env = dict(os.environ, VERIFY_NEW_EXE=EXE, VERIFY_RAM_GB_MAX='13', VERIFY_SET='QT_ARENA=1', VERIFY_TAG=tag,
               COLI_WEX_PATH=wex, COLI_WEX_WEIGHTS='38,62')
    env.pop('VERIFY_TEMP', None)
    if mirror: env['COLI_WEX_EXTRA_MIRRORS'] = mirror
    else: env.pop('COLI_WEX_EXTRA_MIRRORS', None)
    r = subprocess.run([PY, '-I', os.path.join(HERE, 'verify_gateway_t.py'), 'run', 'new'], env=env, capture_output=True, text=True)
    base = os.path.join(HERE, 'verify-' + tag + 'new')
    try: rec = json.load(open(base + '.json', encoding='utf-8'))
    except OSError: return False, 'no result json (rc %d): %s' % (r.returncode, (r.stdout + r.stderr)[-600:])
    err = open(base + '.stderr.log', encoding='utf-8', errors='replace').read()
    reqs = rec.get('requests', [])
    for q in reqs: say('  ', q.get('id'), q.get('completion_tokens'), 'tok', '%.2f tok/s' % (q.get('decode_tok_s') or 0), repr((q.get('text') or '')[:70]))
    marks = {m: (m in err) for m in ('WEX4 container', 'WEX mirror active', 'warmstart done')}
    say('   marks', marks, 'error', rec.get('error'), 'kill', rec.get('kill'))
    if rec.get('error') or rec.get('kill') or len(reqs) != 5 or any(q.get('error') or not q.get('text') for q in reqs):
        return False, 'run failed'
    if not marks['WEX4 container'] or not marks['warmstart done']: return False, 'engine did not report a WEX4 container on the GPU path'
    if mirror and not marks['WEX mirror active']: return False, 'mirror was not accepted'
    return True, ''


def main():
    end = time.time() + 45 * 60; tag = 'q38 WEX4 container upgrade ' + time.strftime('%Y-%m-%dT%H:%M:%S')
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, (tag + '\n').encode()); os.close(fd); break
        except FileExistsError:
            if time.time() > end: say('lock busy for 45 min: nothing done'); return 3
            say('lock busy, waiting'); time.sleep(30)
    say('lock taken')
    done = []
    try:
        for _ in range(20):
            if not busy(): break
            say('still running:', busy()); time.sleep(15)
        else:
            say('an engine or KAIRA is running: nothing done'); return 3
        for f in (U, C):
            if os.path.getsize(f) != WEX1_SIZE: say(f, 'is not the WEX1 file (size %d): nothing done' % os.path.getsize(f)); return 2
        roots = {}
        for f, test in ((C, ('wex4c-', C, None)), (U, ('wex4uc-', U, os.path.dirname(C)))):
            rc, _ = tool('wex4_convert.py', 'upgrade', f, '--names', 'qwen38')
            done.append(f)
            if rc: raise RuntimeError('upgrade failed: ' + f)
            rc, info = tool('wex4.py', 'info', f)
            if rc: raise RuntimeError('info failed: ' + f)
            roots[f] = [l for l in info.splitlines() if 'root' in l.lower()]
            say('size', os.path.getsize(f), 'root', roots[f])
            if f == U and (roots[U] != roots[C] or os.path.getsize(U) != os.path.getsize(C)): raise RuntimeError('the two copies differ after the upgrade')
            ok, why = engine_test(*test)
            if not ok: raise RuntimeError('engine test %s: %s' % (test[0], why))
        say('DONE: both copies are WEX4,', os.path.getsize(U), 'bytes, root', roots[U])
        return 0
    except Exception as e:
        say('FAILED:', e, '-> downgrading', done)
        for f in done:
            tool('wex4_convert.py', 'downgrade', f); say('size now', os.path.getsize(f), '(WEX1 is %d)' % WEX1_SIZE)
        return 1
    finally:
        try:
            if open(LOCK, encoding='utf-8').read().strip() == tag: os.remove(LOCK); say('lock released')
        except OSError: pass


sys.exit(main())
