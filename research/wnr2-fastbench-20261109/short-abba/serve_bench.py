"""M0 warm serve-mode bench: one engine process, prompts in sequence (protocol 3.1).

  python -I serve_bench.py EXE MODEL_DIR PROMPTS.jsonl OUT_PREFIX MAX_TOKENS id[,id...]

The engine runs with SERVE=1 SNAP=MODEL_DIR (the caller's environment passes through:
RAM_GB, DSV4_CUDA, COLI_WEX_PATH, ...). For each prompt it records TTFT (client clock,
SUBMIT to first DATA frame), the DONE line (decode tok/s from the first token, turn hit
rate), the PROF line and the cumulative TIERS bytes. stderr goes to OUT_PREFIX.stderr.log,
the per-prompt records to OUT_PREFIX.jsonl.
"""
import json
import os
import subprocess
import sys
import threading
import time

BOS, USER, ASSIST = '<｜begin▁of▁sentence｜>', '<｜User｜>', '<｜Assistant｜>'


def read_exact(f, n):
    """m0/serve_bench.py read DATA with one f.read(n): on the unbuffered pipe that can return short, and
    the readline after it then swallowed payload bytes (ab-samedrive A2 chat_it1 lost a newline)."""
    buf = b''
    while len(buf) < n:
        part = f.read(n - len(buf))
        if not part:
            break
        buf += part
    return buf


def main():
    exe, model, prompts_path, prefix, max_tok, ids = sys.argv[1:7]
    max_tok = int(max_tok)
    wanted = ids.split(',')
    prompts = {p['id']: p for p in map(json.loads, open(prompts_path, encoding='utf-8'))}
    env = dict(os.environ, SERVE='1', SNAP=model)
    errlog = open(prefix + '.stderr.log', 'w', encoding='utf-8')
    t_start = time.time()
    p = subprocess.Popen([exe], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, bufsize=0)
    with open(prefix + '.pid', 'w') as f:   # short_abba.sh's quiet-machine sampler: this engine is ours
        f.write(f'{p.pid}\n')

    def pump():
        for line in iter(p.stderr.readline, b''):
            errlog.write(f'[{time.time() - t_start:9.3f}] ' + line.decode('utf-8', 'replace'))
            errlog.flush()
    threading.Thread(target=pump, daemon=True).start()

    out = open(prefix + '.jsonl', 'w', encoding='utf-8')
    while True:
        line = p.stdout.readline()
        if not line:
            raise SystemExit('engine exited before READY; see ' + prefix + '.stderr.log')
        if b'READY' in line:
            break
    t_ready = time.time() - t_start
    out.write(json.dumps({'event': 'ready', 'secs': round(t_ready, 2)}) + '\n'); out.flush()
    print('READY after', round(t_ready, 1), 's', flush=True)

    for pid in wanted:
        pr = prompts[pid]
        n = min(max_tok, pr['max_tokens']) if pr.get('warmup') else max_tok
        payload = (BOS + USER + pr['user'] + ASSIST + '</think>').encode('utf-8')
        p.stdin.write(f'SUBMIT {pid} 0 {len(payload)} {n} 0 1\n'.encode() + payload + b'\n')
        p.stdin.flush()
        t0 = time.time(); t_first = None; chunks = []; rec = {'id': pid, 'max_tokens': n}
        while True:
            line = p.stdout.readline()
            if not line:
                raise SystemExit('engine closed mid-request')
            text = line.decode('latin-1').rstrip('\n')
            kind = text.split(' ', 1)[0]
            if kind == 'DATA':
                if t_first is None:
                    t_first = time.time()
                chunks.append(read_exact(p.stdout, int(text.split()[2])))
                p.stdout.readline()
            elif kind == 'DONE':
                rec['done'] = text; rec['t_done'] = time.time() - t0
            elif kind == 'PROF':
                rec['prof'] = text
                break   # PROF follows DONE; TIERS/EMAP/HITS follow PROF
            elif kind == 'TIERS':
                rec['tiers'] = text
            elif kind == 'ERROR':
                rec['error'] = text; break
        # HITS, EMAP, then the turn's TIERS line follow PROF
        line = '' if 'error' in rec else p.stdout.readline().decode('latin-1').rstrip('\n')
        while line and not line.startswith('TIERS'):
            line = p.stdout.readline().decode('latin-1').rstrip('\n')
        rec['tiers'] = line
        rec['ttft'] = (t_first - t0) if t_first else None
        rec['wall'] = time.time() - t0
        rec['text'] = b''.join(chunks).decode('utf-8', 'replace')
        out.write(json.dumps(rec, ensure_ascii=False) + '\n'); out.flush()
        print(pid, 'ttft', round(rec['ttft'] or -1, 2), rec.get('done'), rec.get('prof'), rec['tiers'], flush=True)

    p.stdin.close()
    try:
        p.wait(timeout=120)
    except subprocess.TimeoutExpired:
        p.kill()
    time.sleep(1)
    errlog.close()


if __name__ == '__main__':
    main()
