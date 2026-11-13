"""Step Q of RUNBOOK.md: quality A/B of qwen38 with the AMD MXFP4 routed experts against the FP8 snapshot's own.

Both arms are the SAME engine binary on the SAME FP8 snapshot, CLI reference mode, greedy, MTP and prompt
lookup off; the only difference is COLI_WEX_PATH (arm mx). Every engine run sits in a Job Object capped
like verify_gateway.py (RAM_GB + 2.5 GB + the card's memory + 1 GB, kill on close), and waits for its
machine gate (verify_gateway.machine_ok) first. For each entry of the prompts file:
  "chat" entries:  fp8 generates N tokens; mx generates from the same prompt (first divergence measured);
                   then both arms score FP8's continuation teacher-forced (PPL=1 -> TF-NLL nats/token);
  "text" entries:  both arms score the text teacher-forced (its first 16 tokens are the prompt).
An entry is {"name", "kind": "chat"|"text", "text"} (tokenized with <snap>/tokenizer.json) or {"name",
"kind", "ids": [...]} (the tiny fixture has no tokenizer).
Writes <out>/q_ab.json and <out>/q_ab.md (the continuations decoded, side by side).

  python -I q_ab.py --engine rel\\qwen38.exe --snap U:\\models\\Qwen3.8-Flash-Next-FP8 \\
      --wex U:\\kaira_wex\\qwen38-mxfp4\\q38_experts.wex --out results\\q_wex1 [--n 64] [--ram-gb 16]
Rule (printed at the end): PASS when the mean TF-NLL gap mx - fp8 is <= 0.03 nats/token on the texts and
<= 0.05 on FP8's continuations; otherwise LOOK (read q_ab.md before going on to P4/P5)."""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'qwen38-ramfix-20261109'))
import verify_gateway as vg   # noqa: E402  the Job Object, the machine gate and the KAIRA qwen38 profile env


def engine_env(args, wex):
    env = dict(os.environ)
    for k in list(env):
        if k.startswith(('COLI_WEX', 'Q38_MXFP4', 'FAKE_WEX')) or k in ('PYTHONPATH', 'PYTHONHOME', 'SNAP', 'RAM_GB'):
            env.pop(k)
    if not args.no_profile:
        env.update(vg.PROFILE_ENV)
        env['COLI_MODEL_MIRROR'] = args.snap
        env['PATH'] = vg.CUDA_BIN + ';' + env['PATH']
    for f in vg.MTP_FLAGS:
        env[f] = '0'
    env.update({'COLI_LOOKUP': '0', 'USAGE_SAVE': '0', 'SNAP': args.snap, 'RAM_GB': str(args.ram_gb)})
    env.update(dict(item.split('=', 1) for item in args.env))
    if wex:
        env['COLI_WEX_PATH'] = wex
    return env


def gate(ram_gb):
    """vg.machine_ok() sized for THIS run's --ram-gb (machine_ok sizes for the 16-17 GB profile even at 13)."""
    p, c = vg.perf(), vg.c_free()
    used, total = vg.vram_once()
    if c <= 20 * vg.GB: return 'C: %.1f GB free (needs > 20)' % (c / vg.GB)
    if p['avail'] < (ram_gb + 2.5) * vg.GB: return 'available RAM %.1f GB (needs %.1f)' % (p['avail'] / vg.GB, ram_gb + 2.5)
    if used < 0 or used > 6000: return 'VRAM in use %d MiB (needs <= 6000)' % used
    cap = vg.job_cap(ram_gb, total)
    if p['limit'] - p['commit'] < cap: return 'commit headroom %.1f GB (needs %.1f)' % ((p['limit'] - p['commit']) / vg.GB, cap / vg.GB)
    return None


def run(args, wex, ids_prompt, ids_full, ppl, tag):
    """One capped engine run in reference mode; returns its parsed lines."""
    ref = os.path.join(args.out, tag + '.ref.json')
    with open(ref, 'w', encoding='utf-8') as f:
        json.dump({'schema_version': 1, 'prompt_ids': ids_prompt, 'full_ids': ids_full}, f)
    env = engine_env(args, wex)
    if ppl:
        env['PPL'] = '1'
    if not args.no_gate:
        while True:
            why = gate(args.ram_gb)
            if not why:
                break
            print(f'  waiting: {why}', flush=True)
            time.sleep(30)
    _, vram_total = vg.vram_once()
    job = vg.k32.CreateJobObjectW(None, None)
    li = vg.EXT()
    li.Basic.LimitFlags = vg.JOB_MEMORY | vg.KILL_ON_CLOSE
    li.JobMemoryLimit = int(vg.job_cap(args.ram_gb, max(vram_total, 0)))
    if not job or not vg.k32.SetInformationJobObject(job, vg.EXT_INFO, vg.ctypes.byref(li), vg.ctypes.sizeof(li)):
        sys.exit('job setup failed')
    log = open(os.path.join(args.out, tag + '.stderr.log'), 'wb')
    t0 = time.time()
    # argv[2] is the engine's reserved expert-bits argument (2..8); the token count comes from full_ids in ref
    p = subprocess.Popen([args.engine, str(args.cap), '8', ref], env=env,
                         stdout=subprocess.PIPE, stderr=log, creationflags=vg.CREATE_SUSPENDED)
    if not vg.k32.AssignProcessToJobObject(job, int(p._handle)):
        p.kill()
        sys.exit('assign failed')
    vg.ntdll.NtResumeProcess(int(p._handle))
    out = p.communicate()[0].decode('utf-8', 'replace')
    vg.k32.CloseHandle(job)
    log.close()
    rec = {'rc': p.returncode, 'wall_s': round(time.time() - t0, 1)}
    for line in out.splitlines():
        if line.startswith('C engine :'):
            # the engine prints "Text      : ..." on the same line, right after the ids (no newline)
            rec['tokens'] = [int(t) for t in line.split(':', 1)[1].split('Text', 1)[0].split()]
        m = re.match(r'TF-NLL: ([0-9.]+) nats/token over (\d+) tokens', line)
        if m:
            rec['nll'], rec['scored'] = float(m.group(1)), int(m.group(2))
        m = re.match(r'Speed: ([0-9.]+) tok/s', line)
        if m:
            rec['tok_s'] = float(m.group(1))
    if p.returncode not in (0, 1) or ('nll' not in rec and 'tokens' not in rec):   # 1: tokens differ from the reference
        sys.exit(f'FAIL {tag}: rc {p.returncode}, see {log.name}\n{out[-2000:]}')
    print(f'  {tag}: {json.dumps(rec)}', flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--engine', required=True)
    ap.add_argument('--snap', required=True, help='the FP8 snapshot (trunk, PLE, shared expert, MTP)')
    ap.add_argument('--wex', required=True, help='the MXFP4 container (WEX1 or WNR3) for arm mx')
    ap.add_argument('--out', required=True)
    ap.add_argument('--prompts', default=os.path.join(HERE, 'prompts.json'))
    ap.add_argument('--n', type=int, default=64, help='tokens generated per chat prompt')
    ap.add_argument('--cap', type=int, default=160, help='expert RAM cache per layer asked (the RAM plan sizes it)')
    ap.add_argument('--ram-gb', type=float, default=16)
    ap.add_argument('--env', action='append', default=[], help='NAME=VALUE for both arms')
    ap.add_argument('--no-profile', action='store_true', help='tiny fixture: no KAIRA profile env')
    ap.add_argument('--no-gate', action='store_true', help='skip the machine gate (tiny fixture only)')
    args = ap.parse_args()
    args.engine, args.snap, args.out = (os.path.abspath(p) for p in (args.engine, args.snap, args.out))
    os.makedirs(args.out, exist_ok=True)
    entries = json.load(open(args.prompts, encoding='utf-8'))
    tok = None
    if any('text' in e for e in entries):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(args.snap, 'tokenizer.json'))
    rows = []
    for e in entries:
        ids = e['ids'] if 'ids' in e else tok.encode(e['text'], add_special_tokens=False).ids
        name, row = e['name'], {'name': e['name'], 'kind': e['kind'], 'prompt_tokens': len(ids)}
        print(f'{name} ({e["kind"]}, {len(ids)} tokens)', flush=True)
        if e['kind'] == 'chat':
            fp8 = run(args, None, ids, ids + [0] * args.n, False, name + '.fp8.gen')['tokens']
            full = ids + fp8
            mx = run(args, args.wex, ids, full, False, name + '.mx.gen')['tokens']
            div = next((i for i, (a, b) in enumerate(zip(fp8, mx)) if a != b), len(fp8))
            row.update(fp8_tokens=fp8, mx_tokens=mx, first_divergence=div)
            if tok:
                row.update(fp8_text=tok.decode(fp8), mx_text=tok.decode(mx))
        else:
            full, ids = ids, ids[:16]
        a = run(args, None, ids, full, True, name + '.fp8.nll')
        b = run(args, args.wex, ids, full, True, name + '.mx.nll')
        row.update(fp8_nll=a['nll'], mx_nll=b['nll'], gap=round(b['nll'] - a['nll'], 5), scored=a['scored'])
        rows.append(row)
        json.dump(rows, open(os.path.join(args.out, 'q_ab.json'), 'w', encoding='utf-8'), indent=1)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0
    gap_text = mean([r['gap'] for r in rows if r['kind'] == 'text'])
    gap_chat = mean([r['gap'] for r in rows if r['kind'] == 'chat'])
    divs = [r['first_divergence'] for r in rows if r['kind'] == 'chat']
    verdict = 'PASS' if gap_text <= 0.03 and gap_chat <= 0.05 else 'LOOK'
    summary = {'verdict': verdict, 'gap_text_nats': round(gap_text, 5), 'gap_chat_nats': round(gap_chat, 5),
               'first_divergence': divs, 'identical_continuations': sum(d == args.n for d in divs), 'n': args.n}
    json.dump({'summary': summary, 'rows': rows}, open(os.path.join(args.out, 'q_ab.json'), 'w', encoding='utf-8'),
              indent=1)
    with open(os.path.join(args.out, 'q_ab.md'), 'w', encoding='utf-8') as f:
        f.write(f'# Q: MXFP4 vs FP8 ({args.wex})\n\n{json.dumps(summary)}\n\n')
        for r in rows:
            f.write(f"## {r['name']} ({r['kind']}): TF-NLL fp8 {r['fp8_nll']:.4f}, mx {r['mx_nll']:.4f}, gap {r['gap']:+.4f}"
                    + (f", first divergence {r['first_divergence']}/{args.n}" if r['kind'] == 'chat' else '') + '\n\n')
            if 'fp8_text' in r:
                f.write(f"FP8:\n\n```\n{r['fp8_text']}\n```\n\nMXFP4:\n\n```\n{r['mx_text']}\n```\n\n")
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
