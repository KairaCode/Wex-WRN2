"""Fetch / extract / verify the V4.1 REAP-192 MXFP4 expert sample used by every WNR kit test.

Data sets (bundled manifests in setup/manifests/, sha256 per file):
  v41-sample : 160 routed experts, 4 per layer x 40 layers (seed 20261107)        3,008,102,400 bytes
  extra      : 70 routed experts not in v41-sample + 6 MTP experts (seed 20261108) 1,428,848,640 bytes  (--extra)
Each expert file is <data>/<set>/L{LL}_E{EEE}.bin (MTP: M{LL}_E{EEE}.bin), 18,800,640 bytes = 6 tensors back to back,
in the order w1.scale w2.scale w3.scale (F8_E8M0) w1.weight w2.weight w3.weight (I8, packed FP4 pairs), exactly as
stored in the safetensors shards of the pinned checkpoint
  patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4 @ 4ab649b2254afdcf6c2f7c1fbea1c05eda11828c

Modes (PowerShell, from the kit root):
  python -I setup\\fetch_data.py                       # download v41-sample over HTTP Range (only the tensor bytes)
  python -I setup\\fetch_data.py --extra               # ... plus the extra set
  python -I setup\\fetch_data.py --local-shards D:\\models\\V4.1-REAP192-MXFP4   # copy tensors out of local shards
  python -I setup\\fetch_data.py --verify-only [--extra] [--quick]               # read-only check, no network
  python -I setup\\fetch_data.py --dry-run [--extra]                             # print the plan, no network
Options: --data <dir> (default <root>/data), --out <results dir>, --threads N, --limit N (first N experts per set),
         --sample-dir / --extra-dir (use an existing folder for one set), --base-url (HF mirror).
Resumable: an interrupted file stays as <file>.part and the next run continues it from its current length; a file
whose sha256 does not match is renamed <file>.bad and fetched again. Downloads require HTTP 206 (a server that
ignores Range is refused, so a whole 4-GB shard is never pulled by mistake).
Writes <out>/summary.json and <out>/fetch_data.log; last line: RESULT fetch_data PASS|FAIL <summary>.
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kitenv  # noqa: E402

TEST = 'fetch_data'
REPO = 'patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4'
REV = '4ab649b2254afdcf6c2f7c1fbea1c05eda11828c'
UA = {'User-Agent': 'wnr-testkit-fetch/1.0'}
CHUNK = 1 << 20
ORDER = ['w1.scale', 'w2.scale', 'w3.scale', 'w1.weight', 'w2.weight', 'w3.weight']


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        while True:
            b = fh.read(8 * CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def expert_bytes(e):
    return sum(t['bytes'] for t in e['tensors'])


class Progress:
    def __init__(self, log, total):
        self.log, self.total, self.done, self.t0, self.last = log, total, 0, time.time(), 0.0
        self.lock = threading.Lock()

    def add(self, n):
        with self.lock:
            self.done += n
            now = time.time()
            if now - self.last > 10 or self.done >= self.total:
                self.last = now
                dt = max(now - self.t0, 1e-6)
                self.log(f'  progress {self.done / 1e6:9.1f} / {self.total / 1e6:.1f} MB  '
                         f'{self.done / dt / 1e6:6.1f} MB/s')


# ------------------------------------------------------------------ sources

class HttpSource:
    def __init__(self, base_url, timeout=120, tries=8):
        self.base = f"{base_url.rstrip('/')}/{REPO}/resolve/{REV}/"
        self.timeout, self.tries = timeout, tries
        self.kind = 'http'

    def _get(self, shard, start, length, fh, prog):
        end = start + length - 1
        req = urllib.request.Request(self.base + shard, headers={**UA, 'Range': f'bytes={start}-{end}'})
        got = 0
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            if r.status != 206:
                raise IOError(f'server answered HTTP {r.status} instead of 206 Partial Content; refusing a full '
                              f'download of {shard}')
            cr = r.headers.get('Content-Range', '')
            if cr and not cr.replace(' ', '').startswith(f'bytes{start}-'):
                raise IOError(f'unexpected Content-Range {cr!r} for {start}-{end}')
            while got < length:
                b = r.read(min(CHUNK, length - got))
                if not b:
                    break
                fh.write(b)
                got += len(b)
                if prog:
                    prog.add(len(b))
        fh.flush()
        if got != length:
            raise IOError(f'short read {got} != {length}')

    def fill(self, e, part_path, prog):
        """Append the expert's tensors to part_path, resuming from its current length."""
        segs = [(t['shard'], t['offset'], t['bytes']) for t in e['tensors']]
        total = sum(s[2] for s in segs)
        for attempt in range(self.tries):
            have = os.path.getsize(part_path) if os.path.exists(part_path) else 0
            if have > total:
                os.remove(part_path)
                have = 0
            if have == total:
                return
            try:
                with open(part_path, 'ab') as fh:
                    pos = 0
                    for shard, off, n in segs:
                        if pos + n <= have:
                            pos += n
                            continue
                        skip = max(0, have - pos)
                        self._get(shard, off + skip, n - skip, fh, prog)
                        pos += n
                        have = pos
                return
            except (urllib.error.URLError, IOError, OSError, TimeoutError) as ex:
                if isinstance(ex, IOError) and 'refusing' in str(ex):
                    raise
                if attempt == self.tries - 1:
                    raise
                time.sleep(min(60, 2 + 4 * attempt))


class LocalShardSource:
    """Reads the same tensors out of local *.safetensors shards of the pinned checkpoint."""

    def __init__(self, shard_dir):
        d = os.path.abspath(shard_dir)
        if not any(f.endswith('.safetensors') for f in os.listdir(d)):
            snap = os.path.join(d, 'snapshots')
            if os.path.isdir(snap):  # Hugging Face cache layout
                subs = [os.path.join(snap, s) for s in os.listdir(snap)]
                pick = [s for s in subs if os.path.basename(s) == REV] or subs
                if pick:
                    d = pick[0]
        self.dir = d
        self.kind = 'local'
        self.files = sorted(f for f in os.listdir(d) if f.endswith('.safetensors'))
        if not self.files:
            raise SystemExit(f'no *.safetensors files in {shard_dir}')
        self._hdr = {}
        self.lock = threading.Lock()
        self.weight_map = None
        ix = os.path.join(d, 'model.safetensors.index.json')
        if os.path.exists(ix):
            try:
                self.weight_map = json.load(open(ix, encoding='utf-8'))['weight_map']
            except Exception:  # noqa: BLE001
                self.weight_map = None

    def header(self, fname):
        with self.lock:
            if fname not in self._hdr:
                p = os.path.join(self.dir, fname)
                with open(p, 'rb') as fh:
                    n = struct.unpack('<Q', fh.read(8))[0]
                    if n > 200_000_000:
                        raise IOError(f'{fname}: implausible header length {n}')
                    self._hdr[fname] = (8 + n, json.loads(fh.read(n)))
            return self._hdr[fname]

    def locate(self, name, hint):
        cands = []
        if self.weight_map and name in self.weight_map:
            cands.append(self.weight_map[name])
        if hint in self.files:
            cands.append(hint)
        cands += [f for f in self.files if f not in cands]
        for f in cands:
            if f not in self.files:
                continue
            base, hdr = self.header(f)
            if name in hdr:
                return f, base, hdr[name]
        raise KeyError(f'tensor {name} not found in any shard under {self.dir}')

    def fill(self, e, part_path, prog):
        kind = e.get('kind', 'layers')
        if os.path.exists(part_path):
            os.remove(part_path)  # local copy is fast: no partial resume
        with open(part_path, 'wb') as out:
            for t in e['tensors']:
                name = f"{kind}.{e['layer']}.ffn.experts.{e['expert']}.{t['name']}"
                f, base, m = self.locate(name, t['shard'])
                a, b = m['data_offsets']
                if m['dtype'] != t['dtype'] or list(m['shape']) != list(t['shape']) or b - a != t['bytes']:
                    raise IOError(f"{name}: local {m['dtype']} {m['shape']} {b - a} B differs from pinned "
                                  f"{t['dtype']} {t['shape']} {t['bytes']} B (different checkpoint?)")
                with open(os.path.join(self.dir, f), 'rb') as fh:
                    fh.seek(base + a)
                    left = b - a
                    while left:
                        buf = fh.read(min(8 * CHUNK, left))
                        if not buf:
                            raise IOError(f'{f}: unexpected EOF')
                        out.write(buf)
                        left -= len(buf)
                        if prog:
                            prog.add(len(buf))


# ------------------------------------------------------------------ main logic

def check_file(path, e, do_hash):
    n = expert_bytes(e)
    if not os.path.exists(path):
        return 'partial' if os.path.exists(path + '.part') else 'missing'
    sz = os.path.getsize(path)
    if sz != n:
        return 'size_mismatch'
    if not do_hash:
        return 'ok_size'
    return 'ok' if sha256_file(path) == e['sha256'] else 'sha_mismatch'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=kitenv.default_root(__file__))
    ap.add_argument('--data', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--quick', action='store_true', help='verify: sha256 of every 8th file only (sizes of all)')
    ap.add_argument('--extra', action='store_true', help='also the 70 extra routed + 6 MTP experts')
    ap.add_argument('--only-extra', action='store_true', help='only the extra set')
    ap.add_argument('--local-shards', default=None, help='directory with the checkpoint *.safetensors shards')
    ap.add_argument('--verify-only', action='store_true', help='read-only check, never writes into --data')
    ap.add_argument('--dry-run', action='store_true', help='print what would be fetched, no network, no writes')
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0, help='only the first N experts of each set (testing)')
    ap.add_argument('--sample-dir', default=None, help='folder of the v41-sample set (default <data>/v41-sample)')
    ap.add_argument('--extra-dir', default=None, help='folder of the extra set (default <data>/extra)')
    ap.add_argument('--base-url', default='https://huggingface.co')
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    data = os.path.abspath(a.data or os.path.join(root, 'data'))
    out = os.path.abspath(a.out or os.path.join(root, 'results', TEST))
    os.makedirs(out, exist_ok=True)
    log = kitenv.Log(os.path.join(out, TEST + '.log'))
    mdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'manifests')
    sets = []
    if not a.only_extra:
        sets.append((kitenv.SAMPLE_SET, os.path.join(mdir, 'v41-sample.json'),
                     os.path.abspath(a.sample_dir or os.path.join(data, kitenv.SAMPLE_SET))))
    if a.extra or a.only_extra:
        sets.append((kitenv.EXTRA_SET, os.path.join(mdir, 'extra.json'),
                     os.path.abspath(a.extra_dir or os.path.join(data, kitenv.EXTRA_SET))))
    mode = 'verify' if a.verify_only else 'dry-run' if a.dry_run else ('local' if a.local_shards else 'http')
    log(f'fetch_data mode={mode} data={data} sets={[s[0] for s in sets]} quick={a.quick}')
    t0 = time.time()
    src = None
    if mode == 'local':
        src = LocalShardSource(a.local_shards)
        log(f'local shards: {src.dir} ({len(src.files)} files)')
    elif mode == 'http':
        src = HttpSource(a.base_url)
        log(f'source: {src.base}')

    report = {}
    failures = []
    grand_total = grand_fetched = 0
    for set_name, man_path, tdir in sets:
        man = json.load(open(man_path, encoding='utf-8'))
        if man.get('repo') != REPO or man.get('revision') != REV:
            raise SystemExit(f'{man_path}: manifest is for {man.get("repo")}@{man.get("revision")}, expected pinned')
        exps = man['experts'][:a.limit] if a.limit else man['experts']
        total = sum(expert_bytes(e) for e in exps)
        grand_total += total
        log(f'[{set_name}] {len(exps)} experts, {total:,} bytes -> {tdir}')
        hash_idx = set(range(len(exps)))
        if a.quick:
            hash_idx = set(range(0, len(exps), 8)) | {len(exps) - 1}
        # ---- first pass: state of every file (hash only where needed)
        hash_now = (mode == 'verify') or not a.dry_run
        states = {}

        def st(i_e):
            i, e = i_e
            p = os.path.join(tdir, e['file'])
            return e['file'], check_file(p, e, hash_now and i in hash_idx)

        if os.path.isdir(tdir):
            with ThreadPoolExecutor(max(1, min(a.threads, 8))) as ex:
                for fn, s in ex.map(st, list(enumerate(exps))):
                    states[fn] = s
        else:
            states = {e['file']: 'missing' for e in exps}
        counts = {}
        for s in states.values():
            counts[s] = counts.get(s, 0) + 1
        log(f'[{set_name}] state: {counts}')
        todo = [e for e in exps if states[e['file']] not in ('ok', 'ok_size')]
        todo_bytes = sum(expert_bytes(e) for e in todo)
        fetched = 0
        set_fail = []
        if mode in ('verify', 'dry-run'):
            for e in todo:
                set_fail.append({'file': e['file'], 'state': states[e['file']]})
            if mode == 'dry-run':
                partial = sum(os.path.getsize(os.path.join(tdir, e['file'] + '.part')) for e in todo
                              if os.path.exists(os.path.join(tdir, e['file'] + '.part')))
                log(f'[{set_name}] would fetch {len(todo)} files, {todo_bytes - partial:,} bytes '
                    f'({partial:,} already in .part files)')
        elif todo:
            os.makedirs(tdir, exist_ok=True)
            prog = Progress(log, todo_bytes)

            def work(e):
                p = os.path.join(tdir, e['file'])
                s0 = states[e['file']]
                if s0 in ('sha_mismatch', 'size_mismatch'):
                    os.replace(p, p + '.bad')
                for attempt in range(2):
                    try:
                        src.fill(e, p + '.part', prog)
                    except Exception as ex:  # noqa: BLE001
                        return {'file': e['file'], 'state': 'fetch_error', 'error': f'{type(ex).__name__}: {ex}'}
                    h = sha256_file(p + '.part')
                    if h == e['sha256']:
                        os.replace(p + '.part', p)
                        return None
                    os.replace(p + '.part', p + '.sha-mismatch')
                    if src.kind == 'local':
                        break
                return {'file': e['file'], 'state': 'sha_mismatch_after_fetch',
                        'error': 'sha256 differs from the bundled manifest (kept as .sha-mismatch)'}

            with ThreadPoolExecutor(max(1, a.threads)) as ex:
                for r in ex.map(work, todo):
                    if r:
                        set_fail.append(r)
                        log(f"  FAIL {r['file']}: {r['state']} {r.get('error', '')}")
            fetched = todo_bytes
        # ---- manifest copy next to the data (fetch modes only)
        if mode in ('http', 'local') and os.path.isdir(tdir):
            mp = os.path.join(tdir, 'manifest.json')
            mine = json.dumps(man, indent=1)
            if not os.path.exists(mp):
                open(mp, 'w', encoding='utf-8').write(mine)
            else:
                try:
                    old = json.load(open(mp, encoding='utf-8'))
                    same = [e['sha256'] for e in old.get('experts', [])] == [e['sha256'] for e in man['experts']]
                except Exception:  # noqa: BLE001
                    same = False
                if not same:
                    open(os.path.join(tdir, 'manifest.kit.json'), 'w', encoding='utf-8').write(mine)
                    log(f'[{set_name}] existing manifest.json differs; kit manifest written as manifest.kit.json')
        grand_fetched += fetched
        failures += [dict(f, set=set_name) for f in set_fail]
        report[set_name] = {'dir': tdir, 'experts': len(exps), 'bytes': total, 'states_before': counts,
                            'hashed': len(hash_idx) if hash_now else 0, 'to_fetch': len(todo),
                            'bytes_to_fetch': todo_bytes, 'bytes_fetched': fetched, 'failures': set_fail}

    dt = time.time() - t0
    status = 'PASS' if not failures and mode != 'dry-run' else ('DONE' if mode == 'dry-run' else 'FAIL')
    summary = {'test': TEST, 'status': status, 'mode': mode, 'quick': a.quick, 'seconds': round(dt, 1),
               'repo': REPO, 'revision': REV, 'data': data, 'sets': report, 'total_bytes': grand_total,
               'fetched_bytes': grand_fetched, 'failures': failures,
               'environment': kitenv.env_info(with_gpu=False)}
    kitenv.write_json(os.path.join(out, 'summary.json'), summary)
    log(f'total bytes in selected sets: {grand_total:,}; fetched/copied this run: {grand_fetched:,}; {dt:.1f} s')
    short = ' '.join(f"{k}={v['experts'] - len(v['failures'])}/{v['experts']}" for k, v in report.items())
    log(f'RESULT {TEST} {status} mode={mode} {short} bytes={grand_total} failures={len(failures)}')
    log.close()
    return 0 if status != 'FAIL' else 1


if __name__ == '__main__':
    sys.exit(main())
