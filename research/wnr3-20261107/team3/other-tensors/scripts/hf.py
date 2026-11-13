"""Tiny HTTP Range helper for the pinned HF revision (no code from the repo is executed)."""
import json, struct, time, os
import urllib.request

REPO = 'patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4'
REV = '4ab649b2254afdcf6c2f7c1fbea1c05eda11828c'
BASE = f'https://huggingface.co/{REPO}/resolve/{REV}/'
UA = {'User-Agent': 'wnr2-research-team3/1.0'}
# The two Engram shards are not in the REAP repo; its README/engram_pointer point to the base repo at this revision.
BASE_REPO = 'deepseek-ai/DeepSeek-V4.1-Flash-1017'
BASE_REV = '1bae0e1088993b9b82de08f59de5be5a34e42170'
BASE_BASE = f'https://huggingface.co/{BASE_REPO}/resolve/{BASE_REV}/'
ENGRAM_FILES = {'model-00047-of-00048.safetensors', 'model-00048-of-00048.safetensors'}


def url_for(fname):
    return (BASE_BASE if fname in ENGRAM_FILES else BASE) + fname


def get_range(fname, start, end_incl, tries=6):
    for t in range(tries):
        try:
            req = urllib.request.Request(url_for(fname), headers={**UA, 'Range': f'bytes={start}-{end_incl}'})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
            if len(data) != end_incl - start + 1:
                raise IOError(f'short read {len(data)} != {end_incl - start + 1}')
            return data
        except Exception:
            if t == tries - 1:
                raise
            time.sleep(2 + 3 * t)


def get_file(fname, tries=6):
    for t in range(tries):
        try:
            req = urllib.request.Request(url_for(fname), headers=UA)
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception:
            if t == tries - 1:
                raise
            time.sleep(2 + 3 * t)


def header(fname, cache_dir):
    p = os.path.join(cache_dir, fname + '.header.json')
    if os.path.exists(p):
        d = json.load(open(p, encoding='utf-8'))
        return d['base'], d['header']
    n = struct.unpack('<Q', get_range(fname, 0, 7))[0]
    hdr = json.loads(get_range(fname, 8, 8 + n - 1))
    json.dump({'base': 8 + n, 'header': hdr}, open(p + '.part', 'w', encoding='utf-8'))
    os.replace(p + '.part', p)
    return 8 + n, hdr
