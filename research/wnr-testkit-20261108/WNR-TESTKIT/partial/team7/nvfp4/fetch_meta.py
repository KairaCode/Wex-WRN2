"""Fetch public metadata of HF repos (no token): revision sha, file list, small json/md files.

Usage: python -I fetch_meta.py OUT_DIR repo1 [repo2 ...]
Writes OUT_DIR/<owner>__<name>__<file> for config.json, hf_quant_config.json, quantization_config.json,
recipe.yaml, README.md, model.safetensors.index.json (if present) and OUT_DIR/<owner>__<name>__api.json.
Nothing downloaded is executed.
"""
import json, os, sys, time, urllib.request

UA = {'User-Agent': 'team7-nvfp4/1.0'}
WANT = ['config.json', 'hf_quant_config.json', 'quantization_config.json', 'recipe.yaml', 'README.md',
        'model.safetensors.index.json', 'generation_config.json']


def get(url, tries=4):
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                raise
            if t == tries - 1:
                raise
        except Exception:
            if t == tries - 1:
                raise
        time.sleep(2 + 3 * t)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for repo in sys.argv[2:]:
        tag = repo.replace('/', '__')
        try:
            api = json.loads(get(f'https://huggingface.co/api/models/{repo}?blobs=true'))
        except Exception as e:
            print(repo, 'API ERROR', e)
            continue
        sha = api.get('sha')
        sib = api.get('siblings', [])
        tot = sum((s.get('size') or 0) for s in sib)
        keep = {'id': api.get('id'), 'sha': sha, 'gated': api.get('gated'), 'private': api.get('private'),
                'disabled': api.get('disabled'), 'cardData': {k: api.get('cardData', {}).get(k) for k in ('license', 'base_model', 'license_name')} if api.get('cardData') else None,
                'lastModified': api.get('lastModified'), 'total_bytes': tot,
                'siblings': [{'f': s['rfilename'], 'size': s.get('size')} for s in sib]}
        json.dump(keep, open(os.path.join(out, f'{tag}__api.json'), 'w'), indent=1)
        names = {s['rfilename'] for s in sib}
        got = []
        for w in WANT:
            if w in names:
                p = os.path.join(out, f'{tag}__{w}')
                if not os.path.exists(p):
                    data = get(f'https://huggingface.co/{repo}/resolve/{sha}/{w}')
                    open(p, 'wb').write(data)
                got.append(w)
        nst = sum(1 for n in names if n.endswith('.safetensors'))
        print(repo, sha, 'gated', api.get('gated'), 'files', len(sib), 'safetensors', nst, 'GB', round(tot / 1e9, 2), got, flush=True)


if __name__ == '__main__':
    main()
