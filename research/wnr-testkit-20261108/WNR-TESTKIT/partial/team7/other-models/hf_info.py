"""Query public HF model API (no auth): sha, gated, private, lastModified, license tag, siblings count + total size.
Usage: python -I hf_info.py REPO [REPO ...]"""
import json, sys, urllib.request

for repo in sys.argv[1:]:
    url = f'https://huggingface.co/api/models/{repo}?blobs=true'
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'team7-other-models/1.0'}), timeout=60) as r:
            j = json.loads(r.read())
    except Exception as e:
        print(repo, 'ERROR', e)
        continue
    sib = j.get('siblings', [])
    tot = sum(s.get('size') or 0 for s in sib)
    st = sum(s.get('size') or 0 for s in sib if s['rfilename'].endswith('.safetensors'))
    lic = [t for t in j.get('tags', []) if t.startswith('license:')]
    base = [t for t in j.get('tags', []) if t.startswith('base_model:')]
    print(json.dumps({'repo': repo, 'sha': j.get('sha'), 'gated': j.get('gated'), 'private': j.get('private'),
                      'disabled': j.get('disabled'), 'lastModified': j.get('lastModified'), 'license': lic,
                      'base_model': base, 'n_files': len(sib), 'total_bytes': tot, 'safetensors_bytes': st}))
