"""Download small public text files (configs, model cards) at pinned revisions. Usage: python -I fetch_small.py OUT_DIR REPO@REV:FILE ..."""
import os, sys, urllib.request

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
for spec in sys.argv[2:]:
    rr, f = spec.split(':', 1)
    repo, rev = rr.split('@')
    url = f'https://huggingface.co/{repo}/resolve/{rev}/{f}'
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'team7-other-models/1.0'}), timeout=60) as r:
            data = r.read(2_000_000)
        p = os.path.join(out, repo.replace('/', '__') + '__' + f.replace('/', '_'))
        open(p, 'wb').write(data)
        print('ok', spec, len(data))
    except Exception as e:
        print('ERR', spec, e)
