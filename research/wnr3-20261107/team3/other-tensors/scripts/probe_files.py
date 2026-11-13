"""HEAD-ish probe: Range 0-7 for every shard, report status and header length."""
import json, sys, os, struct, urllib.request, urllib.error
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf
wm = json.load(open(sys.argv[1], encoding='utf-8'))['weight_map']
files = sorted(set(wm.values())) + ['config.json', 'inference/engram.py', 'inference/model.py', 'README.md']
for f in files:
    try:
        req = urllib.request.Request(hf.BASE + f, headers={**hf.UA, 'Range': 'bytes=0-7'})
        with urllib.request.urlopen(req, timeout=60) as r:
            d = r.read(); cr = r.headers.get('Content-Range')
        print(f, 'OK', struct.unpack('<Q', d)[0] if len(d) == 8 and f.endswith('.safetensors') else d, cr)
    except urllib.error.HTTPError as e:
        print(f, 'HTTP', e.code)
