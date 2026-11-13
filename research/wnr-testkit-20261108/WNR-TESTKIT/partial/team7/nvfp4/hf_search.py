"""List public NVFP4 / FP4 model repos on Hugging Face via the public API (no token).

Usage: python -I hf_search.py OUT_JSON query1 [query2 ...]
Each query is 'search' or 'author:search'. Writes id, gated, private, downloads, lastModified, tags.
"""
import json, sys, urllib.request, urllib.parse

UA = {'User-Agent': 'team7-nvfp4/1.0'}


def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def main():
    out = sys.argv[1]
    res = {}
    for q in sys.argv[2:]:
        author, _, s = q.rpartition(':')
        p = {'search': s, 'limit': 200, 'full': 'true'}
        if author:
            p['author'] = author
        url = 'https://huggingface.co/api/models?' + urllib.parse.urlencode(p)
        try:
            lst = get(url)
        except Exception as e:
            res[q] = {'error': repr(e)}
            continue
        res[q] = [{'id': m.get('id'), 'gated': m.get('gated'), 'private': m.get('private'),
                   'downloads': m.get('downloads'), 'lastModified': m.get('lastModified'),
                   'sha': m.get('sha'), 'tags': [t for t in m.get('tags', []) if not t.startswith('region')][:12]}
                  for m in lst]
        print(q, len(lst))
    json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
