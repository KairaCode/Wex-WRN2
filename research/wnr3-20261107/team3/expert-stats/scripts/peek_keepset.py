"""Print the structure of reap_keepset.json (untrusted data: only json.load, no eval)."""
import json, sys

d = json.load(open(sys.argv[1], encoding='utf-8'))


def show(x, pre='', depth=0):
    if depth > 4:
        return
    if isinstance(x, dict):
        print(pre + f'dict[{len(x)}] keys: {list(x.keys())[:12]}')
        for k in list(x.keys())[:4]:
            show(x[k], pre + f'  [{k!r}] ', depth + 1)
    elif isinstance(x, list):
        print(pre + f'list[{len(x)}] first: {repr(x[:8])[:300]}')
        if x and isinstance(x[0], (dict, list)):
            show(x[0], pre + '  [0] ', depth + 1)
    else:
        print(pre + repr(x)[:300])


show(d)
