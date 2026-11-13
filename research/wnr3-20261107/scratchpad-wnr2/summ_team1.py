"""Summarise team 1 workflow output JSON: structure and per-candidate key fields."""
import json, sys

d = json.load(open(sys.argv[1], encoding='utf-8'))
mode = sys.argv[2] if len(sys.argv) > 2 else 'struct'
res = d.get('result', d)
if isinstance(res, str):
    try:
        res = json.loads(res)
    except Exception:
        pass


def struct(o, pre='', depth=0):
    if depth > 3:
        return
    if isinstance(o, dict):
        for k, v in o.items():
            t = type(v).__name__
            ln = len(v) if hasattr(v, '__len__') else ''
            print(f'{pre}{k}: {t} {ln}')
            struct(v, pre + '  ', depth + 1)
    elif isinstance(o, list) and o:
        print(f'{pre}[0]:')
        struct(o[0], pre + '  ', depth + 1)


if mode == 'struct':
    print(list(d.keys()))
    struct(res)
elif mode == 'cands':
    for sw in res['sweeps']:
        print('=== ANGLE', sw.get('angle'))
        for c in sw.get('candidates', []):
            print('-', c.get('name'), '|', c.get('category'), '|', c.get('lossless'), '|', c.get('confidence'))
            ee = c.get('expected_effect', '')
            print('   EFFECT:', ee[:600])
elif mode == 'key':
    k = sys.argv[3]
    o = res
    for part in k.split('.'):
        o = o[int(part)] if part.isdigit() else o[part]
    print(json.dumps(o, indent=1, ensure_ascii=False)[: int(sys.argv[4]) if len(sys.argv) > 4 else 20000])
