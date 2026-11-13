"""Write list_<cfg>.txt / list_all.txt for wnrdec.exe (record paths relative to the member folder; source .bin paths absolute from WNR_SAMPLE)."""
import json, os, sys
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.abspath(os.environ.get('WNR_SAMPLE', 'v41-sample'))  # kit
try:  # wnrdec.c reads the list with fscanf %s: a path with spaces breaks it, so prefer a relative path
    SAMPLE = os.path.relpath(SAMPLE, root)
except ValueError:  # different drive
    pass
SAMPLE = SAMPLE.replace(os.sep, '/')
if ' ' in SAMPLE:
    sys.exit('mklist: sample path contains a space (%s); move/junction the data to a path without spaces' % SAMPLE)
m = json.load(open(os.path.join(root, 'rec', 'records.json')))
L = {'wnr2': [], 'fge': []}
for r in sorted(m['records'], key=lambda r: r['idx']):
    s = ' '.join(f'{a} {b}' for a, b in r['shapes'])
    L[r['config']].append(f"rec/{r['rec']} {SAMPLE}/{r['file']} {r['layer']} {r['expert']} {r['model_id']} "
                          f"{r['config']} {s} rec/models/{r['config']}")
for k, v in L.items():
    open(os.path.join(root, f'list_{k}.txt'), 'w').write('\n'.join(v) + '\n')
open(os.path.join(root, 'list_all.txt'), 'w').write('\n'.join(L['wnr2'] + L['fge']) + '\n')
for k in L:
    open(os.path.join(root, f'list_{k}1.txt'), 'w').write(L[k][0] + '\n')
print(L['fge'][0])
