"""G3: WNR3 encoder (non-CZ, lab CFG model) vs the 5090 lab CFG records.
usage: python -I g3_compare.py OUR.bin.json LAB_RECORDS.json LAB_CFG_all52.bin"""
import json
import struct
import sys

ours = json.load(open(sys.argv[1]))['rows']
lab = json.load(open(sys.argv[2]))['rows']
anchors = []
with open(sys.argv[3], 'rb') as f:
    mb = struct.unpack('<I', f.read(4))[0]
    f.seek(4 + mb)
    n = struct.unpack('<I', f.read(4))[0]
    for _ in range(n):
        e = struct.unpack('<QIII6H6H3I3II', f.read(72))
        rb = struct.unpack('<I', f.read(4))[0]
        f.seek(rb, 1)
        head = f.read(32)
        anchors.append(list(head[24:27]))
        f.seek(e[1] - 32, 1)
assert len(ours) == len(lab) == len(anchors) == 52
bad_anchor = [(o['layer'], o['expert']) for o, a in zip(ours, anchors) if o['anchors'] != a]
same_id = all((o['layer'], o['expert']) == (l['layer'], l['expert']) for o, l in zip(ours, lab))
test = [(o, l) for o, l in zip(ours, lab) if l['set'] == 'test']
d_used = [o['used'] - l['CFG_used'] for o, l in test]
mean_total = sum(o['total'] for o, _ in test) / len(test)
lab_mean = sum(l['CFG_total'] for _, l in test) / len(test)
out = {'same_experts': same_id, 'anchor_mismatches': bad_anchor,
       'test40_mean_total': mean_total, 'lab_CFG_mean_total': lab_mean,
       'delta_mean_total': mean_total - lab_mean, 'mean_used_delta_bytes': sum(d_used) / len(d_used),
       'min_used_delta': min(d_used), 'max_used_delta': max(d_used),
       'tail12_max_total': max(o['total'] for o, l in zip(ours, lab) if l['set'] == 'big'),
       'gate_failures': sum(len(o['gate_failed']) for o in ours),
       'G3_window_ok': abs(mean_total - (16578252.8 + 1352)) <= 1500}
print(json.dumps(out, indent=1))
