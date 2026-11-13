"""Crude PDF text probe: inflate FlateDecode streams, decode TJ/Tj strings (literal or hex with ToUnicode CMaps), print
lines matching a regex. Usage: python -I pdf_grep.py PDF REGEX"""
import re, sys, zlib

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
data = open(sys.argv[1], 'rb').read()
pat = re.compile(sys.argv[2], re.I)
streams = []
for m in re.finditer(rb'stream\r?\n', data):
    s = m.end()
    e = data.find(b'endstream', s)
    try:
        streams.append(zlib.decompress(data[s:e]))
    except Exception:
        pass
# collect ToUnicode cmaps (bfchar / bfrange), merged (crude: ignores which font)
cmap = {}
for st in streams:
    if b'begincmap' in st:
        for blk in re.findall(rb'beginbfchar(.*?)endbfchar', st, re.S):
            for a, b in re.findall(rb'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', blk):
                cmap.setdefault(len(a), {})[int(a, 16)] = bytes.fromhex(b.decode()).decode('utf-16-be', 'replace')
        for blk in re.findall(rb'beginbfrange(.*?)endbfrange', st, re.S):
            for a, b, c in re.findall(rb'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', blk):
                lo, hi, base = int(a, 16), int(b, 16), int(c, 16)
                if len(c) > 4:
                    try:
                        base = ord(bytes.fromhex(c.decode()).decode('utf-16-be'))
                    except Exception:
                        continue
                for k in range(lo, min(hi, lo + 4096) + 1):
                    cp = base + k - lo
                    if cp < 0x110000:
                        cmap.setdefault(len(a), {})[k] = chr(cp)


def dec_hex(h):
    h = h.decode()
    for w in (4, 2):
        if w in cmap and len(h) % w == 0:
            return ''.join(cmap[w].get(int(h[i:i + w], 16), '?') for i in range(0, len(h), w))
    return ''


page_texts = []
for st in streams:
    if b'BT' not in st:
        continue
    out = []
    for line in re.split(rb'\n', st):
        if b'TJ' in line or b'Tj' in line:
            parts = re.findall(rb'\((.*?)(?<!\\)\)|<([0-9A-Fa-f]+)>', line)
            out.append(''.join((p[0].decode('latin-1') if p[0] else dec_hex(p[1])) for p in parts))
        elif b'ET' in line or b'Td' in line or b'TD' in line or b'T*' in line:
            out.append('\n')
    page_texts.append(''.join(out))
txt = '\n'.join(page_texts)
for i, ln in enumerate(txt.split('\n')):
    if pat.search(ln):
        print(i, ln[:300])
