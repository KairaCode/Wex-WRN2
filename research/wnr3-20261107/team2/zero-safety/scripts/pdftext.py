"""Minimal PDF text extractor (no external deps): inflate content streams, map
CID/byte strings through ToUnicode CMaps, print text per page-ish stream.
Usage: python -I pdftext.py <pdf> <out.txt>"""
import re, sys, zlib

data = open(sys.argv[1], 'rb').read()
objs = {}
for m in re.finditer(rb'(\d+)\s+0\s+obj(.*?)endobj', data, re.S):
    objs[int(m.group(1))] = m.group(2)

def stream_of(body):
    m = re.search(rb'stream\r?\n(.*?)\r?\nendstream', body, re.S)
    if not m:
        return None
    raw = m.group(1)
    if b'FlateDecode' in body[:m.start()]:
        try:
            return zlib.decompress(raw)
        except Exception:
            try:
                return zlib.decompressobj().decompress(raw)
            except Exception:
                return None
    return raw

# ToUnicode cmaps per font object
def parse_cmap(s):
    cmap = {}
    for blk in re.findall(rb'beginbfchar(.*?)endbfchar', s, re.S):
        for a, b in re.findall(rb'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', blk):
            try:
                cmap[int(a, 16)] = bytes.fromhex(b.decode()).decode('utf-16-be', 'replace')
            except Exception:
                pass
    for blk in re.findall(rb'beginbfrange(.*?)endbfrange', s, re.S):
        for a, b, c in re.findall(rb'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', blk):
            lo, hi = int(a, 16), int(b, 16)
            cb = bytes.fromhex(c.decode())
            try:
                st = ord(cb.decode('utf-16-be')[-1])
            except Exception:
                continue
            for k in range(lo, min(hi, lo + 4096) + 1):
                if st + k - lo < 0x110000:
                    cmap[k] = chr(st + k - lo)
    return cmap

fonts = {}  # font obj id -> (cmap, two_byte)
for oid, body in objs.items():
    if b'/Type/Font' in body.replace(b' ', b'') or b'/Type /Font' in body:
        tu = re.search(rb'/ToUnicode\s+(\d+)\s+0\s+R', body)
        two = b'Identity-H' in body or b'Type0' in body
        cm = {}
        if tu and int(tu.group(1)) in objs:
            st = stream_of(objs[int(tu.group(1))])
            if st:
                cm = parse_cmap(st)
        fonts[oid] = (cm, two)

# resource name -> font obj (global, approximate)
fontnames = {}
for oid, body in objs.items():
    for m in re.finditer(rb'/Font\s*<<(.*?)>>', body, re.S):
        for n, r in re.findall(rb'/(\w+)\s+(\d+)\s+0\s+R', m.group(1)):
            fontnames[n] = int(r)

def decode_str(sbytes, font):
    cm, two = fonts.get(font, ({}, False))
    out = []
    if two:
        for i in range(0, len(sbytes) - 1, 2):
            code = (sbytes[i] << 8) | sbytes[i + 1]
            out.append(cm.get(code, '?'))
    else:
        for b in sbytes:
            out.append(cm.get(b, chr(b)))
    return ''.join(out)

def lit(s):
    # unescape PDF literal string
    out = bytearray(); i = 0
    while i < len(s):
        c = s[i]
        if c == 0x5c and i + 1 < len(s):
            n = s[i + 1]
            if n in b'nrtbf':
                out.append({0x6e: 10, 0x72: 13, 0x74: 9, 0x62: 8, 0x66: 12}[n]); i += 2
            elif 0x30 <= n <= 0x37:
                j = i + 1; v = 0
                while j < len(s) and j < i + 4 and 0x30 <= s[j] <= 0x37:
                    v = v * 8 + s[j] - 0x30; j += 1
                out.append(v & 255); i = j
            else:
                out.append(n); i += 2
        else:
            out.append(c); i += 1
    return bytes(out)

tok = re.compile(rb'/(\w+)\s+[\d.]+\s+Tf|\[((?:\\.|[^\]])*)\]\s*TJ|\(((?:\\.|[^\\)])*)\)\s*Tj|<([0-9A-Fa-f]+)>\s*Tj|(T\*|Td|TD|Tm|ET)')
res = []
for oid in sorted(objs):
    body = objs[oid]
    if b'/Length' not in body or b'FontFile' in body:
        continue
    st = stream_of(body)
    if not st or (b'Tj' not in st and b'TJ' not in st):
        continue
    cur = None; line = []
    for m in tok.finditer(st):
        if m.group(1):
            cur = fontnames.get(m.group(1))
        elif m.group(2) is not None:
            for sm in re.finditer(rb'\(((?:\\.|[^\\)])*)\)|<([0-9A-Fa-f]*)>|(-?[\d.]+)', m.group(2)):
                if sm.group(1) is not None:
                    line.append(decode_str(lit(sm.group(1)), cur))
                elif sm.group(2) is not None:
                    line.append(decode_str(bytes.fromhex(sm.group(2).decode()), cur))
                elif sm.group(3) and float(sm.group(3)) < -200:
                    line.append(' ')
        elif m.group(3) is not None:
            line.append(decode_str(lit(m.group(3)), cur))
        elif m.group(4) is not None:
            line.append(decode_str(bytes.fromhex(m.group(4).decode()), cur))
        else:
            line.append('\n')
    res.append('=== stream obj %d ===\n' % oid + ''.join(line))
open(sys.argv[2], 'w', encoding='utf-8').write('\n'.join(res))
print('streams', len(res), 'fonts', len(fonts))
