"""Engram on-disk layout + per-token access arithmetic from tensors.json (headers only) and config.json."""
import json, sys
tens = json.load(open(sys.argv[1], encoding='utf-8'))
cfg = json.load(open(sys.argv[2], encoding='utf-8'))['text_config']
eg = [t for t in tens if '.engram.' in t['name']]
for t in sorted(eg, key=lambda t: (t['file'], t['offset'])):
    print(f"{t['file']} {t['name']:40s} {t['dtype']:8s} {str(t['shape']):20s} bytes={t['bytes']:>15,d} abs_off={t['offset']:>15,d} off%4096={t['offset'] % 4096} off%256={t['offset'] % 256}")
n_cols = (cfg['engram_max_ngram_size'] - 1) * cfg['engram_n_heads']
row_w = cfg['engram_head_dim']  # fp8 bytes
row_s = cfg['engram_head_dim'] // 32
L = len(cfg['engram_layer_ids'])
print('hash columns (rows) per token per engram layer:', n_cols, '; layers:', L, '; rows/token:', n_cols * L)
print('payload bytes/token:', n_cols * L * (row_w + row_s), f'(weights {row_w} B + scale {row_s} B per row)')
print('separate reads/token (weight row and scale row are in different tensors):', 2 * n_cols * L)
print('bytes moved/token at 4 KiB granularity:', 2 * n_cols * L * 4096)
# fraction of 256-B weight rows that straddle a 4 KiB page, and of 8-B scale rows
for t in eg:
    if t['name'].endswith('embed.weight'):
        print(t['name'], 'weight rows straddling 4KiB pages: 0 if offset%256==0 ->', t['offset'] % 256 == 0)
# size of a combined 264-B row record, packed vs 4 KiB-page aligned (15 rows/page)
rows = sum(t['shape'][0] for t in eg if t['name'].endswith('embed.weight'))
print('total rows', rows, 'raw bytes', rows * 264, f'= {rows * 264 / 2**30:.2f} GiB')
print('combined 264-B rows, 15 per 4 KiB page:', -(-rows // 15) * 4096, f'= {-(-rows // 15) * 4096 / 2**30:.2f} GiB')
