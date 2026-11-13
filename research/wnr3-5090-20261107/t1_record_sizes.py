"""Task 1: V4.1 WNR2 record-size scan (read-only).

Reads the WEX3 header, the 7680-entry table and the 32-B header of every WNR2 record,
then reports on-disk (4096-aligned) and used (unpadded) sizes and the tier-slot
capacity of fixed max-record slots vs byte-sized slots.

    python -I t1_record_sizes.py E:\\kaira_wex\\v41-reap\\v4_flash_experts.nr2.wex [out.json]
"""
import json, math, struct, sys

HEADER = struct.Struct('<IIIIQQII')          # 40 B
ENTRY = struct.Struct('<QIII6H6H3I3II')        # 72 B
REC = struct.Struct('<IIIIII3BBI')             # 32 B WNR2 record header
WNR2_MAGIC = 0x32524E57


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(math.ceil(q * len(s))) - 1)]


def main(path, out=None):
    with open(path, 'rb', buffering=0) as f:
        hdr = HEADER.unpack(f.read(4096)[:40])
        magic, ver, nl, ne, aligned, table_off, flags, _ = hdr
        assert magic == 0x57455833 and ver == 3 and flags & 16, hdr
        n = nl * ne
        f.seek(table_off)
        tab = f.read(n * ENTRY.size)
        entries = [ENTRY.unpack_from(tab, i * ENTRY.size) for i in range(n)]
        rows = []
        for i, e in enumerate(entries):
            off, total, sb, wb = e[0], e[1], e[2], e[3]
            scale_shapes = e[4:10]
            weight_shapes = e[10:16]
            codec = e[-1] & 0xff
            raw = sb + wb
            if codec != 4:
                rows.append(dict(i=i, codec=codec, total=total, used=total, raw=raw))
                continue
            f.seek(off)
            rh = REC.unpack(f.read(4096)[:32])
            assert rh[0] == WNR2_MAGIC, (i, hex(rh[0]))
            _, mid, spay, wpay, wchunk, schunk, m0, m1, m2, z27, z28 = rh
            ns = sum(math.ceil(scale_shapes[2 * m] * scale_shapes[2 * m + 1] / schunk) for m in range(3))
            # weight_shapes are byte shapes (2 nibbles per byte): symbols = rows*cols*2
            nw = sum(math.ceil(weight_shapes[2 * m] * weight_shapes[2 * m + 1] * 2 / wchunk) for m in range(3))
            used = 32 + 2 * (ns + nw) + spay + wpay
            assert total == (used + 4095) // 4096 * 4096, (i, total, used)
            rows.append(dict(i=i, codec=4, total=total, used=used, raw=raw, model=mid,
                             median=[m0, m1, m2], streams=ns + nw, spay=spay, wpay=wpay))
    raw = rows[0]['raw']
    assert all(r['raw'] == raw for r in rows)
    tot = [r['total'] for r in rows]
    used = [r['used'] for r in rows]
    med = {}
    for r in rows:
        for m, v in enumerate(r.get('median', [])):
            med.setdefault(f'm{m}', {}).setdefault(v, 0)
            med[f'm{m}'][v] += 1

    def stats(v):
        return dict(max=max(v), p99=pct(v, 0.99), p999=pct(v, 0.999), mean=sum(v) / len(v),
                    min=min(v), argmax=v.index(max(v)))

    st_tot, st_used = stats(tot), stats(used)
    res = dict(container=path, records=n, aligned_expert_bytes=aligned, raw_record_bytes=raw,
               codecs={c: sum(1 for r in rows if r['codec'] == c) for c in set(r['codec'] for r in rows)},
               on_disk_aligned=st_tot, used_unpadded=st_used,
               ratio_mean_on_disk=st_tot['mean'] / raw, ratio_mean_used=st_used['mean'] / raw,
               ratio_max_on_disk=st_tot['max'] / raw,
               median_bytes=med)
    # capacity per fixed byte budget, relative to raw slots of aligned_expert_bytes (= 18,800,640)
    base = aligned
    res['capacity_vs_raw_slots'] = {
        'fixed_slot_max_on_disk': base / st_tot['max'] - 1,
        'fixed_slot_p99_on_disk': base / st_tot['p99'] - 1,
        'byte_sized_on_disk_mean': base / st_tot['mean'] - 1,
        'byte_sized_used_mean': base / st_used['mean'] - 1,
        'byte_sized_used_mean_64B_align': base / (sum((u + 63) // 64 * 64 for u in used) / n) - 1,
    }
    res['byte_sized_gain_over_max_slots'] = st_tot['max'] / st_tot['mean'] - 1
    am = rows[st_tot['argmax']]['i']
    res['argmax_layer_expert'] = [am // ne, am % ne]
    print(json.dumps(res, indent=1))
    if out:
        json.dump(dict(summary=res, rows=rows), open(out, 'w'))


if __name__ == '__main__':
    main(*sys.argv[1:])
