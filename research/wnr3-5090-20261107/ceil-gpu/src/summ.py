import json, sys
r = json.load(open(sys.argv[1]))
print(json.dumps(r['info']))
base = None
for k, v in r['results'].items():
    nm = [x for x in v if x.endswith('rec_8streams')][0]
    print(f"{k:22s} regs {v['regs']:3d} lmem {v['local_bytes']:3d} smem {v['smem_bytes']:6d} occW {v.get('occ_warps_per_sm')} "
          f"exact {v['byte_exact']} | m1 {v['m1']['us_median']:7.1f} (b {v['m1']['us_batched_mean']:7.1f}, {v['m1']['sm_mhz_after']:.0f}MHz) "
          f"m3 {v['m3']['us_median']:7.1f} (b {v['m3']['us_batched_mean']:7.1f}) m24 {v['m24']['us_median']:8.1f} "
          f"(min {v['m24']['us_min']:8.1f} b {v['m24']['us_batched_mean']:8.1f} {v['m24']['sm_mhz_after']:.0f}MHz) "
          f"8str {v[nm]['us_median']:8.1f} GiB/s m3 {v['m3']['decoded_GiB_s']:5.1f} m24 {v['m24']['decoded_GiB_s']:5.1f}")
