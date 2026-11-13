"""Fixed-length exponent code for FP8 E4M3 experts (ZipServ-style idea, our
estimate): per tile of T values, store the 4-bit exponent as a k-bit offset
from a per-tile base (window of 2^k - 1 codes, one code reserved as escape);
escaped values keep their full byte in a side list. Sign and 3 mantissa bits
stay raw. Bits per value = 1 + 3 + k + escape_rate * (8 + index bits) +
per-tile metadata. Fixed-rate decode (shift/mask/add), so it could run inside
a GEMV, unlike rANS.

usage: python -I fixed_exp_probe.py MODEL_DIR [experts=48] [OUT.json]"""
import json
import random
import sys
from pathlib import Path

import numpy as np

MODEL = Path(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 48
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else None
sys.argv = [sys.argv[0], str(MODEL), '1', '1']
sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_q38_experts as probe   # noqa: E402


def best_window(counts, k):
    """base maximizing the values whose exponent falls in [base, base + 2^k - 2]"""
    width = (1 << k) - 1
    sums = [counts[b:b + width].sum() for b in range(16)]
    return int(np.argmax(sums))


def main():
    cfg = json.loads((MODEL / 'config.json').read_text())
    text = cfg.get('text_config', cfg)
    tensors = probe.shard_tensors()
    rng = random.Random(99)
    keys = rng.sample([(l, e) for l in range(text['num_hidden_layers']) for e in range(text['num_experts'])], N)
    plan = [(key, probe.expert_ranges(tensors, *key)[0]) for key in keys]
    _, _, data = probe.run_side(plan, False)
    values = np.concatenate([np.frombuffer(b, np.uint8) for b in data.values()])
    exp = (values >> 3) & 0xF
    report = {'experts': N, 'values': int(values.size),
              'exponent_histogram': np.bincount(exp, minlength=16).tolist(), 'configs': []}
    for tile in (64, 128, 256, 1024):
        e = exp[: values.size // tile * tile].reshape(-1, tile)
        for k in (2, 3):
            width = (1 << k) - 1
            counts = np.stack([np.bincount(row, minlength=16) for row in e[:: max(1, len(e) // 4000)]])
            # per-tile base on a sample of tiles, then escape rate over that sample
            esc = 0
            for c in counts:
                b = best_window(c, k)
                esc += tile - c[b:b + width].sum()
            rate = esc / (len(counts) * tile)
            index_bits = int(np.ceil(np.log2(tile)))
            meta = 4 + 8  # base nibble + escape count byte, per tile (estimate)
            bits = 1 + 3 + k + rate * (8 + index_bits) + meta / tile
            report['configs'].append({'tile': tile, 'k': k, 'escape_rate': round(float(rate), 5),
                                      'bits_per_value': round(float(bits), 4), 'saving_vs_fp8': round(1 - bits / 8, 4)})
    text_out = json.dumps(report, indent=1)
    print(text_out)
    if OUT:
        OUT.write_text(text_out, encoding='utf-8')


if __name__ == '__main__':
    main()
