# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Re-encode real WNR2 experts with shorter streams (weight_chunk) to test the
decoder's parallelism: more streams per record = more lanes = lower single-
record latency, at the cost of 6 bytes per stream.

Reads N records from the full WNR2 container, decodes each to its raw MXFP4
slab with the deployed helper (kaira_wex_cuda.dll), re-encodes it with the lab
GPU encoder (wex_nr2_tools.dll) at each chunk size, checks the round trip with
the encoder's verifier, and writes records_<chunk>.bin for bench_nr2_rechunk:
  u32 model_bytes, model, u32 count, then per record:
  72-byte entry, u32 raw_bytes, raw, record (entry.total_bytes, zero padded)

usage: python -I rechunk.py CONTAINER HELPER_DLL OUT_DIR [records=16] [chunks=4096,1024,512]"""
import ctypes
import json
import struct
import sys
from pathlib import Path

NR2 = Path(__file__).resolve().parents[1] / 'nr2'   # repo nr2/
sys.path.insert(0, str(NR2))
import nr2lib as L   # noqa: E402

CONTAINER, HELPER, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
N = int(sys.argv[4]) if len(sys.argv) > 4 else 16
CHUNKS = [tuple(int(x) for x in (c + ':' + str(min(2048, int(c.split(':')[0])))).split(':')[:2])
          for c in (sys.argv[5] if len(sys.argv) > 5 else '4096,1024,512').split(',')]   # w or w:s


def main():
    header, page, table, entries = L.read_container(CONTAINER)
    d = 64
    model_at = struct.unpack_from('<Q', page, d + 8)[0]
    with open(CONTAINER, 'rb') as f:
        f.seek(model_at)
        blob = f.read(L.MODEL_BYTES)
    tools = L.Tools(NR2 / 'wex_nr2_tools.dll')
    helper = ctypes.CDLL(str(HELPER))
    helper.wex_cuda_register_model.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    helper.wex_cuda_decode_to_host.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    assert helper.wex_cuda_register_model(L.ptr(blob), len(blob)) == 0
    step = len(entries) // N
    picked = [i for i in range(0, len(entries), step) if entries[i][L.F_FLAGS] == L.WEX_ENTRY_NEURAL_RANS2][:N]
    samples = []
    with open(CONTAINER, 'rb') as f:
        for i in picked:
            e = entries[i]
            raw_bytes = L.raw_size(e)
            aligned = L.align(raw_bytes)
            f.seek(e[L.F_OFFSET])
            record = f.read(e[L.F_TOTAL])
            slab = bytearray(aligned)
            slab[aligned - e[L.F_TOTAL]:] = record
            eb = L.ENTRY.pack(*e)
            assert helper.wex_cuda_decode_to_host(L.ptr(eb), L.ptr(slab), aligned) == 0, i
            samples.append((e, bytes(slab[:raw_bytes])))
    report = {'records': len(samples), 'original_bytes': sum(e[L.F_TOTAL] for e, _ in samples), 'chunks': {}}
    for chunk, schunk in CHUNKS:
        out = bytearray(struct.pack('<I', len(blob)) + blob + struct.pack('<I', len(samples)))
        total = 0
        for e, raw in samples:
            plain = list(e)
            plain[L.F_FLAGS] = 0
            plain[L.F_TOTAL] = L.align(len(raw))   # the encoder takes a raw (WEX1) entry
            eb = L.ENTRY.pack(*plain)
            capacity = L.align(len(raw)) + (1 << 20)
            rc, rec, written = tools.encode(blob, raw, eb, 0, chunk, schunk, capacity, gpu=True)
            assert rc == 0, (chunk, rc, written)
            padded = L.align(len(rec))
            assert padded < L.align(len(raw)), 'no longer smaller than raw'
            new = list(e)
            new[L.F_TOTAL] = padded
            new[L.F_OFFSET] = 0
            neb = L.ENTRY.pack(*new)
            padded_rec = rec + bytes(padded - len(rec))
            rc = tools.verify(blob, neb, padded_rec, raw)
            assert rc == 0, ('verify', chunk, rc)
            out += neb + struct.pack('<I', len(raw)) + raw + rec + bytes(padded - len(rec))
            total += padded
        (OUT / f'records_{chunk}_{schunk}.bin').write_bytes(out)
        report['chunks'][f'{chunk}:{schunk}'] = {'bytes': total, 'vs_original': round(total / report['original_bytes'], 5)}
    tools.release()
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
