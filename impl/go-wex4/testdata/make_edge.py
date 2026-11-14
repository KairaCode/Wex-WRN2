#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Writes testdata/ragged.wex4 and testdata/extra-page.wex4 with the repository's Python writer.

ragged.wex4 (valid): four WBP1 records whose decoded size is not a multiple of the plane count (2, 4, 8 and
8 planes), with chunk_log2 = 11, so that the planes of one record have different lengths and several
streams. One plane of the 4-plane and of the 8-plane records is random bytes, which the writer stores raw.

extra-page.wex4 (must be refused): one WBP1 record of 1 plane, written with an encoder that appends one page
of zero bytes to the record. stored_bytes is then one page more than the used bytes rounded up to a page,
and still less than the raw record. Every hash of the file is right.

    python -I make_edge.py OUT_DIR

prints the SHA-256 of every tensor's bytes. The writer is wex4/wex4.py, imported from this checkout and
used through its public calls (Tensor, Record, write, bp_encode); no reader of that module is called.
Standard library only. Deterministic.
"""
import hashlib
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "wex4"))
import wex4  # noqa: E402

U8 = 2


def data(rng: random.Random, size: int, planes: int, noisy) -> bytes:
    """Compressible bytes with 255 frequent; the bytes of plane `noisy` (None: no plane) are random."""
    out = bytearray(rng.choices([0, 1, 2, 255, 254, 7], [40, 20, 10, 20, 5, 5], k=size))
    for k in range(size if noisy is None else noisy, size, planes):
        out[k] = rng.randrange(256)
    return bytes(out)


class ExtraPage:
    """The reference encoder, with one page of zero bytes after the record."""

    @staticmethod
    def bp_encode(raw, planes, chunk_log2, owner, transform=0):
        return wex4.bp_encode(raw, planes, chunk_log2, owner, transform) + bytes(4096)


def record(name: str, planes: int, raw: bytes):
    print(name, hashlib.sha256(raw).hexdigest())
    return wex4.Record([wex4.Tensor(name, U8, (len(raw),), len(raw))], len(raw), lambda: raw, codec=("bp", planes, 0))


def main() -> int:
    out = sys.argv[1]
    rng = random.Random(20261114)
    ragged = [("r2", 2, 5001, None), ("r4", 4, 10001, 1), ("r8", 8, 9999, 6), ("r8.long", 8, 20003, 0)]
    records = [record(name, planes, data(rng, size, planes, noisy)) for name, planes, size, noisy in ragged]
    wex4.write(os.path.join(out, "ragged.wex4"), records, chunk_log2=11)
    wex4.write(os.path.join(out, "extra-page.wex4"), [record("x", 1, data(rng, 10000, 1, None))], encoder=ExtraPage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
