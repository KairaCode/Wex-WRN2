# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Seed corpora for the fuzzing harnesses, from the conformance vectors.

    python -I wex4/fuzz/make_seeds.py OUT_DIR

OUT_DIR/file gets every file of wex4/vectors/MANIFEST.tsv (valid, invalid, legacy and the review findings): the
seeds of fuzz_wex4_file and fuzz_wex4_verify. OUT_DIR/wnr3 gets the stored bytes of every WNR3 record of the
accepted vectors: the seeds of fuzz_wnr3_record. A seed is named by the first 16 hex digits of the SHA-256 of its
bytes, so the same bytes are never stored twice and the paths stay short (Windows opens no path longer than
260 characters by default). Standard library only."""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import wex4  # noqa: E402

VECTORS = os.path.join(HERE, '..', 'vectors')


def put(folder, data):
    with open(os.path.join(folder, hashlib.sha256(data).hexdigest()[:16]), 'wb') as f:
        f.write(data)


def main():
    out = sys.argv[1]
    files, records = os.path.join(out, 'file'), os.path.join(out, 'wnr3')
    os.makedirs(files, exist_ok=True)
    os.makedirs(records, exist_ok=True)
    rows = [line.split('\t') for line in open(os.path.join(VECTORS, 'MANIFEST.tsv'), encoding='utf-8').read().splitlines()
            if line and not line.startswith('#')]
    for path, verdict in ((row[0], row[1]) for row in rows):
        data = open(os.path.join(VECTORS, path), 'rb').read()
        put(files, data)
        if verdict != 'accept':
            continue
        reader = wex4.Reader(io.BytesIO(data))
        for record in reader.records:
            if record[3] == 5:   # codec 5: WNR3
                put(records, data[record[0]:record[0] + record[1]])
    print(f'seeds: {len(os.listdir(files))} files in {files}, {len(os.listdir(records))} WNR3 records in {records}')


if __name__ == '__main__':
    main()
