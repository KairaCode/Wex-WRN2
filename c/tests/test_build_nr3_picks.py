# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""tools/build_nr3.py --train-lab80 picks: unchanged on V4.1 (40 x 192), every layer on Qwen3.8 (48 x 512)."""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
import build_nr3  # noqa: E402


class BuildNr3PicksTest(unittest.TestCase):
    def test_v41_picks_unchanged(self):
        rng = random.Random(20261107)
        old = [(L, e) for L in range(40) for e in sorted(rng.sample(range(192), 4))]
        self.assertEqual(build_nr3.lab80(40, 192), [old[i] for i in range(0, 160, 2)],
                         'V4.1 lab80 picks changed')

    def test_qwen38_picks_cover_every_layer(self):
        q = build_nr3.lab80(48, 512)
        self.assertEqual(len(set(q)), 80, q)
        self.assertEqual({L for L, _ in q}, set(range(48)), q)


if __name__ == '__main__':
    unittest.main()
