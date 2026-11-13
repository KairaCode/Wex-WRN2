# SPDX-License-Identifier: Apache-2.0
# Derived from colibri c/tests/test_v4_mirror_guard_source.py (https://github.com/JustVugg/colibri), Copyright 2026 Vincenzo Fornaro.
# Modified by Ludovico Zanichelli, 2026.
"""The decode mirror cache's growth guard counts what it grew between probes.

v4_gpu_expert_attach_cached_fill re-reads free VRAM only every 64 misses and
reuses the last answer in between. Measured against that stale answer alone,
the cache grew up to 64 mirrors (816 MiB) past an explicit
DSV4_CUDA_VRAM_RESERVE_MB or the 600 MiB default, past the card's free VRAM.
On the full DeepSeek V4 Flash at 24 GB the compressed-KV buffer then could not
grow ("kv comp allocation: out of memory", hundreds per request), its layers'
attention ran on the CPU, and greedy texts changed from run to run. No fixture
reaches a full card, so this pins the guard at source level:

- the mirrors appended since the last probe (v4_gpu_mirror_grown_bytes) count
  against the reserve, the probe resets them;
- only an appended slot adds to them (a recycled one reuses its VRAM), both on
  the plain create path and on the WEX direct fill path;
- the guard applies from 8 live mirrors on.
"""
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "deepseek_v4.c"


def find(text, needle, start=0, where="deepseek_v4.c"):
    at = text.find(needle, start)
    if at < 0:
        raise AssertionError(f"{where}: no {needle!r}")
    return at


def definition(text, signature):
    """The body of a function, past any prototype of it (which ends in ';' first)."""
    start = find(text, signature)
    while find(text, ";", start) < find(text, "{", start):
        start = find(text, signature, start + 1)
    return text[start:find(text, "\n}\n", start)]


class V4MirrorGuardSourceTest(unittest.TestCase):
    def setUp(self):
        self.text = SOURCE.read_text(encoding="utf-8")
        start = find(self.text, "static int v4_gpu_expert_attach_cached_fill(")
        self.body = self.text[start:find(self.text, "\n}\n", start)]

    def test_probe_resets_the_grown_count(self):
        probe = find(self.body, "last_free_mb = dsv4_cuda_mem_free_mb(cache->device);")
        close = find(self.body, "}", probe)
        self.assertIn("v4_gpu_mirror_grown_bytes = 0;", self.body[probe:close])

    def test_grown_mirrors_count_against_the_reserve(self):
        self.assertRegex(self.body, r"grown_mb = \(v4_gpu_mirror_grown_bytes \+ v4_gpu_mirror_bytes\(view\)")
        self.assertRegex(
            self.body,
            r"cache->count >= 8 && last_free_mb - grown_mb < needed")

    def test_only_appended_slots_add(self):
        self.assertIn("found = cache->count++;\n            appended = 1;", self.body)
        self.assertIn("if (appended) v4_gpu_mirror_grown_bytes += v4_gpu_mirror_bytes(view);",
                      self.body)
        # the producer fill (WEX direct) ends in v4_gpu_mirror_publish, which counts a new slot
        fill = definition(self.text, "static int v4_gpu_mirror_fill_unlocked(")
        self.assertIn("return v4_gpu_mirror_publish(cache, &h, view, ok);", fill)
        publish = definition(self.text, "static int v4_gpu_mirror_publish(")
        self.assertIn("if (!h->recycle) v4_gpu_mirror_grown_bytes += v4_gpu_mirror_bytes(view);",
                      publish)

    def test_mirror_bytes_match_the_upload(self):
        start = find(self.text, "static long long v4_gpu_mirror_bytes(")
        helper = self.text[start:find(self.text, "\n}\n", start)]
        self.assertIn("elements / 2 + elements / 32", helper)


if __name__ == "__main__":
    unittest.main()
