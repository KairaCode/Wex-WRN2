# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""scripts/verify_wex1.py on the synthetic model of c/tests/test_repack_v4_wex.py: exact WEX1 and WEX2 pass,
one flipped weight byte fails. Run: python -I scripts/test_verify_wex1.py"""
import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fixtures = load("test_repack_v4_wex", HERE.parent / "c" / "tests" / "test_repack_v4_wex.py")
verify = load("verify_wex1", HERE / "verify_wex1.py")
wex = fixtures.wex


class VerifyWex1Tests(unittest.TestCase):
    def check(self, model, path):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            rc = verify.main(["--model-dir", str(model), "--wex", str(path)])
        return rc, out.getvalue()

    def test_exact_containers_pass_and_a_flipped_byte_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model"
            fixtures.make_model(model)
            for version in (1, 2):   # v2: expert 0 of layer 1 uses scale tiles, expert 1 stays raw
                path = Path(wex.repack(model, format_version=version)["output"])
                rc, out = self.check(model, path)
                self.assertEqual(rc, 0, out)
                self.assertIn('"checked": 2', out)
                entry = wex.validate_container(path)["entries"][3]
                self.assertEqual(entry[22], 0)
                data = bytearray(path.read_bytes())
                data[entry[0] + entry[2] + 5] ^= 0xFF   # a weight byte of the raw record
                path.write_bytes(data)
                rc, out = self.check(model, path)
                self.assertEqual(rc, 1)
                self.assertIn("FAIL bytes differ: layer 1 expert 1", out)


if __name__ == "__main__":
    unittest.main()
