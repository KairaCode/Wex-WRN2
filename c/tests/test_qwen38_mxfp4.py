# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""qwen38's MXFP4 routed experts: tools/repack_q38_mxfp4_wex.py, and the engine end to end.

As a unittest (numpy only, part of `make test`) it repacks synthetic Quark shards with a
matching FP8 snapshot and checks: every record is the Quark bytes in WEX1 order (w1 gate,
w2 down, w3 up, scales then nibbles), the container validates and verifies against its
manifest, a corrupted record or a layer subset is caught or kept absent, and the FP8 check
refuses swapped nibbles, a wrong dtype and a wrong shape.

As a script it runs qwen38 on tools/make_qwen38_tiny_mxfp4.py's fixture. The fixture's FP8 experts hold exactly the MXFP4 values, so the snapshot's own FP8 path is
the reference the MXFP4 path must reproduce (that path is itself checked against
transformers by make qwen38-tiny-fp8-check). Gates:
  * the FP8 run's tokens differ from the source fixture's ref.json: the new experts move
    the output, so the comparisons below can see them;
  * every MXFP4 run (Quark shards and WEX1 container; caps 1, 2, 4; single, batched and
    prefill paths) produces the FP8 run's tokens, and its last logits are within float
    rounding of the FP8 run's (cosine >= 0.99999; the kernels sum in another order);
  * the MXFP4 runs' last logits are identical to the bit across sources, caps and paths;
  * the repack manifest verifies, and the run logs say which experts ran.
When the model holds experts-mxfp4.nr3.wex (make qwen38-tiny-mxfp4-nr3-generate) its WNR3
records join as two more sources: decoded on the CPU (COLI_WEX_CUDA_LIB=0) and, with --helper,
through that kaira_wex_cuda.dll (tests/fake_wex_cuda.dll, or the real one on a CUDA box: then
the load waves decode on the GPU); both must give the same tokens and last-logit bits.

  python tests/test_qwen38_mxfp4.py --model ./qwen38_tiny_mxfp4 --source ./qwen38_tiny_fp8 --engine ./qwen38
          [--helper ./kaira_wex_cuda.dll] [--env NAME=VALUE ...]
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import unittest

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools"


def write_safetensors(path, tensors):
    """tensors: name -> (dtype, shape, bytes)"""
    header, blob = {}, b""
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [len(blob), len(blob) + len(data)]}
        blob += data
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    Path(path).write_bytes(struct.pack("<Q", len(raw)) + raw + blob)


@unittest.skipUnless(HAVE_NUMPY, "numpy is not installed")
class RepackTest(unittest.TestCase):
    LAYERS, EXPERTS, HIDDEN, INTER = 2, 3, 64, 32
    PREFIX = "model.language_model.layers."

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(TOOLS))
        import repack_q38_mxfp4_wex as tool
        cls.tool = tool
        cls.e4m3 = {}
        for byte, v in enumerate(tool.E4M3):
            if np.isfinite(v):
                cls.e4m3.setdefault((float(v), bool(np.signbit(v))), byte)

    def fp8_bytes(self, values):
        return bytes(self.e4m3[(float(v), bool(np.signbit(v)))] for v in values.ravel())

    def make(self, root, swap=False, scale_dtype="U8", down_shape=None):
        rng = np.random.default_rng(7)
        quark, fp8, root = {}, {}, Path(root)
        (root / "quark").mkdir()
        (root / "fp8").mkdir()
        config = {"text_config": {"num_hidden_layers": self.LAYERS, "num_experts": self.EXPERTS,
                                  "hidden_size": self.HIDDEN, "moe_intermediate_size": self.INTER}}
        (root / "fp8" / "config.json").write_text(json.dumps(config))
        dims = {"gate_proj": (self.INTER, self.HIDDEN), "up_proj": (self.INTER, self.HIDDEN),
                "down_proj": (self.HIDDEN, self.INTER)}
        self.records = {}
        for layer in range(self.LAYERS):
            for expert in range(self.EXPERTS):
                parts = {}
                for proj, (rows, cols) in dims.items():
                    name = f"{self.PREFIX}{layer}.mlp.experts.{expert}.{proj}"
                    nib = rng.integers(0, 256, (rows, cols // 2), dtype=np.uint8)
                    nib[(nib & 15) == 8] &= 0xF0   # Quark writes no -0
                    nib[(nib >> 4) == 8] &= 0x0F
                    sc = rng.integers(120, 125, (rows, cols // 32), dtype=np.uint8)
                    values = self.tool.mxfp4_values(nib.tobytes(), sc.tobytes(), rows, cols)
                    if swap:
                        values = self.tool.mxfp4_values(((nib >> 4) | (nib << 4)).astype(np.uint8).tobytes(),
                                                        sc.tobytes(), rows, cols)
                    shape = down_shape if proj == "down_proj" and down_shape else (rows, cols // 2)
                    quark[name + ".weight"] = ("U8", shape, nib.tobytes())
                    quark[name + ".weight_scale"] = (scale_dtype, (rows, cols // 32), sc.tobytes())
                    fp8[name + ".weight"] = ("F8_E4M3", (rows, cols), self.fp8_bytes(values))
                    fp8[name + ".weight_scale_inv"] = ("BF16", (1, 1), struct.pack("<H", 0x3F80))
                    parts[proj] = (sc.tobytes(), nib.tobytes())
                order = ("gate_proj", "down_proj", "up_proj")
                self.records[(layer, expert)] = b"".join(parts[p][0] for p in order) + b"".join(parts[p][1] for p in order)
        write_safetensors(root / "quark" / "model-00001.safetensors", quark)
        write_safetensors(root / "fp8" / "model-00001.safetensors", fp8)
        return root / "quark", root / "fp8", root / "out" / "experts.wex"

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            quark, fp8, out = self.make(tmp)
            manifest = self.tool.repack(quark, fp8, out, min_cos=0.999999)
            self.assertEqual(manifest["record_bytes"], 4096)
            self.assertTrue(all(v["min_cos"] == 1.0 for v in manifest["fp8_check"].values()))
            info = self.tool.wex.validate_container(out)
            with open(out, "rb") as f:
                for (layer, expert), want in self.records.items():
                    entry = info["entries"][layer * self.EXPERTS + expert]
                    self.assertEqual(entry[22], 0)
                    f.seek(entry[0])
                    self.assertEqual(f.read(len(want)), want)
            self.assertTrue(self.tool.verify(out)["ok"])
            data = bytearray(out.read_bytes())
            data[info["entries"][self.EXPERTS][0] + 100] ^= 1
            out.write_bytes(bytes(data))
            with self.assertRaisesRegex(ValueError, "layer 1"):
                self.tool.verify(out)

    def test_layer_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            quark, fp8, out = self.make(tmp)
            manifest = self.tool.repack(quark, fp8, out, layers_wanted=[1])
            info = self.tool.wex.validate_container(out)
            self.assertEqual(sorted(manifest["layers"]), ["1"])
            self.assertFalse(any(info["entries"][e][1] for e in range(self.EXPERTS)))
            self.assertTrue(self.tool.verify(out)["ok"])

    def test_refusals(self):
        for kwargs, pattern in (({"swap": True}, "cosine"), ({"scale_dtype": "I8"}, "dtypes"),
                                ({"down_shape": (32, 32)}, "shapes")):
            with self.subTest(**kwargs), tempfile.TemporaryDirectory() as tmp:
                quark, fp8, out = self.make(tmp, **kwargs)
                with self.assertRaisesRegex(ValueError, pattern):
                    self.tool.repack(quark, fp8, out)
                self.assertFalse(out.exists())


def run(engine, model, ref, cap, env_extra, dump):
    import numpy as np
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("COLI_WEX", "Q38_MXFP4", "Q38_EXPERT_INT4", "FAKE_WEX"))}
    env.update({"SNAP": str(model), "OMP_NUM_THREADS": "2", "DUMP": str(dump)}, **env_extra)
    proc = subprocess.run([str(engine), str(cap), "8", str(ref)], env=env, capture_output=True, text=True)
    tokens = None
    for line in proc.stdout.splitlines():
        if line.startswith("C engine :"):
            tokens = [int(t) for t in line.split(":", 1)[1].split()]
    if tokens is None:
        sys.exit(f"FAIL {env_extra} cap {cap}: no 'C engine' line\n{proc.stdout}\n{proc.stderr}")
    return tokens, np.fromfile(dump, np.float32), proc


def main():
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path, help="the FP8 fixture the model was made from")
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--helper", type=Path, help="a kaira_wex_cuda.dll for the WNR3 source")
    parser.add_argument("--env", action="append", default=[], help="NAME=VALUE for every run (e.g. COLI_CUDA=1)")
    args = parser.parse_args()
    common = dict(item.split("=", 1) for item in args.env)
    model, work, args.engine = args.model.resolve(), args.model.resolve() / "logits", args.engine.resolve()
    work.mkdir(exist_ok=True)
    source_ref = json.loads((args.source / "ref.json").read_text(encoding="utf-8"))
    prompt, n_new = source_ref["prompt_ids"], len(source_ref["generated_ids"])
    probe = work / "ref_probe.json"
    probe.write_text(json.dumps({"schema_version": 1, "prompt_ids": prompt, "full_ids": source_ref["full_ids"]}))

    fp8_tokens, fp8_logits, _ = run(args.engine, model, probe, 2, dict(common), work / "fp8.f32")
    if fp8_tokens == source_ref["generated_ids"]:
        sys.exit("FAIL the fixture's experts do not change the tokens: the gate below could not see them")
    print(f"ok   FP8 control: {fp8_tokens} (source fixture: {source_ref['generated_ids']})")
    ref = work / "ref_fp8.json"
    ref.write_text(json.dumps({"schema_version": 1, "prompt_ids": prompt, "full_ids": prompt + fp8_tokens}))

    sources = {"quark": {"Q38_MXFP4_DIR": str(model / "quark")},
               "wex": {"COLI_WEX_PATH": str(model / "experts-mxfp4.wex")}}
    nr3 = model / "experts-mxfp4.nr3.wex"
    if nr3.exists():
        sources["wnr3-cpu"] = {"COLI_WEX_PATH": str(nr3), "COLI_WEX_CUDA_LIB": "0"}
        if args.helper:
            sources["wnr3-helper"] = {"COLI_WEX_PATH": str(nr3), "COLI_WEX_CUDA_LIB": str(args.helper.resolve())}
    modes = {"single": {"Q38_PREFILL_BATCH": "0", "Q38_EXPERT_PARALLEL_READS": "0"},
             "batched": {"Q38_PREFILL_BATCH": "0"}, "prefill": {"Q38_PREFILL_BATCH": "1"}}
    first, n = None, 0
    for name, src in sources.items():
        for cap in (1, 2, 4):
            for mode, extra in modes.items():
                tokens, logits, proc = run(args.engine, model, ref, cap, {**common, **src, **extra},
                                           work / f"{name}-{cap}-{mode}.f32")
                label = f"{name} cap {cap} {mode}"
                if proc.returncode or tokens != fp8_tokens:
                    sys.exit(f"FAIL {label}: tokens {tokens} != FP8 {fp8_tokens}\n{proc.stdout}\n{proc.stderr}")
                if "MXFP4 experts from" not in proc.stderr:
                    sys.exit(f"FAIL {label}: the log does not say the MXFP4 experts ran\n{proc.stderr}")
                if name.startswith("wnr3") and "MXFP4 experts from WNR3" not in proc.stderr:
                    sys.exit(f"FAIL {label}: the log does not say the WNR3 container ran\n{proc.stderr}")
                if name == "wex" and "MXFP4 experts from WEX1" not in proc.stderr:
                    sys.exit(f"FAIL {label}: the log does not name the WEX1 container\n{proc.stderr}")
                if (name == "wnr3-helper" and cap == 4 and mode != "single" and "COLI_CUDA" not in common   # the tier serves them all
                        and "one helper call" not in proc.stderr):
                    sys.exit(f"FAIL {label}: the load waves did not decode through the helper\n{proc.stderr}")
                cos = float(np.dot(logits, fp8_logits) / (np.linalg.norm(logits) * np.linalg.norm(fp8_logits)))
                if cos < 0.99999:
                    sys.exit(f"FAIL {label}: last logits cosine {cos:.8f} vs FP8")
                if first is None:
                    first = logits
                elif not np.array_equal(logits.view(np.uint32), first.view(np.uint32)):
                    sys.exit(f"FAIL {label}: last logits differ from the first MXFP4 run's bits")
                n += 1
    print(f"ok   {n} MXFP4 runs ({', '.join(sources)}; caps 1, 2, 4; single, batched, prefill): FP8 tokens, "
          f"last logits identical across runs, cosine vs FP8 "
          f"{float(np.dot(first, fp8_logits) / (np.linalg.norm(first) * np.linalg.norm(fp8_logits))):.8f}")
    verify = subprocess.run([sys.executable, str(HERE.parent / "tools" / "repack_q38_mxfp4_wex.py"),
                             "--verify", str(model / "experts-mxfp4.wex")], capture_output=True, text=True)
    if verify.returncode:
        sys.exit(f"FAIL manifest verify\n{verify.stdout}\n{verify.stderr}")
    print("ok   the container matches its manifest")
    print("test_qwen38_mxfp4.py: ok")


if __name__ == "__main__":
    main()
