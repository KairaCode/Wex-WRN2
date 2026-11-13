# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Small, model-independent proofs for WEX1/WEX2 repacking and publication."""

import contextlib
import importlib.util
import io
import json
import itertools
from pathlib import Path
import random
import struct
import sys
import tempfile
import unittest
from unittest import mock

WRITER_PATH = Path(__file__).resolve().parents[1] / "tools" / "repack_v4_wex.py"
SPEC = importlib.util.spec_from_file_location("repack_v4_wex", WRITER_PATH)
wex = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = wex
SPEC.loader.exec_module(wex)


def write_safetensors(path, tensors):
    header, payload = {}, bytearray()
    for name, dtype, shape, data in tensors:
        start = len(payload)
        payload.extend(data)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [start, len(payload)]}
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def make_model(directory, mixed=True):
    directory.mkdir(exist_ok=True)
    (directory / "config.json").write_text(json.dumps({"num_hidden_layers": 2, "n_routed_experts": 2}))
    tensors, expected = [], []
    for expert in range(2):
        scale_parts, weight_parts = [], []
        for matrix, name in enumerate(("w1", "w2", "w3")):
            scales = bytes(range(256)) * 16 if mixed and expert else bytes((120 + matrix,)) * 4096
            weights = bytes((index + expert + matrix) & 255 for index in range(64 * 1024))
            prefix = f"layers.1.ffn.experts.{expert}.{name}"
            tensors.append((prefix + ".scale", "F8_E8M0", (64, 64), scales))
            tensors.append((prefix + ".weight", "I8", (64, 1024), weights))
            scale_parts.append(scales)
            weight_parts.append(weights)
        expected.append((b"".join(scale_parts), b"".join(weight_parts)))
    write_safetensors(directory / "model.safetensors", tensors)
    return tensors, expected


V41_TEXT = {"model_type": "deepseek_v41_text", "num_hidden_layers": 2, "n_routed_experts": 3,
            "hidden_size": 64, "moe_intermediate_size": 32}


def make_v41_model(directory, config):
    """V4.1 layout in miniature: nested text_config, H=64, I=32, plus MTP experts that must stay out."""
    directory.mkdir(exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config))
    tensors, expected = [], []
    shapes = {"w1": (32, 32), "w2": (64, 16), "w3": (32, 32)}
    for prefix, layers, experts in (("layers", 2, 3), ("mtp", 1, 4)):
        for layer, expert in itertools.product(range(layers), range(experts)):
            scale_parts, weight_parts = [], []
            for matrix, (name, (rows, cols)) in enumerate(shapes.items()):
                scales = bytes((118 + (layer + expert + matrix) % 5,)) * (rows * cols // 16)
                weights = bytes((index * 7 + layer + expert + matrix) & 255 for index in range(rows * cols))
                base = f"{prefix}.{layer}.ffn.experts.{expert}.{name}"
                tensors.append((base + ".scale", "F8_E8M0", (rows, cols // 16), scales))
                tensors.append((base + ".weight", "I8", (rows, cols), weights))
                scale_parts.append(scales)
                weight_parts.append(weights)
            if prefix == "layers":
                expected.append((b"".join(scale_parts), b"".join(weight_parts)))
    write_safetensors(directory / "model.safetensors", tensors)
    return expected


def decoded_payload(path, entry):
    offset, _, scale_bytes, weight_bytes = entry[:4]
    flags = entry[22]
    with Path(path).open("rb") as source:
        source.seek(offset)
        if flags == wex.WEX_PALETTE3:
            magic, encoded_scales, _, reserved = wex.palette3.PREFIX.unpack(source.read(16))
            assert magic == wex.palette3.MAGIC and reserved == 0
            tables = source.read(wex.palette3.TABLE_BYTES)
            scales = wex.unpack_scale_tiles(source.read(encoded_scales), scale_bytes)
            weights, cursor = [], 0
            for matrix in range(3):
                groups = entry[4 + matrix * 2] * entry[5 + matrix * 2]
                weights.append(wex.palette3.unpack_matrix(scales[cursor:cursor + groups],
                    tables[matrix * 1024:(matrix + 1) * 1024], source.read(groups * 12)))
                cursor += groups
            return scales, b"".join(weights)
        elif flags:
            magic, decoded, encoded = wex.SCALE_PREFIX.unpack(source.read(wex.SCALE_PREFIX.size))
            assert magic == wex.WEX_SCALE_TILE_MAGIC and decoded == scale_bytes
            scales = wex.unpack_scale_tiles(source.read(encoded), decoded)
            source.seek(offset + (flags >> 8) * wex.WEX_ALIGN)
        else:
            scales = source.read(scale_bytes)
        return scales, source.read(weight_bytes)


class NeuralRansTests(unittest.TestCase):
    def test_model_cdf_guards_and_dependency_gate(self):
        for raw in (b"", bytes(512 + wex.neural_rans.CDF_BYTES), bytes(513)):
            with self.assertRaises(ValueError):
                wex.neural_rans.validate_model(raw)
        with self.assertRaisesRegex(ValueError, "requires neural_cdf"):
            wex.repack("unused", format_version=3, weight_codec="neural-rans")

    def test_lossless_publication_alias_and_corruption(self):
        # Independent one-state reference encoder; GPU dependencies are mocked
        # here, while lab/wex_neural/verify_lossless.py exercises the real CUDA.
        cdf = [0, 4081] + list(range(4082, 4097))
        model = struct.pack("<256H", *([0] * 256)) + struct.pack("<17H", *cdf) * 17
        class ReferenceEncoder:
            mapping = [0] * 256
            def __init__(self, *args):
                pass
            def encode(self, scales, weights, scale_stream):
                assert not any(weights)
                chunk = 512
                streams = (len(weights) * 2 + chunk - 1) // chunk
                encoded, offsets = [], [0]
                for stream in range(streams):
                    state, emitted = 1 << 23, bytearray()
                    count = min(chunk, len(weights) * 2 - stream * chunk)
                    for _ in range(count):
                        while state >= ((1 << 23 >> 12) << 8) * 4081:
                            emitted.append(state & 255); state >>= 8
                        state = (state // 4081 << 12) + state % 4081
                    payload = state.to_bytes(4, "big") + emitted[::-1]
                    encoded.append(payload); offsets.append(offsets[-1] + len(payload))
                return (wex.neural_rans.PREFIX.pack(wex.neural_rans.MAGIC, len(scale_stream), offsets[-1],
                        1, streams, chunk, 0, 0) + model + struct.pack(f"<{streams + 1}I", *offsets) +
                        scale_stream + b"".join(encoded))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensors, _ = make_model(root, mixed=False)
            tensors = [(name, dtype, shape, bytes(len(data)) if name.endswith(".weight") else data)
                       for name, dtype, shape, data in tensors]
            write_safetensors(root / "model.safetensors", tensors)
            with mock.patch.object(wex.neural_rans, "Encoder", ReferenceEncoder):
                result = wex.repack(root, format_version=3, weight_codec="neural-rans",
                                    neural_cdf="compiled", neural_encoder_dll="cuda")
            self.assertFalse(result["lossy"])
            self.assertEqual(result["neural_rans_experts"], 2)
            self.assertEqual(result["deduplicated_experts"], 1)
            manifest = wex.validate_container(result["output"], verify_payload=True)
            entry = manifest["entries"][2]
            with Path(result["output"]).open("r+b") as stream:
                stream.seek(entry[0] + 544)
                stream.write(b"\x01\x00")
            with self.assertRaisesRegex(ValueError, "CDF"):
                wex.validate_container(result["output"])


class ScaleTileTests(unittest.TestCase):
    def test_all_bit_widths_and_partial_tiles(self):
        rng = random.Random(749)
        for bits in range(9):
            maximum = (1 << bits) - 1
            values = [0, maximum] + [rng.randrange(maximum + 1) for _ in range(255)]
            raw = bytes(values)
            with self.subTest(bits=bits):
                packed = wex.pack_scale_tiles(raw)
                self.assertEqual(packed[1], bits)
                self.assertEqual(wex.unpack_scale_tiles(packed, len(raw)), raw)
        self.assertEqual(wex.unpack_scale_tiles(b"", 0), b"")

    def test_lsb_golden_vector(self):
        self.assertEqual(wex.pack_scale_tiles(bytes((10, 11, 12, 13, 14))), bytes((10, 3, 0x88, 0x46)))

    def test_malformed_streams_rejected(self):
        cases = [(b"", 1), (b"\x00", 1), (b"\x00\x09", 1),
                 (b"\x00\x01", 1), (b"\xff\x01\x01", 1),
                 (b"\x00\x03\x80", 1), (b"\x00\x00\x00", 1)]
        for stream, count in cases:
            with self.subTest(stream=stream):
                with self.assertRaises(ValueError):
                    wex.unpack_scale_tiles(stream, count)

    def test_compression_needs_whole_pages_and_actual_record_gain(self):
        self.assertEqual(wex._scale_storage(bytes(4095), 65536, 2)[3], 0)
        self.assertEqual(wex._scale_storage(bytes(4096), 65536, 2)[3], 0)
        self.assertEqual(wex._scale_storage(bytes(range(256)) * 48, 196608, 2)[3], 0)
        self.assertEqual(wex._scale_storage(bytes(12288), 196608, 1)[3], 0)
        self.assertEqual(wex._scale_storage(bytes(12288), 196608, 2)[3], 257)


class PaletteTests(unittest.TestCase):
    def test_discrete_optimizer_matches_exhaustive_reference(self):
        hist = [3, 5, 7, 2, 1, 0, 0, 0, 11, 8, 9, 4, 2, 1, 1, 0]
        levels = wex.palette3.LEVELS
        palette, mapping = wex.palette3.fit_palette(hist)
        actual = sum(hist[s] * (levels[s] - levels[palette[mapping[s]]]) ** 2 for s in range(16))
        optimum = min(sum(hist[s] * min((levels[s] - levels[c]) ** 2 for c in candidates)
                          for s in range(16))
                      for candidates in itertools.combinations(wex.palette3.ORDER, 8))
        self.assertEqual(actual, optimum)
        self.assertEqual(len(set(palette)), 8)

    def test_eight_symbol_payload_can_be_exact(self):
        scales = bytes([120, 123])
        raw = bytes([0x10, 0x32, 0x54, 0x76]) * 8
        tables, packed, error = wex.palette3.pack_matrix(scales, raw)
        self.assertEqual(error, 0)
        self.assertEqual(len(packed), 24)
        self.assertEqual(wex.palette3.unpack_matrix(scales, tables, packed), raw)


class RepackerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="wex-repacker-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = self.root / "model"
        self.tensors, self.expected = make_model(self.model)

    def assert_no_temps(self):
        self.assertEqual(list(self.root.rglob("*.tmp")), [])

    def test_v3_lossless_dedup_and_random_access_aliases(self):
        tensors = []
        originals = {name: data for name, _, _, data in self.tensors}
        for name, dtype, shape, data in self.tensors:
            tensors.append((name, dtype, shape, originals[name.replace("experts.1.", "experts.0.")]))
        write_safetensors(self.model / "model.safetensors", tensors)
        result = wex.repack(self.model, format_version=3)
        valid = wex.validate_container(result["output"], verify_payload=True)
        self.assertEqual(result["deduplicated_experts"], 1)
        self.assertFalse(result["lossy"])
        self.assertEqual(valid["entries"][2], valid["entries"][3])
        for entry in valid["entries"][2:]:
            self.assertEqual(decoded_payload(result["output"], entry), self.expected[0])
        # Aliases require the explicit V3 header capability.
        payload = bytearray(Path(result["output"]).read_bytes())
        struct.pack_into("<I", payload, 32, wex.WEX_SCALE_TILES)
        Path(result["output"]).write_bytes(payload)
        with self.assertRaises(ValueError):
            wex.validate_container(result["output"])

    def test_v3_palette_explicit_budget_and_raw_fallback(self):
        for budget in (None, -1, float("nan"), True):
            with self.assertRaises(ValueError):
                wex.repack(self.model, format_version=3, weight_codec="palette3", max_weight_relative_error=budget)
        with self.assertRaises(ValueError):
            wex.repack(self.model, format_version=2, weight_codec="palette3", max_weight_relative_error=1)
        result = wex.repack(self.model, format_version=3, weight_codec="palette3", max_weight_relative_error=.5)
        self.assertEqual(result["palette3_experts"], 1)  # expert 1 has non-finite UE8M0=255: native fallback
        self.assertTrue(result["lossy"])
        entries = wex.validate_container(result["output"], verify_payload=True)["entries"][2:]
        for expert, entry in enumerate(entries):
            scales, weights = decoded_payload(result["output"], entry)
            self.assertEqual(scales, self.expected[expert][0])
            self.assertEqual(len(weights), len(self.expected[expert][1]))
            if entry[22] == wex.WEX_PALETTE3:
                self.assertNotEqual(weights, self.expected[expert][1])
            else:
                self.assertEqual(weights, self.expected[expert][1])
        fallback = wex.repack(self.model, format_version=3, weight_codec="palette3", max_weight_relative_error=0.)
        self.assertEqual(fallback["palette3_experts"], 0)
        self.assertFalse(fallback["lossy"])
        for expert, entry in enumerate(wex.validate_container(fallback["output"])["entries"][2:]):
            self.assertEqual(decoded_payload(fallback["output"], entry), self.expected[expert])

    def test_v3_hash_is_not_an_equality_proof(self):
        fake = mock.Mock()
        fake.digest.return_value = bytes(32)
        with mock.patch.object(wex.hashlib, "sha256", return_value=fake):
            result = wex.repack(self.model, format_version=3)
        self.assertEqual(result["deduplicated_experts"], 0)
        entries = wex.validate_container(result["output"])["entries"][2:]
        self.assertNotEqual(entries[0][0], entries[1][0])
        for expert, entry in enumerate(entries):
            self.assertEqual(decoded_payload(result["output"], entry), self.expected[expert])

    def test_v3_corrupt_prefix_and_alias_metadata_rejected(self):
        result = wex.repack(self.model, format_version=3, weight_codec="palette3", max_weight_relative_error=.5)
        path = Path(result["output"])
        original = path.read_bytes()
        entry = wex.validate_container(path)["entries"][2]
        for position, value in ((0, 0), (4, 0), (8, 1), (12, 1)):
            damaged = bytearray(original)
            struct.pack_into("<I", damaged, entry[0] + position, value)
            path.write_bytes(damaged)
            with self.assertRaises(ValueError):
                wex.validate_container(path, verify_payload=True)
        path.write_bytes(original)
        dry = wex.repack(self.model, self.root / "absent.wex", format_version=3, dry_run=True,
                         weight_codec="palette3", max_weight_relative_error=.5)
        self.assertEqual(dry["file_bytes"], result["file_bytes"])
        self.assertFalse((self.root / "absent.wex").exists())

    def test_v1_default_and_v2_mixed_roundtrip(self):
        v1 = wex.repack(self.model)
        v2 = wex.repack(self.model, format_version=2)
        self.assertEqual(Path(v1["output"]).name, "v4_flash_experts.wex")
        self.assertEqual(Path(v2["output"]).name, "v4_flash_experts.v2.wex")
        self.assertEqual(v1["saved_bytes"], 0)
        self.assertEqual(v2["compressed_experts"], 1)
        self.assertEqual(v2["saved_bytes"], 8192)
        for result in (v1, v2):
            parsed = wex.validate_container(result["output"], verify_payload=True)
            self.assertEqual(parsed["file_bytes"], result["file_bytes"])
            self.assertEqual(parsed["record_bytes"], 208896)
            self.assertTrue(all(not any(entry) for entry in parsed["entries"][:2]))
            for entry, expected in zip(parsed["entries"][2:], self.expected):
                self.assertEqual(decoded_payload(result["output"], entry), expected)
                self.assertEqual(entry[16:19], (0, 4096, 8192))
                self.assertEqual(entry[19:22], (12288, 77824, 143360))
        self.assertEqual(wex.validate_container(v1["output"])["entries"][2][22], 0)
        self.assertEqual(wex.validate_container(v2["output"])["entries"][2][22], 257)
        self.assertEqual(wex.validate_container(v2["output"])["entries"][3][22], 0)
        self.assertEqual(Path(v1["output"]).read_bytes()[40:4096], bytes(4096 - 40))
        self.assertNotEqual(Path(v2["output"]).read_bytes()[40:56], bytes(16))
        self.assert_no_temps()

    def test_v2_build_ids_differ_and_mirror_is_exact(self):
        output = self.root / "custom.wex"
        mirror = self.root / "mirror"
        mirror.mkdir()
        previous_default = mirror / "v4_flash_experts.wex"
        previous_default.write_bytes(b"existing default mirror")
        wex.repack(self.model, output, 2, mirror)
        first = output.read_bytes()
        self.assertEqual(first, (mirror / "custom.wex").read_bytes())
        self.assertEqual(previous_default.read_bytes(), b"existing default mirror")
        wex.repack(self.model, output, 2)
        second = output.read_bytes()
        self.assertNotEqual(first[40:56], second[40:56])
        self.assertEqual(first[:40] + first[56:], second[:40] + second[56:])
        self.assert_no_temps()

    def test_default_mirror_names_follow_version(self):
        mirror = self.root / "mirror"
        for version, name in ((1, "v4_flash_experts.wex"), (2, "v4_flash_experts.v2.wex")):
            result = wex.repack(self.model, format_version=version, mirror_dir=mirror)
            self.assertEqual(Path(result["output"]).read_bytes(), (mirror / name).read_bytes())

    def test_dry_run_reads_no_weight_payload_and_writes_nothing(self):
        original = wex._read_tensor
        read_dtypes = []

        def read_scale_only(ref, handles):
            self.assertIn(ref.dtype, ("F8_E8M0", "F8_E8M0FNU"))
            read_dtypes.append(ref.dtype)
            return original(ref, handles)

        output = self.root / "new-directory" / "estimate.wex"
        with mock.patch.object(wex, "_read_tensor", side_effect=read_scale_only):
            estimate = wex.repack(self.model, output, 2, self.root / "mirror", dry_run=True)
        self.assertEqual(len(read_dtypes), 6)
        self.assertFalse(output.parent.exists())
        self.assertFalse((self.root / "mirror").exists())
        actual = wex.repack(self.model, output, 2)
        self.assertEqual(estimate["file_bytes"], actual["file_bytes"])
        self.assertEqual(estimate["saved_bytes"], actual["saved_bytes"])

    def test_atomic_replace_failure_preserves_existing_primary(self):
        output = self.root / "existing.wex"
        output.write_bytes(b"previous valid container sentinel")
        with mock.patch.object(wex.os, "replace", side_effect=OSError("publish denied")):
            with self.assertRaisesRegex(OSError, "publish denied"):
                wex.repack(self.model, output, 2)
        self.assertEqual(output.read_bytes(), b"previous valid container sentinel")
        self.assert_no_temps()

    def test_read_failure_preserves_existing_primary(self):
        output = self.root / "existing.wex"
        output.write_bytes(b"previous")
        with mock.patch.object(wex, "_read_tensor", side_effect=ValueError("short tensor read")):
            with self.assertRaisesRegex(ValueError, "short tensor read"):
                wex.repack(self.model, output, 2)
        self.assertEqual(output.read_bytes(), b"previous")
        self.assert_no_temps()

    def test_mirror_copy_failure_preserves_existing_mirror(self):
        primary = Path(wex.repack(self.model, format_version=2)["output"])
        mirror = self.root / "mirror" / "v4_flash_experts.v2.wex"
        mirror.parent.mkdir()
        mirror.write_bytes(b"old mirror")

        def partial_copy(source, destination, **unused):
            destination.write(source.read(100))
            raise OSError("copy interrupted")

        with mock.patch.object(wex.shutil, "copyfileobj", side_effect=partial_copy):
            with self.assertRaisesRegex(OSError, "copy interrupted"):
                wex._atomic_mirror(primary, mirror)
        self.assertEqual(mirror.read_bytes(), b"old mirror")
        self.assert_no_temps()

    def test_mirror_replace_failure_preserves_existing_mirror(self):
        primary = Path(wex.repack(self.model, format_version=2)["output"])
        mirror = self.root / "mirror" / "v4_flash_experts.v2.wex"
        mirror.parent.mkdir()
        mirror.write_bytes(b"old mirror")
        with mock.patch.object(wex.os, "replace", side_effect=OSError("mirror locked")):
            with self.assertRaisesRegex(OSError, "mirror locked"):
                wex._atomic_mirror(primary, mirror)
        self.assertEqual(mirror.read_bytes(), b"old mirror")
        self.assert_no_temps()

    def test_input_paths_cannot_be_output(self):
        index = self.model / "model.safetensors.index.json"
        index.write_text('{"weight_map": {}}')
        for output in (self.model / "config.json", self.model / "model.safetensors", index):
            before = output.read_bytes()
            with self.assertRaisesRegex(ValueError, "model input"):
                wex.repack(self.model, output)
            self.assertEqual(output.read_bytes(), before)

    def test_invalid_expert_metadata_is_rejected_before_publication(self):
        output = self.root / "existing.wex"
        output.write_bytes(b"previous")
        variants = []
        missing = self.tensors[1:]
        variants.append(missing)
        for dtype, shape, payload in (("U8", (64, 64), bytes(4096)),
                                      ("F8_E8M0", (4096,), bytes(4096)),
                                      ("F8_E8M0", (64, 32), bytes(2048)),
                                      ("F8_E8M0", (65536, 1), bytes(65536))):
            malformed = list(self.tensors)
            malformed[0] = (malformed[0][0], dtype, shape, payload)
            variants.append(malformed)
        for tensors in variants:
            write_safetensors(self.model / "model.safetensors", tensors)
            with self.assertRaises(ValueError):
                wex.repack(self.model, output, 2)
            self.assertEqual(output.read_bytes(), b"previous")
        self.assert_no_temps()

    def test_invalid_shard_metadata_and_duplicates_rejected(self):
        path = self.model / "broken.safetensors"
        cases = [(struct.pack("<Q", 100), b"{}"),
                 (None, b'{"x":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}'),
                 (None, b'{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"x":{}}'),
                 (None, b'{"x":{"dtype":"U8","shape":[1],"data_offsets":[1,0]}}')]
        for prefix, body in cases:
            path.write_bytes((prefix or struct.pack("<Q", len(body))) + body)
            with self.assertRaises(ValueError):
                wex.index_safetensors(self.model)
        path.unlink()
        write_safetensors(path, self.tensors[:1])
        with self.assertRaisesRegex(ValueError, "duplicate tensor"):
            wex.index_safetensors(self.model)

    def test_short_read_detected_after_indexing(self):
        references = wex.index_safetensors(self.model)
        reference = next(iter(references.values()))
        with self.assertRaisesRegex(ValueError, "short tensor read"):
            wex._read_tensor(reference, {reference.path: io.BytesIO(b"")})

    def test_container_validation_rejects_truncation_and_unknown_version(self):
        output = Path(wex.repack(self.model, format_version=2)["output"])
        valid = output.read_bytes()
        for content in (valid[:39], valid[:4096], valid[:-1], valid + b"extra"):
            output.write_bytes(content)
            with self.assertRaises(ValueError):
                wex.validate_container(output)
        unknown = bytearray(valid)
        struct.pack_into("<I", unknown, 4, 99)
        output.write_bytes(unknown)
        with self.assertRaisesRegex(ValueError, "magic/version"):
            wex.validate_container(output)

    def test_container_validation_rejects_bad_layout_and_compressed_prefix(self):
        output = Path(wex.repack(self.model, format_version=2)["output"])
        valid = output.read_bytes()
        compressed_entry = wex.validate_container(output)["entries"][2]
        corruptions = []
        header_flags = bytearray(valid)
        struct.pack_into("<I", header_flags, 32, 3)
        corruptions.append(header_flags)
        entry_start = wex.WEX_ALIGN + 2 * wex.ENTRY.size
        entry_flags = bytearray(valid)
        struct.pack_into("<I", entry_flags, entry_start + 68, 258)
        corruptions.append(entry_flags)
        scale_offset = bytearray(valid)
        struct.pack_into("<I", scale_offset, entry_start + 44, 1)
        corruptions.append(scale_offset)
        prefix_magic = bytearray(valid)
        struct.pack_into("<I", prefix_magic, compressed_entry[0], 0)
        corruptions.append(prefix_magic)
        prefix_length = bytearray(valid)
        struct.pack_into("<I", prefix_length, compressed_entry[0] + 8, 4096)
        corruptions.append(prefix_length)
        tile_width = bytearray(valid)
        tile_width[compressed_entry[0] + wex.SCALE_PREFIX.size + 1] = 9
        corruptions.append(tile_width)
        for malformed in corruptions:
            output.write_bytes(malformed)
            with self.assertRaises(ValueError):
                wex.validate_container(output, verify_payload=True)

    def test_v4_geometry_sources_unchanged(self):
        v4 = {"num_hidden_layers": 43, "n_routed_experts": 256, "hidden_size": 4096, "moe_intermediate_size": 2048}
        self.assertEqual(wex._model_dims(self.model, v4), (43, 256, (4096, 2048)))
        self.assertEqual(wex._model_dims(self.model, {}), (43, 256, None))
        self.assertEqual(wex._model_dims(self.model, {"num_hidden_layers": 2, "n_routed_experts": 2}), (2, 2, None))

    def test_v41_text_config_header_and_mtp_excluded(self):
        model = self.root / "v41"
        expected = make_v41_model(model, {"model_type": "deepseek_v41", "text_config": V41_TEXT,
                                          "vision_config": {"num_hidden_layers": 32}})
        estimate = wex.repack(model, dry_run=True)
        result = wex.repack(model)
        self.assertEqual(estimate["file_bytes"], result["file_bytes"])
        self.assertEqual((result["layers"], result["experts_per_layer"], result["experts"]), (2, 3, 6))
        self.assertEqual(result["record_bytes"], 4096)
        parsed = wex.validate_container(result["output"], verify_payload=True)
        self.assertEqual((parsed["layers"], parsed["experts"]), (2, 3))
        self.assertEqual(len(parsed["entries"]), 6)
        for entry, payload in zip(parsed["entries"], expected):
            self.assertEqual(decoded_payload(result["output"], entry), payload)

    def test_v41_vendor_config_fallback_and_dims_guard(self):
        model = self.root / "v41"
        make_v41_model(model, {"model_type": "deepseek_v41"})
        # Without any geometry the V4 defaults (43 x 256) apply and REAP-192 must fail loudly.
        with self.assertRaisesRegex(ValueError, "missing native expert tensor"):
            wex.repack(model, dry_run=True)
        (model / "inference").mkdir()
        vendor = model / "inference" / "config.json"
        vendor.write_text(json.dumps({"n_layers": 2, "n_routed_experts": 3, "dim": 64, "moe_inter_dim": 32}))
        self.assertEqual(wex._model_dims(model, {}), (2, 3, (64, 32)))
        self.assertEqual(wex.repack(model, dry_run=True)["experts"], 6)
        with self.assertRaisesRegex(ValueError, "model input"):
            wex.repack(model, vendor)
        for broken in ({**V41_TEXT, "hidden_size": 128}, {**V41_TEXT, "moe_intermediate_size": 16}):
            (model / "config.json").write_text(json.dumps({"text_config": broken}))
            with self.assertRaisesRegex(ValueError, "config dims"):
                wex.repack(model, dry_run=True)
        (model / "config.json").write_text(json.dumps({"text_config": [V41_TEXT]}))
        with self.assertRaisesRegex(ValueError, "text_config"):
            wex.repack(model, dry_run=True)

    def test_cli_defaults_and_opt_in(self):
        for arguments, version in (([], 1), (["--format-version", "2"], 2)):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(wex.main(["--model-dir", str(self.model), "--dry-run"] + arguments), 0)
            self.assertEqual(json.loads(output.getvalue())["format_version"], version)


if __name__ == "__main__":
    unittest.main()
