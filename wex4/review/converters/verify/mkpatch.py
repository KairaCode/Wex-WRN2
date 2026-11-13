# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Builds patched/ = copies of wex4/wex4.py, wex4_convert.py, test_wex4.py with the candidate fixes (verification only)."""
import os
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'patched')


def sub(fn, pairs):
    s = open(os.path.join(P, fn), encoding='utf-8').read()
    for a, b in pairs:
        assert s.count(a) == 1, (fn, a)
        s = s.replace(a, b)
    open(os.path.join(P, fn), 'w', encoding='utf-8', newline='\n').write(s)


# ---- wex4.py: S4/S8 writer self-check + .partial cleanup
s = open(os.path.join(P, 'wex4.py'), encoding='utf-8').read()
a = s.index("    with open(tmp, 'wb') as f:\n        for kind, ident, data in blobs:")
b = s.index("    os.replace(tmp, path)\n    return root")
body = s[a:b]
body = '    try:\n' + ''.join('    ' + l if l.strip() else l for l in body.splitlines(True)) + \
    "        with open(tmp, 'rb') as f:   # never publish a file the reader refuses\n            Reader(f)\n" \
    "    except BaseException:\n        if os.path.exists(tmp):\n            os.remove(tmp)\n        raise\n"
open(os.path.join(P, 'wex4.py'), 'w', encoding='utf-8', newline='\n').write(s[:a] + body + s[b:])

sub('wex4_convert.py', [
 # S1
 (r"rb'^layers\.(\d+)\.ffn\.experts\.(\d+)\.w([123])\.(weight|scale)$'",
  r"rb'layers\.(0|[1-9]\d*)\.ffn\.experts\.(0|[1-9]\d*)\.w([123])\.(weight|scale)\Z'"),
 # S8 json
 ("    except (UnicodeDecodeError, json.JSONDecodeError) as error:\n        raise wex4.FormatError(f'safetensors: header is not UTF-8 JSON ({error})') from None\n    need(isinstance(doc, dict), 'safetensors: header is not an object')",
  "    except (ValueError, RecursionError) as error:\n        raise wex4.FormatError(f'safetensors: header is not UTF-8 JSON ({error})') from None\n    need(isinstance(doc, dict), 'safetensors: header is not an object')"),
 # S4 overflow, S8 names
 ("        elements = 1\n        for d in shape:\n            elements *= d\n",
  "        elements = 1\n        for d in shape:\n            elements *= d\n            need(elements <= wex4.U64_MAX, f'safetensors: {name} shape overflow')\n"
  "        try:\n            name = name.encode('utf-8')\n        except UnicodeEncodeError:\n            raise wex4.FormatError('safetensors: tensor name is not UTF-8') from None\n"
  "        need(len(name) <= 65535 and b'\\0' not in name, 'safetensors: tensor name too long or with NUL')\n"),
 ("        tensors.append((name.encode('utf-8'), dtype, tuple(shape), span[0], span[1]))",
  "        tensors.append((name, dtype, tuple(shape), span[0], span[1]))"),
 # S3 pack + S8 part_bytes + S11b
 ("    for path in sources:\n        raw, start, tensors = st_index(path)\n",
  "    need(sources and part_bytes > 0, 'pack: no source file or bad part size')\n"
  "    need(len({os.path.normcase(os.path.basename(p)) for p in sources}) == len(sources), 'two sources have the same file name')\n"
  "    for path in sources:\n        raw, start, tensors = st_index(path)\n"),
 # S11a
 ("                specs, reads, total = pending.get(width, ([], [], 0))\n                pending[width] = (specs + [spec], reads + [read], total + size)\n                if total + size >= GROUP:\n                    flush(width)\n",
  "                if pending.get(width, (0, 0, 0))[2] + size > GROUP:\n                    flush(width)\n"
  "                specs, reads, total = pending.get(width, ([], [], 0))\n                pending[width] = (specs + [spec], reads + [read], total + size)\n"),
 # S4/S8 expert shapes
 ("            specs, reads = [s for s, _ in order], [r for _, r in order]\n",
  "            specs, reads = [s for s, _ in order], [r for _, r in order]\n"
  "            need(all(len(s.shape) == 2 and all(0 < d <= 65535 for d in s.shape) for s in specs), f'L{layer} E{expert}: expert tensor shape')\n"),
 # S9/S3/S5/S6 unpack
 ("        doc = json.loads(rd.meta.decode('utf-8'), object_pairs_hook=_unique)\n        cache = {}\n",
  "        try:\n            doc = json.loads(rd.meta.decode('utf-8'), object_pairs_hook=_unique)\n            sources = [(s['name'], s['header'], s['sha256']) for s in doc['sources']]\n"
  "            need(all(type(x) is str for s in sources for x in s), 'metadata: source fields')\n"
  "        except (ValueError, RecursionError, KeyError, TypeError) as error:\n            raise wex4.FormatError(f'metadata: {error!r}') from None\n"
  "        names = [os.path.normcase(n) for n, _, _ in sources]\n"
  "        need(len(set(names)) == len(names) and all(n and n == os.path.basename(n) and n not in ('.', '..') and\n"
  "                                                   os.path.normcase(os.path.abspath(os.path.join(out_dir, n))) != os.path.normcase(os.path.abspath(in_path)) for n, _, _ in sources),\n"
  "             'metadata: source file name (unsafe, repeated or the container itself)')\n"
  "        cache = {}\n"),
 ("        for source in doc['sources']:\n            name, raw = os.path.basename(source['name']), source['header'].encode('utf-8')\n            header = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique)\n            header.pop('__metadata__', None)\n            spans = sorted((info['data_offsets'][0], key.encode('utf-8')) for key, info in header.items())\n",
  "        for name, text, recorded in sources:\n            try:\n                raw = text.encode('utf-8')\n            except UnicodeEncodeError:\n                raise wex4.FormatError('metadata: header text') from None\n"),
 ("            with open(tmp, 'wb') as out:\n                for piece in [struct.pack('<Q', len(raw)), raw]:\n                    out.write(piece)\n                    digest.update(piece)\n                for _, key in spans:\n",
  "            try:\n              with open(tmp, 'wb') as out:\n                for piece in [struct.pack('<Q', len(raw)), raw]:\n                    out.write(piece)\n                    digest.update(piece)\n"
  "                out.flush()\n                for key in _header_tensors(tmp, rd):\n"),
 ("            os.replace(tmp, os.path.join(out_dir, name))\n            results.append((name, digest.hexdigest() == source['sha256']))\n        need(rd.required & wex4.REQ_VALUE_EXACT or all(ok for _, ok in results), 'a rebuilt file does not match its SHA-256')\n",
  "              ok = digest.hexdigest() == recorded\n              need(ok or rd.required & wex4.REQ_VALUE_EXACT, f'{name}: the rebuilt file does not match its SHA-256')\n"
  "            except BaseException:\n                os.remove(tmp)\n                raise\n"
  "            os.replace(tmp, os.path.join(out_dir, name))\n            results.append((name, ok))\n"),
 ("def unpack(in_path, out_dir, clib=None):",
  "def _header_tensors(path, rd):\n    \"\"\"Names in data order of a file that so far holds only its header; every entry must equal the container's.\"\"\"\n"
  "    raw, start, tensors = st_index(path, data_check=False)\n"
  "    for name, dtype, shape, begin, end in tensors:\n        need(name in rd.by_name, f'tensor {name!r} missing from the container')\n"
  "        t = rd.tensors[rd.by_name[name]]\n"
  "        need((t[1], tuple(t[3]), t[7]) == (wex4.DTYPE_CODE.get(dtype, 0), shape, end - begin), f'header and container disagree on {name!r}')\n"
  "    return [t[0] for t in tensors]\n\n\ndef unpack(in_path, out_dir, clib=None):"),
 ("def st_index(path):", "def st_index(path, data_check=True):"),
 ("    need(8 + n + at == size, 'safetensors: data does not end at the end of the file')",
  "    need(not data_check or 8 + n + at == size, 'safetensors: data does not end at the end of the file')"),
 # S7
 ("        page = f.read(PAGE)\n        magic, version, layers, n_experts, _, table_offset, hflags, _ = WEX_HEADER.unpack_from(page)",
  "        page = f.read(PAGE)\n        need(len(page) == PAGE, 'file shorter than a header page')\n        magic, version, layers, n_experts, _, table_offset, hflags, _ = WEX_HEADER.unpack_from(page)"),
 ("        table_bytes = layers * n_experts * ENTRY.size\n",
  "        table_bytes = layers * n_experts * ENTRY.size\n        need(layers <= 65536 and n_experts <= 65536 and table_bytes <= old_bytes - min(table_offset, old_bytes), 'expert grid or short entry table')\n"),
 ("        need(len(table) == table_bytes and old_bytes % PAGE == 0, 'short entry table')\n",
  "        need(len(table) == table_bytes, 'short entry table')\n"),
 ("        need(at == old_bytes, f'records end",
  "        need(at <= old_bytes, 'a record runs past the end of the file')\n        need(at == old_bytes, f'records end"),
 # S2
 ("        f.seek(legacy_page_offset)   # 1. append",
  "        wex4.Reader(_Probe(file_bytes, {0: new, index_offset: index + bytes(align(len(index)) - len(index))}))   # refuse before writing\n"
  "        f.seek(legacy_page_offset)   # 1. append"),
 ("def upgrade(path, hash_records=True):",
  "class _Probe:\n    \"\"\"What wex4.Reader reads to validate: the file size, the header page and the index.\"\"\"\n\n"
  "    def __init__(self, size, parts):\n        self.size, self.parts, self.at = size, parts, 0\n\n"
  "    def seek(self, offset, whence=0):\n        self.at = self.size if whence else offset\n\n"
  "    def tell(self):\n        return self.at\n\n    def read(self, n):\n        return self.parts[self.at][:n]\n\n\n"
  "def upgrade(path, hash_records=True):"),
 # S10
 ("        need(sha256(page) == legacy[0][4], 'saved header page is damaged')\n",
  "        need(sha256(page) == legacy[0][4], 'saved header page is damaged')\n"
  "        need(legacy[0][2] + PAGE == rd.index_offset and rd.index_offset + align(rd.index_bytes) == rd.file_bytes and\n"
  "             page[:4] in (b'1XEW', b'3XEW'), 'not the layout that upgrade writes')\n"),
 # S4 main
 ("        mode = option('--experts')\n", "        mode = option('--experts')\n        need(mode in (None, 'raw', 'nr3'), '--experts takes raw or nr3')\n"),
])
sub('test_wex4.py', [("FIXTURE = os.path.join(HERE, '..', 'fixtures'", "FIXTURE = os.path.join(HERE, '..', '..', '..', '..', '..', 'fixtures'")])
