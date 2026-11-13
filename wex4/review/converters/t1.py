# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Review probes for wex4_convert.py: python -I t1.py   (writes only under ./out)"""
import os, sys, json, struct, io, hashlib, shutil, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
NEXT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, NEXT)
import wex4, wex4_convert as conv
OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)
C = wex4.CLib(os.path.join(NEXT, 'out', 'wex4_tools_ref.dll'))


def run(label, fn, *a, **k):
    try:
        r = fn(*a, **k)
        print(f'[{label}] ACCEPTED -> {str(r)[:100]}')
        return r
    except wex4.FormatError as e:
        print(f'[{label}] clean refusal: {str(e)[:110]}')
    except BaseException as e:
        print(f'[{label}] UNCAUGHT {type(e).__name__}: {str(e)[:110]}')


def st_raw(path, raw, payload=b''):
    open(path, 'wb').write(struct.pack('<Q', len(raw)) + raw + payload)
    return path


def st(path, tensors, metadata=None):
    header, payload = {}, bytearray()
    if metadata is not None:
        header['__metadata__'] = metadata
    for name, dtype, shape, data in tensors:
        header[name] = {'dtype': dtype, 'shape': list(shape), 'data_offsets': [len(payload), len(payload) + len(data)]}
        payload += data
    return st_raw(path, json.dumps(header, separators=(',', ':')).encode(), bytes(payload))


def verdicts(path):
    data = open(path, 'rb').read()
    try:
        py = 'ok ' + wex4.check(data, True, C).hex()[:8]
    except wex4.FormatError as e:
        py = 'REFUSED: ' + str(e)
    return py, 'C status %d' % C.check(data, 3)[0]


def pack_then_read(label, src, **k):
    out = os.path.join(OUT, label + '.wex4')
    for x in (out, out + '.partial'):
        if os.path.exists(x):
            os.remove(x)
    run(label + ' pack', conv.pack, out, src if isinstance(src, list) else [src], **k)
    if os.path.exists(out):
        print(f'    output file exists; readers: {verdicts(out)}')
        return out
    print('    no output file;', 'leftover .partial' if os.path.exists(out + '.partial') else 'no .partial')


def p(n):
    return os.path.join(OUT, n)


def exp(layer, e, dt_s='F8_E8M0', dt_w='I8', rows=4, sc=2, extra=()):
    t = []
    for m in (1, 2, 3):
        t.append((f'layers.{layer}.ffn.experts.{e}.w{m}.weight', dt_w, (rows, sc * 16),
                  bytes([m]) * (rows * sc * 16 * wex4.BITS[wex4.DTYPE_CODE[dt_w]] // 8)))
        t.append((f'layers.{layer}.ffn.experts.{e}.w{m}.scale', dt_s, (rows, sc),
                  bytes([16 + m]) * (rows * sc * wex4.BITS[wex4.DTYPE_CODE[dt_s]] // 8)))
    return t + list(extra)


if __name__ == '__main__':
    print('--- A. st_index hostile headers')
    run('deep nesting', conv.st_index, st_raw(p('a1.st'), b'{"a":' + b'[' * 100000 + b']' * 100000 + b'}'))
    run('5000-digit int', conv.st_index, st_raw(p('a2.st'), b'{"a":{"dtype":"U8","shape":[' + b'9' * 5000 + b'],"data_offsets":[0,0]}}'))
    run('lone surrogate name', conv.st_index, st_raw(p('a3.st'), b'{"\\ud800":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}', b'x'))
    run('unknown dtype', conv.st_index, st_raw(p('a4.st'), b'{"a":{"dtype":"NOPE","shape":[7,7],"data_offsets":[0,3]}}', b'xyz'))
    run('leading whitespace', conv.st_index, st_raw(p('a5.st'), b'  \n{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}', b'x'))
    run('dup inner key', conv.st_index, st_raw(p('a6.st'), b'{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1],"dtype":"U8"}}', b'x'))
    run('rank 9', conv.st_index, st(p('a7.st'), [('r9', 'U8', (1,) * 9, b'x')]))
    run('metadata not str', conv.st_index, st_raw(p('a8.st'), b'{"__metadata__":{"a":1}}'))
    run('empty {}', conv.st_index, st_raw(p('a9.st'), b'{}'))
    run('float shape', conv.st_index, st_raw(p('a10.st'), b'{"a":{"dtype":"U8","shape":[1.0],"data_offsets":[0,1]}}', b'x'))
    run('bool shape', conv.st_index, st_raw(p('a11.st'), b'{"a":{"dtype":"U8","shape":[true],"data_offsets":[0,1]}}', b'x'))
    run('F4 odd', conv.st_index, st_raw(p('a12.st'), b'{"a":{"dtype":"F4","shape":[3],"data_offsets":[0,2]}}', b'xy'))
    run('trailing junk after json', conv.st_index, st_raw(p('a13.st'), b'{} x'))
    run('NaN top', conv.st_index, st_raw(p('a14.st'), b'{"a":NaN}'))
    run('overflow shape x0', conv.st_index, st_raw(p('a15.st'), b'{"a":{"dtype":"F32","shape":[9223372036854775808,4,0],"data_offsets":[0,0]}}'))

    print('--- B. pack accepts, reader refuses')
    pack_then_read('b1_overflow_shape', p('a15.st'))
    pack_then_read('b2_nul_name', st_raw(p('b2.st'), b'{"a\\u0000b":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}', b'x'))
    pack_then_read('b3_long_name', st(p('b3.st'), [('n' * 70000, 'U8', (1,), b'x')]))
    pack_then_read('b4_empty', st_raw(p('b4.st'), b'{}'))
    pack_then_read('b5_emptyname', st(p('b5.st'), [('', 'U8', (1,), b'x')]))
    pack_then_read('b6_partmib0', st(p('b6.st'), [('big', 'U8', (70000,), bytes(70000))]), part_bytes=0)
    pack_then_read('b7_unknown_dtype_bigshape', st_raw(
        p('b7.st'), b'{"a":{"dtype":"NOPE","shape":[18446744073709551615,18446744073709551615],"data_offsets":[0,3]}}', b'xyz'))

    print('--- C. expert grouping')
    pack_then_read('c0_good', st(p('c0.st'), exp(0, 0)), experts='raw')
    pack_then_read('c1_f32_experts', st(p('c1.st'), exp(0, 0, 'F32', 'F32')), experts='raw')
    o = pack_then_read('c2_alias', st(p('c2.st'), exp(0, 0, extra=[('layers.00.ffn.experts.0.w1.weight', 'I8', (4, 32), b'\x7f' * 128)])),
                       experts='raw')
    if o:
        with open(o, 'rb') as f:
            rd = wex4.Reader(f)
            print('    tensors in container:', len(rd.tensors), '(source had 7);',
                  'alias present' if b'layers.00.ffn.experts.0.w1.weight' in rd.by_name else 'ALIAS TENSOR DROPPED', ';',
                  'canonical w1.weight present' if b'layers.0.ffn.experts.0.w1.weight' in rd.by_name else 'CANONICAL w1.weight DROPPED')
        run('c2 unpack', conv.unpack, o, p('c2_back'))
    t = exp(0, 0); t[0] = (t[0][0], 'I8', (128,), t[0][3])
    pack_then_read('c3_rank1', st(p('c3.st'), t), experts='raw')
    t = exp(0, 0); t[0] = (t[0][0], 'I8', (1, 70000), bytes(70000))
    pack_then_read('c4_dim70000', st(p('c4.st'), t), experts='raw')
    t = exp(0, 0); t[1] = (t[1][0], 'F8_E8M0', (0, 2), b'')
    pack_then_read('c5_zero_scale', st(p('c5.st'), t), experts='raw')
    t = exp(0, 0); t[0] = (t[0][0], 'I8', (4, 16), bytes(64))
    pack_then_read('c6_cols_mismatch', st(p('c6.st'), t), experts='raw')
    pack_then_read('c7_missing', st(p('c7.st'), exp(0, 0)[:5]), experts='raw')
    pack_then_read('c8_layer_big', st(p('c8.st'), [(n.replace('layers.0.', 'layers.70000.'), d, s, b) for n, d, s, b in exp(0, 0)]),
                   experts='raw')
    pack_then_read('c9_experts_foo', st(p('c9.st'), exp(0, 0, 'F32', 'F32')), experts='foo')

    print('--- D. same basename in two folders')
    os.makedirs(p('d1'), exist_ok=True); os.makedirs(p('d2'), exist_ok=True)
    s1 = st(os.path.join(p('d1'), 'model.safetensors'), [('x', 'U8', (4,), b'AAAA')])
    s2 = st(os.path.join(p('d2'), 'model.safetensors'), [('y', 'U8', (4,), b'BBBB')])
    o = pack_then_read('d_same_basename', [s1, s2])
    shutil.rmtree(p('d_back'), ignore_errors=True)
    run('d unpack', conv.unpack, o, p('d_back'))
    print('    files rebuilt:', os.listdir(p('d_back')), '(2 sources packed)')
