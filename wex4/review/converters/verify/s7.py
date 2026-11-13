# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
import os, sys, struct
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[1]))
import wex4, wex4_convert as conv
src = open(os.path.join('out', 'work', 'tiny_cz.wex'), 'rb').read()
page = bytearray(src[:4096]); struct.pack_into('<II', page, 8, 60000, 60000)
p = os.path.join('out', 's7_grid.wex'); open(p, 'wb').write(bytes(page) + src[4096:])
try:
    print(conv.upgrade(p))
except wex4.FormatError as e:
    print('clean refusal:', e)
except BaseException as e:
    print('UNCAUGHT', type(e).__name__, str(e)[:80])
print('file unchanged:', open(p, 'rb').read() == bytes(page) + src[4096:])
