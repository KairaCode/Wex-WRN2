# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
import sys, os, safetensors
for n in ('a5.st','a15.st','a7.st','b2.st','a4.st','a3.st','b7.st','b5.st','a9.st','a13.st'):
    try:
        with safetensors.safe_open(os.path.join('out', n), 'np') as f:
            print(n, 'safetensors ACCEPTS', list(f.keys())[:2])
    except BaseException as e:
        print(n, 'safetensors refuses:', type(e).__name__, str(e)[:90])
