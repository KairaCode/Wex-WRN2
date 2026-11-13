# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""WEX I/O probe: how fast do the real V4.1 containers read at a deeper queue? (V4_WEX_SUBREADS, R6)

Read-only. Opens each container with FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED (GENERIC_READ only)
and keeps QD unbuffered ReadFile calls in flight at random 4 KiB-aligned offsets, per request size, for
C: alone, U: alone and both at once. Each setting stops at --cap-gb (default 2) of reads, the whole run
at --budget-s (default 300). Prints GB/s and the per-request latency (issue -> landed), p50 / p99 / mean.

It refuses to run while KAIRA or an engine (deepseek_v4*, qwen3*, colibri*) runs: the numbers would be
wrong and the run would steal the engine's disk.

  python -I wexio_probe.py                      # the default containers below
  python -I wexio_probe.py --file C=X:\\a.wex --file U=Y:\\b.wex --qd 1,4,16 --sizes 16,4 --json out.json
  python -I wexio_probe.py --self-test          # synthetic temp file, checks every byte read
"""
import argparse, ctypes, json, os, random, subprocess, sys, tempfile, time
from ctypes import wintypes

DEFAULT_FILES = [("C", r"C:\kaira_wex\v41-reap\v4_flash_experts.nr3.wex"),
                 ("U", r"U:\kaira_wex\v41-reap\v4_flash_experts.nr3.wex")]
ENGINE_PREFIXES = ("kaira", "deepseek_v4", "qwen3", "colibri")   # image names, lower case
PAGE = 4096
MAX_WAIT = 64   # WaitForMultipleObjects

k32 = ctypes.WinDLL("kernel32", use_last_error=True) if os.name == "nt" else None


class OVERLAPPED(ctypes.Structure):
    _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t),
                ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD), ("hEvent", wintypes.HANDLE)]


def _bind():
    sig = {
        "CreateFileW": (wintypes.HANDLE, [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                          wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]),
        "ReadFile": (wintypes.BOOL, [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                                     ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(OVERLAPPED)]),
        "GetOverlappedResult": (wintypes.BOOL, [wintypes.HANDLE, ctypes.POINTER(OVERLAPPED),
                                                ctypes.POINTER(wintypes.DWORD), wintypes.BOOL]),
        "CreateEventW": (wintypes.HANDLE, [wintypes.LPVOID, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]),
        "ResetEvent": (wintypes.BOOL, [wintypes.HANDLE]),
        "WaitForMultipleObjects": (wintypes.DWORD, [wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE),
                                                    wintypes.BOOL, wintypes.DWORD]),
        "CloseHandle": (wintypes.BOOL, [wintypes.HANDLE]),
        "CancelIoEx": (wintypes.BOOL, [wintypes.HANDLE, ctypes.POINTER(OVERLAPPED)]),
        "GetFileSizeEx": (wintypes.BOOL, [wintypes.HANDLE, ctypes.POINTER(ctypes.c_longlong)]),
        "VirtualAlloc": (wintypes.LPVOID, [wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]),
        "VirtualFree": (wintypes.BOOL, [wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD]),
    }
    for name, (res, args) in sig.items():
        f = getattr(k32, name)
        f.restype, f.argtypes = res, args


if k32:
    _bind()
INVALID = ctypes.c_void_p(-1).value


def engine_running(images):
    """the KAIRA / engine images among process image names"""
    return sorted({i for i in images if i.lower().startswith(ENGINE_PREFIXES)})


def running_images():
    out = subprocess.run(["tasklist", "/fo", "csv", "/nh"], capture_output=True, text=True, check=True).stdout
    return [line.split('","')[0].strip('"') for line in out.splitlines() if line.startswith('"')]


class Target:
    def __init__(self, label, path):
        self.label, self.path = label, path
        self.h = k32.CreateFileW(path, 0x80000000, 1, None, 3, 0x20000000 | 0x40000000, None)  # read, share read
        if not self.h or self.h == INVALID:
            raise OSError(ctypes.get_last_error(), f"cannot open {path} read-only unbuffered")
        size = ctypes.c_longlong()
        if not k32.GetFileSizeEx(self.h, ctypes.byref(size)):
            raise OSError(ctypes.get_last_error(), f"size of {path}")
        self.size = size.value

    def close(self):
        k32.CloseHandle(self.h)


def run_setting(targets, qd, req, cap_bytes, deadline, rng, check=None):
    """qd reads of req bytes in flight on each target until cap_bytes landed in all (or deadline):
    {'bytes', 'seconds', 'lat_ms': [...]}; check(target, offset, data) sees every landed buffer"""
    slots = []   # (target, OVERLAPPED, event, buffer)
    assert req % PAGE == 0 and len(targets) * qd <= MAX_WAIT
    for t in targets:
        if t.size < req:
            raise ValueError(f"{t.path} is smaller than one {req}-byte request")
        for _ in range(qd):
            buf = k32.VirtualAlloc(None, req, 0x3000, 4)
            ev = k32.CreateEventW(None, True, False, None)
            if not buf or not ev:
                raise OSError(ctypes.get_last_error(), "VirtualAlloc / CreateEvent")
            slots.append([t, OVERLAPPED(), ev, buf, 0, 0])   # ..., issued at (ns), offset
    events = (wintypes.HANDLE * len(slots))(*[s[2] for s in slots])
    live = [False] * len(slots)
    issued = landed = 0
    lat = []
    pages = {t.label: (t.size - req) // PAGE for t in targets}

    def issue(i):
        nonlocal issued
        t, ov, ev, buf = slots[i][:4]
        off = rng.randint(0, pages[t.label]) * PAGE
        ctypes.memset(ctypes.byref(ov), 0, ctypes.sizeof(ov))
        ov.Offset, ov.OffsetHigh, ov.hEvent = off & 0xFFFFFFFF, off >> 32, ev
        k32.ResetEvent(ev)
        slots[i][4], slots[i][5] = time.perf_counter_ns(), off
        got = wintypes.DWORD()
        if not k32.ReadFile(t.h, buf, req, ctypes.byref(got), ctypes.byref(ov)) and ctypes.get_last_error() != 997:
            raise OSError(ctypes.get_last_error(), f"ReadFile {t.path} @ {off}")
        live[i] = True   # landed at once or pending: its event is set either way
        issued += req

    t0 = time.perf_counter()
    try:
        for i in range(len(slots)):
            if issued < cap_bytes:
                issue(i)
        while any(live):
            # only slots in flight: a finished slot's event stays set
            idx = [i for i in range(len(slots)) if live[i]]
            arr = (wintypes.HANDLE * len(idx))(*[events[i] for i in idx])
            w = k32.WaitForMultipleObjects(len(idx), arr, False, 0xFFFFFFFF)
            if w >= len(idx):
                raise OSError(ctypes.get_last_error(), "WaitForMultipleObjects")
            i = idx[w]
            t, ov, ev, buf, at, off = slots[i]
            got = wintypes.DWORD()
            ok = k32.GetOverlappedResult(t.h, ctypes.byref(ov), ctypes.byref(got), True)
            live[i] = False
            if not ok or got.value != req:
                raise OSError(ctypes.get_last_error(), f"read {t.path} @ {off}: {got.value} of {req}")
            lat.append((time.perf_counter_ns() - at) / 1e6)
            landed += req
            if check:
                check(t, off, ctypes.string_at(buf, req))
            if issued < cap_bytes and time.perf_counter() < deadline:
                issue(i)
    finally:
        for i, s in enumerate(slots):
            if live[i]:   # an error left reads in flight: never free a buffer a read still writes
                k32.CancelIoEx(s[0].h, ctypes.byref(s[1]))
                k32.GetOverlappedResult(s[0].h, ctypes.byref(s[1]), ctypes.byref(wintypes.DWORD()), True)
            k32.CloseHandle(s[2])
            k32.VirtualFree(s[3], 0, 0x8000)
    return {"bytes": landed, "seconds": time.perf_counter() - t0, "lat_ms": lat}


def summary(r):
    lat = sorted(r["lat_ms"])
    n = len(lat)
    pick = lambda q: lat[min(n - 1, int(q * n))] if n else 0.0
    return {"gb_s": r["bytes"] / r["seconds"] / 1e9 if r["seconds"] else 0.0, "reads": n,
            "p50_ms": pick(0.5), "p99_ms": pick(0.99), "mean_ms": sum(lat) / n if n else 0.0,
            "gb": r["bytes"] / 1e9, "s": r["seconds"]}


def self_test():
    assert engine_running(["KAIRA.exe", "deepseek_v41.exe", "Qwen38.exe", "colibri.exe", "code.exe"]) == \
        ["KAIRA.exe", "Qwen38.exe", "colibri.exe", "deepseek_v41.exe"]
    assert engine_running(["explorer.exe", "python.exe", "qwen.exe"]) == []
    data = random.Random(7).randbytes(24 << 20)
    fd, path = tempfile.mkstemp(suffix=".wexio")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        t = Target("T", path)
        seen = []

        def check(target, off, got):
            assert got == data[off:off + len(got)], f"bytes differ at {off}"
            seen.append(off)
        try:
            for qd, req in ((1, 1 << 20), (4, 1 << 20), (16, 2 << 20)):
                r = run_setting([t], qd, req, 64 << 20, time.perf_counter() + 30, random.Random(qd), check)
                assert r["bytes"] == 64 << 20 and len(r["lat_ms"]) == (64 << 20) // req
            assert all(o % PAGE == 0 and o + (1 << 20) <= len(data) for o in seen)
            r = run_setting([t, t], 2, 1 << 20, 8 << 20, time.perf_counter() + 30, random.Random(1), check)
            assert r["bytes"] == 8 << 20   # two targets share one cap
        finally:
            t.close()
    finally:
        os.remove(path)
    print("wexio_probe self-test: every byte read at QD 1/4/16 (1-2 MiB) matches the file; caps, "
          "offsets and the engine check hold")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", action="append", default=[], help="LABEL=PATH (repeat); default C: and U: containers")
    ap.add_argument("--qd", default="1,2,4,8,16")
    ap.add_argument("--sizes", default="16,4,2", help="request sizes in MiB")
    ap.add_argument("--cap-gb", type=float, default=2.0, help="read cap per setting (all its drives)")
    ap.add_argument("--budget-s", type=float, default=300.0, help="the whole run")
    ap.add_argument("--json", help="write every setting's numbers here")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if os.name != "nt":
        sys.exit("wexio_probe: Windows only")
    if a.self_test:
        return self_test()
    busy = engine_running(running_images())
    if busy:
        sys.exit("wexio_probe: refused, running now: " + ", ".join(busy) + " (stop KAIRA and the engines first)")
    files = [tuple(f.split("=", 1)) for f in a.file] or DEFAULT_FILES
    targets = [Target(label, path) for label, path in files]
    groups = [[t] for t in targets] + ([targets] if len(targets) > 1 else [])
    qds = [int(x) for x in a.qd.split(",")]
    sizes = [int(float(x) * (1 << 20)) // PAGE * PAGE for x in a.sizes.split(",")]
    cap = int(a.cap_gb * 1e9)
    end = time.perf_counter() + a.budget_s
    rng, rows = random.Random(20261112), []
    print(f"{'drives':<8}{'MiB':>6}{'QD':>4}{'GB/s':>8}{'p50 ms':>9}{'p99 ms':>9}{'mean ms':>9}{'reads':>7}{'GB':>7}")
    try:
        for req in sizes:
            for qd in qds:
                for g in groups:
                    if time.perf_counter() >= end:
                        print(f"budget of {a.budget_s:.0f} s used up: the remaining settings skipped")
                        return
                    if len(g) * qd > MAX_WAIT:
                        continue
                    s = summary(run_setting(g, qd, req, cap, end, rng))
                    name = "+".join(t.label for t in g)
                    rows.append({"drives": name, "mib": req / (1 << 20), "qd": qd, **s})
                    print(f"{name:<8}{req / (1 << 20):>6.1f}{qd:>4}{s['gb_s']:>8.2f}{s['p50_ms']:>9.2f}"
                          f"{s['p99_ms']:>9.2f}{s['mean_ms']:>9.2f}{s['reads']:>7}{s['gb']:>7.2f}", flush=True)
    finally:
        for t in targets:
            t.close()
        if a.json:
            with open(a.json, "w") as f:
                json.dump({"files": dict(files), "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
