# SSD/NVMe temperatures without admin: IOCTL_STORAGE_QUERY_PROPERTY, StorageDeviceTemperatureProperty (52).
import ctypes, struct, sys
from ctypes import wintypes as w
k32 = ctypes.WinDLL('kernel32', use_last_error=True)
k32.CreateFileW.restype = w.HANDLE
k32.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, w.LPVOID, w.DWORD, w.DWORD, w.HANDLE]
INVALID = w.HANDLE(-1).value

def temps(n):
    h = k32.CreateFileW(r'\\.\PhysicalDrive%d' % n, 0, 3, None, 3, 0, None)
    if not h or h == INVALID:
        return None
    try:
        q = struct.pack('<II4x', 52, 0)
        out = ctypes.create_string_buffer(512)
        ret = w.DWORD()
        if not k32.DeviceIoControl(w.HANDLE(h), 0x2D1400, q, len(q), out, 512, ctypes.byref(ret), None):
            return None
        b = out.raw
        crit, warn, cnt = struct.unpack_from('<hhH', b, 8)
        vals = [struct.unpack_from('<h', b, 24 + 16 * i + 2)[0] for i in range(cnt)]
        return vals, warn, crit
    finally:
        k32.CloseHandle(w.HANDLE(h))

names = dict(a.split('=', 1) for a in sys.argv[1:])
for n in range(12):
    t = temps(n)
    if t:
        print('disk %d %-4s temp %s C  warning %d C  critical %d C' % (n, names.get(str(n), ''), t[0], t[1], t[2]))
