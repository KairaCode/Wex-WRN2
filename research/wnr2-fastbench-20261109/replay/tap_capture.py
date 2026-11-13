"""Capture the engine's routing tap while a command runs.

  python -I tap_capture.py --port 8779 --out RUN.v4rt -- CMD [ARGS...]

Binds 127.0.0.1:PORT (UDP), sets V4_ROUTE_UDP=127.0.0.1:PORT, V4_ROUTE_TEXT=0,
V4_ROUTE_PREFILL=0 and V4_ROUTE_RESID_EVERY=1 for CMD (which inherits the rest of the
environment), and writes every datagram to OUT as u32 length + bytes until CMD exits.
Exit code: CMD's. The tap is fire-and-forget: a full socket buffer drops datagrams, so the
receive buffer is large and the END datagram's 'dropped' counter is reported by make_trace.py.
"""
import os
import socket
import struct
import subprocess
import sys
import threading


def main(argv):
    if '--' not in argv:
        raise SystemExit(__doc__)
    cut = argv.index('--')
    opts, cmd = argv[:cut], argv[cut + 1:]
    port = int(opts[opts.index('--port') + 1]) if '--port' in opts else 8779
    out = opts[opts.index('--out') + 1]
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32 << 20)
    sock.bind(('127.0.0.1', port))   # fails loudly if another capture holds the port
    sock.settimeout(0.5)
    counts, stop = {}, threading.Event()
    f = open(out, 'wb')

    def pump():
        while not stop.is_set():
            try:
                d = sock.recv(65535)
            except socket.timeout:
                continue
            f.write(struct.pack('<I', len(d)) + d)
            kind = d[5] if len(d) > 5 else -1
            counts[kind] = counts.get(kind, 0) + 1

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    env = dict(os.environ, V4_ROUTE_UDP=f'127.0.0.1:{port}', V4_ROUTE_TEXT='0',
               V4_ROUTE_PREFILL='0', V4_ROUTE_RESID_EVERY='1')
    rc = subprocess.call(cmd, env=env)
    stop.wait(1.0)   # the last END/RESID datagrams
    stop.set()
    th.join()
    f.close()
    names = {1: 'hello', 2: 'turn', 3: 'frame', 4: 'prefill', 5: 'end', 7: 'resid'}
    print('tap_capture:', out, {names.get(k, k): v for k, v in sorted(counts.items())}, 'rc', rc, flush=True)
    return rc


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
