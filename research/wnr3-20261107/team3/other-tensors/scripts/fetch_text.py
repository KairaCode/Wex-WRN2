"""Download small text files from the pinned revision into OUT_DIR (read as data only).
Usage: python -I fetch_text.py OUT_DIR file1 [file2 ...]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
for f in sys.argv[2:]:
    d = hf.get_file(f)
    p = os.path.join(out, f.replace('/', '__') + '.txt')
    open(p, 'wb').write(d)
    print(f, len(d), '->', p)
