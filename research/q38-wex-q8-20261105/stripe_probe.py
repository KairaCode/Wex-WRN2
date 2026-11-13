"""Cold expert reads from one NVMe against two (identical shard copies), the
engine's buffered path (gate+up range, down range), THREADS workers.

usage: python -I stripe_probe.py PRIMARY_DIR MIRROR_DIR [experts_per_side=400] [threads=8] [OUT.json]
Side 'single' reads every expert from PRIMARY; side 'striped' sends even
experts to PRIMARY and odd ones to MIRROR. Disjoint random sets, read-only."""
import json
import random
import sys
from pathlib import Path

PRIMARY, MIRROR = Path(sys.argv[1]), Path(sys.argv[2])
PER_SIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 400
THREADS = int(sys.argv[4]) if len(sys.argv) > 4 else 8
OUT = Path(sys.argv[5]) if len(sys.argv) > 5 else None

sys.argv = [sys.argv[0], str(PRIMARY), '1', str(THREADS)]
sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I drops the script dir
import probe_q38_experts as probe   # noqa: E402  (reads its own argv at import)


def main():
    cfg = json.loads((PRIMARY / 'config.json').read_text())
    text = cfg.get('text_config', cfg)
    tensors = probe.shard_tensors()
    rng = random.Random(20261106)
    picks = rng.sample([(l, e) for l in range(text['num_hidden_layers']) for e in range(text['num_experts'])], 2 * PER_SIDE)
    report = {'primary': str(PRIMARY), 'mirror': str(MIRROR), 'threads': THREADS, 'experts_per_side': PER_SIDE}
    for side, keys in (('single', picks[:PER_SIDE]), ('striped', picks[PER_SIDE:])):
        plan = []
        for i, key in enumerate(keys):
            ranges, _ = probe.expert_ranges(tensors, *key)
            root = MIRROR if side == 'striped' and i % 2 else PRIMARY
            plan.append((key, [(str(root / shard), off, n) for shard, off, n in ranges]))
        probe.MODEL = Path('')          # ranges now carry absolute shard paths
        seconds, moved, _ = probe.run_side(plan, False)
        report[side] = {'seconds': round(seconds, 3), 'payload_gib_s': round(moved / seconds / 2**30, 3),
                        'ms_per_expert_wall': round(1000 * seconds / len(plan), 3)}
    report['speedup'] = round(report['striped']['payload_gib_s'] / report['single']['payload_gib_s'], 3)
    text_out = json.dumps(report, indent=1)
    print(text_out)
    if OUT:
        OUT.write_text(text_out, encoding='utf-8')


if __name__ == '__main__':
    main()
