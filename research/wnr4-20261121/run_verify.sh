#!/usr/bin/env bash
# Short GPU verification after the review fixes: the bench (every record bit-exact, single-bit flips, speed)
# and the helper test (async/sync, corrupted record, unknown model, wrong owner, slot ring) on the three packs.
# LOCKSH (optional) = a shell file defining take_lock LABEL / release_lock for a machine-wide GPU lock.
set -u
cd "$(dirname "$0")"
if [ -n "${LOCKSH:-}" ]; then source "$LOCKSH"; else take_lock() { :; }; release_lock() { :; }; fi
take_lock "wnr4 research: GPU verification after review fixes (~4 min)" || exit 3
for p in q38mx q38nv v41; do
  ./out/bench_gpu9.exe out/packs3/$p.pack 5 > results/bench9-$p-G128.log 2>&1
  ./out/test_helper9.exe out/packs3/$p.pack > results/helper9-$p-G128.log 2>&1; echo "helper $p exit $?" >> results/helper9-$p-G128.log
done
release_lock
echo verify done
