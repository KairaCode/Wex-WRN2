# WEX4 gate M3 on the production container (2026-11-11)

Engine: KAIRA branch `wex4-20261111`, commit 75932c1, release build from a fresh `git archive`
(`make -f Makefile.deepseek-v4 ARCH=x86-64-v3 deepseek-v4`), run outside `app\` with the installed
`coli_cuda_dsv4_dg.dll` and `kaira_wex_cuda.dll`. Harness: `<KAIRA>\lab\wnr2-fastbench-20261109\short-abba\short_abba.sh`,
RAM budget 16 GB, CTX 32768, container on U: only (no mirror), 7 prompts, 64 tokens each, 2 runs per form.

| step | result |
|---|---|
| baseline, WEX3 form (`A-wex3.out`) | 2 valid runs, mean 2.070 tok/s, wait 292.1 ms/token |
| `wex4_convert.py upgrade` in place | 93.6 s; +6,717,440 B (index 6,712,992 B, 7,680 records); root `f0f78a84d040a3b336f3abef3cf8b0a36c649deb38586c813e94d35dea3beeb1` |
| `wex4.py verify --dll` | 8 min 7 s: every blob and record SHA-256 right, all 7,680 records decoded on the CPU (fast decoder) |
| WEX4 form, root pinned with `COLI_WEX_ROOT` (`B-wex4.out`) | engine log "WEX4 container (hashed header and index)"; 2 valid runs, mean 2.077 tok/s, wait 293.6 ms/token |
| difference WEX4 vs WEX3 | +0.32 % tok/s (per prompt -0.45 .. +0.96 %): inside the run-to-run noise; READY time 17.9-22.5 s in both forms |
| tokens and logits | identical in all 4 runs (logits hash a14761eb37afa76c) |
| `wex4_convert.py downgrade` | file back to 123,306,192,896 B; header page equal to the saved one; whole-file SHA-256 490e8208...e71e71, the value recorded when the container was built |

Limits: the two forms were run one after the other (2 + 2), not interleaved, and without a closing WEX3 run;
one drive only; the summary lines of the harness say "no container" because its parser does not know this
engine's log line (the stderr logs show the container active and 60,793 / 60,833 GPU-direct reads).
The production container is WEX3 again; nothing was installed.

## Gate M4 and installation (same day)

| step | result |
|---|---|
| upgrade of U: and C: | 101 s and 98 s; both 123,312,910,336 B; header pages equal, root `f0f78a84...3beeb1` |
| `verify` on each copy | all hashes right, 7,680 records CPU-decoded, same content digest `fb085e12...8ae512` |
| engine 75932c1 outside `app\`, U: + C: mirror, weights 38,62, root pinned | "Matching mirror container active", "read split over 2 drives: 38% 62%"; OS reads U: 39 %, C: 61 %; 3.051 and 3.059 tok/s over the 7 prompts (earlier WEX3 + mirror measurements: about 3.04); logits file SHA-256 equal in both runs and equal to the single-drive WEX3 baseline (a14761eb...) |
| installation | `app\colibri-optimized\deepseek_v4.exe` replaced by the 75932c1 build (sha256 1657b634...); the old one (b7c7b6a2...) is in `<KAIRA>\lab\app-before-wex4-20261111` with restore notes. `kaira.json` unchanged (no pinned root) |

Limits: the harness marked the second mirror run INVALID because its sampler took our own engine process
for a foreign one; both runs completed and the figures come from their logs. No same-session WEX3 + mirror
baseline. The installed engine was not started from `app\` by this work. KAIRA branch `wex4-20261111` is not
merged into master.

## First-read SHA-256 (spec 8.2), 2026-11-11

KAIRA commit 4ba5568 on `wex4-20261111` (patch `integration/kaira-wex4-first-read-4ba5568.patch`): the store
keeps the 32-byte SHA-256 of every record from the index and checks a record the first time the process reads
it; a mismatch refuses the record and the engine falls back to safetensors, as for a level-1 failure. Switch:
`COLI_WEX_VERIFY=0|1`, default on when `COLI_WEX_ROOT` pins the root, off otherwise. `next/wex4.h` gained a
SHA-NI path (1.83 GB/s against 0.33 GB/s scalar, checked against hashlib). `tests/test_v4_wex_nr3` covers a
clean container, a flipped record byte with the check on (refused, fallback) and off (served as before).

Measurement: release build outside `app\`, U: + C: mirror, weights 38,62, root pinned; A = `COLI_WEX_VERIFY=0`,
B = default (on). Planned A1 B1 B2 A2; the harness stopped after B1 because another engine started at the end
of B1, so one pair only.

| | A1 (off) | B1 (on) |
|---|---|---|
| ready after | 20.8 s | 26.4 s (log: "first-read SHA-256 of every record: on") |
| first prompt (16 tokens) | 9.4 s, ttft 4.3 s | 14.3 s, ttft 7.0 s |
| six 64-token prompts, mean tok/s | 2.768 | 2.821 |
| total generation time, 8 prompts | 212.8 s | 213.0 s |
| logits file SHA-256 | a14761eb... | a14761eb... (same) |
| failures / fallbacks | 0 / 0 | 0 / 0 |

Reading: the check costs CPU only while hot records are read for the first time (start-up and the first prompt);
after that the per-prompt difference is inside the noise (single prompts vary by +/-0.3 tok/s). One pair is not
a gate: a full ABBA set is needed before installing. The installed engine is still the 75932c1 build and no
root is pinned in `kaira.json`.

Installation (user's decision, 2026-11-11): `app\colibri-optimized\deepseek_v4.exe` replaced by the 4ba5568
build (sha256 d9a4ac29...); the `v41gpu` profile of `app\kaira.json` pins `COLI_WEX_ROOT=f0f78a84...3beeb1`,
which turns the first-read check on. The 75932c1 exe and the previous `kaira.json` are in
`<KAIRA>\lab\app-before-wex4-verify-20261111` with restore notes. The full ABBA set and a start from KAIRA
itself are not run yet.
