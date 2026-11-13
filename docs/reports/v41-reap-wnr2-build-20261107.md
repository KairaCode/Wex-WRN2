# W1: V4.1 REAP-192 WNR2 container (2026-11-07)

Status: **done and verified.** The container was built at chunk 2048:512 with a WNM2 model retrained on the V4.1 experts. Every record decodes exactly. 64 random records decode on the GPU byte-identical to the WEX1 container on D:.

## Result

> Update 2026-11-09: WEX1 only on U: (D: copy deleted 05:02); WNR2 original on E:, sha-verified copies on U: (11-08 17:58) and C: (11-09 05:20), sha256 3f703fa4.... The E: free-space figure below is from 11-07.
>
> Update 2026-11-10: WNR3 replaced WNR2 in production; WEX1 and every WNR2 copy (U:, C:, E:) were deleted ([WNR3-PRODUCTION-20261110.md](WNR3-PRODUCTION-20261110.md)). This file is history.

| | WEX1 (source) | WNR2 (W1) |
|---|---|---|
| file | `D:\kaira_wex\v41-reap\v4_flash_experts.wex` (byte copy on `U:\kaira_wex\v41-reap\`) | `E:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex` |
| bytes | 144,389,472,256 | **129,905,360,896** (120.98 GiB) |
| per expert record | 18,800,640 | 16,914,680 avg (0.8997 of raw) |
| sha256 | 4c055238...7fa549 (computed outside this build, `.verified`) | `3f703fa4d993ed46d2873676c71be07cf11c11c7baef3e0bce6428c30921eb9d` |

- Saved: 14,484,111,360 B. That is **-10.03 %** of the file and of the records.
  - This is less than the 12.8 % the plan expected [M].
  - The V4.1 FP4 weights carry more entropy: 3.781 bits per weight against 3.670 on V4.
  - The finer chunks add about 0.3 points [I]: 19,440 streams per record, 38,880 B of lengths per record.
- Capacity in a fixed RAM or VRAM tier: 1/0.8997 = **+11.1 % records**, not the +14 % that section 3.2 assumes. W3 should use this number.
- Header: WEX3 v3, 40 x 192, flags 21 (SCALE_TILES|DEDUP|NEURAL_RANS2), table at 65536, records from 618,496, uuid 01f11564c51f44599b9885b727ad2937.
- Model: WNM2 id `0xab4261bd`, 59,408 B, sha256 f83fee82734219358d0f1290245298d44e2808d7abb48c9c0a742cd6dd855e3e, trained on all 7680 experts. It is embedded in the container; a copy is at `w1\model-full-nr2.wnm2`.
- 7680 of 7680 records are WNR2. There are 0 raw fallbacks.
- Retraining was worth it [M]. In the pre-check, 8 V4.1 experts encoded with the old V4 model came out at 0.941-0.950 of raw, against 0.8997 with the V4.1 model.

## Time

- The build ran from 02:34:57 to 02:47:01, **723 s** in total, under `heavy.lock`:
  - training: 62 s
  - GPU encode with the CPU reference verify (12 workers): 373 s
  - readback and whole-file hash: 288 s
- Free space on E: was 135.2 GiB at the start. **It is 15.3 GB now.** Anything else that writes to E: must check first.

## Verification

1. **The build script's own checks** (`w1\build-full-nr2.json`, `w1\build-v41-nr2.log`), exit 0:
   - All 7680 records were decoded with the CPU reference (`wex_nr2_verify`) against the raw slab: all exact.
   - 179 GPU records are byte-equal to the CPU encoder (every 43rd).
   - The container check passed: `wex_validate_header`, `wex_validate_table`, `wex_nr2_model_location` and `wex_nr2_model_open`, and the model id matches the descriptor.
   - 180 records were read back from disk through the slab path.
   - All 7680 on-disk records were rehashed: equal to the verified bytes.
2. **64 random records on the engine's GPU decoder** (`w1\verify64.log`), ALL OK:
   - `w1\sample_records.py` (seed 20261107) takes each NR2 record from E: and pairs it with the slab of the same expert from **D:**, the original WEX1. It also asserts that the entry geometry is identical in both tables and that the model id matches.
   - `test_decode_batch.exe` (lab\wex-prod-20261105 source) ran with a `kaira_wex_cuda.dll` built today from the clean src at 79aff2b:
     - `wex_cuda_bridge.cu` sha256 a346652f...
     - `wex_cuda.cuh` b6a6563f...
     - `wex_neural_rans.h` d875e1d1...
   - Every record was decoded through `wex_cuda_decode_batch_to_device` at batch sizes 1, 2, 3, 4, 6, 8, 10 and 16. That is 512 decodes, and every destination byte equals the WEX1 slab.
   - The corruption case behaved as expected: a damaged payload and an invalid entry return -1 and leave their destination untouched.
   - Timings: single call 0.759 ms per record; batch of 8, 0.402 ms; batch of 16, 0.378 ms.
     - These timings are **not** a benchmark. The GPU was shared with outside work (13-98 % busy, 10-24 GB in use before the lock).
     - For reference, V4 records measured on 11-05 (single / batch 8): 0.734 / 0.339 ms at 2048:512 and 0.848 / 0.399 ms at 4096:2048.

## Command and deviations

```
python -I lab\wex-gpu-direct-20261019\nr2\build_nr2_full.py --source U:\kaira_wex\v41-reap\v4_flash_experts.wex
  --out E:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex --weight-chunk 2048 --scale-chunk 512
  --verify-workers 12 --min-free-gib 5 --report-dir lab\v41-gpu-20261107\w1
```

- **Source was U:, not D:.**
  - The verified byte copy, made outside this build, appeared at 02:34, when the lock passed to this build. Its sha256 matches and the `.verified` marker exists.
  - U: reads at 6.6 GB/s against 1.6 GB/s on D:.
  - The 64-record check reads D:, so both copies are covered.
- **Lab tool edit:** `build_nr2_full.py` gained `--report-dir`.
  - Without it, the report, model copy and counts would have overwritten the V4 `build-full-nr2.json`, `model-full-nr2.wnm2` and `counts-full-nr2.npz` in `lab\wex-gpu-direct-20261019\nr2\`.
  - The default is unchanged. The build above is its test.
- `--min-free-gib 5` instead of 2, to leave room on E:, which also holds engram shard 48. The script's up-front check (0.9 x source + margin) passed: 135.2 GiB free.
- **No source edits under src/, no commits.** The lock was taken at 02:34:33 and released at 02:47:55.

## Files

- `w1\build-v41-nr2.log`, `w1\build-full-nr2.json` (per-record rows), `w1\model-full-nr2.wnm2`, `w1\counts-full-nr2.npz`: from the build.
- `w1\verify64.log`, `w1\sample_records.py`: the 64-record GPU check.
- `w1\precheck_records.py`: the pre-check, 8 experts encoded with the V4 model, then decoded by the new helper.
  - The batch-correctness part passed.
  - Its single-call timing loop needs at least 16 records and failed on 8 by design of the test, not the data.
- `w1\build_helper.cmd`, `w1\build\kaira_wex_cuda.dll`, `w1\build\test_decode_batch.exe`.
- `w1\take_lock.ps1`: the lock poller.

## For W2 / W3

- **Same-drive A/B (PLAN update, W3).** Copying this container to `U:\kaira_wex\v41-reap\` needs 130 GB. U: had 279 GB free at 02:34. Verify the copy against the sha256 above.
- Expected byte effect on the WNR2 arm: about -10 % bytes per NVMe miss and +11.1 % records per RAM tier [M for sizes, I for the speed effect]. This is smaller than the plan assumed, so the +3 % decision rule in 3.2 is harder to reach.
