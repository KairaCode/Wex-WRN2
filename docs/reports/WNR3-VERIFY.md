# WNR3 su record V4.1 reali: verifica (2026-11-10)

> **Dated report (2026-11-10).** "WNR2 (production)" and "deployed" describe the state before the switch of that day; WNR3 is the production record codec since then: superseded for what runs today by [STATUS.md](../../STATUS.md).

**Status (English).** WNR3 ([WNR3-SPEC.md](../WNR3-SPEC.md)) on 103 real DeepSeek V4.1 REAP-192 records, code of engine branch wnr3-20261110 (the WNR3 review-fix commit for the encodes and the helper; the WNR3 branch head adds only the lab switch of section 4).

| check | WNR3-CZ (shipping profile) | WNR3 without CZ | WNR2 (production) |
|---|---|---|---|
| size, 40 held-out experts (of raw 18,800,640 B) | **0.8545** | 0.8818 | 0.8994 |
| size, top-64 WNR2 tail | 0.8724 | 0.9003 | 0.9329 |
| value-exact decode, CPU and GPU | **103/103** (CZ: -0 -> +0 only) | 103/103 byte-exact | - |
| wrong single-bit flips accepted (GPU exhaustive regions / CPU random over the record) | **0 / 2,231,336** and **0 / 3,000** | 0 | 0.13-0.43% and **1.8-1.9%** |
| GPU decode p50, batch-16 / single (RTX 5090) | 0.366-0.373 / 0.706-0.713 ms | - | 0.376-0.377 / 0.718-0.723 ms (parity; WNR3 -1.5%) |

- Full container estimate: **~123.4 GB = 114.9 GiB** (WNR2 129.9 GB, WEX1 144.4 GB), extrapolated from 103 records with models trained on 80 experts.
- **G7 (end-to-end per-step logits proof of canonical zero): PASSED 2026-11-10** (section "G7-lab result" below). **Full WNR3 container: BUILT 2026-11-10** (123.3 GB = 114.8 GiB, 0.949 of WNR2, 7,680/7,680 records read back and decoded exactly; [`research/wnr3-20261110/full-build/RESULT-20261110.md`](../../research/wnr3-20261110/full-build/RESULT-20261110.md)). **Update 2026-11-10: in production.** The maintainer's application runs this container on U: with a C: mirror; WEX1 and every WNR2 copy were deleted ([WNR3-PRODUCTION-20261110.md](WNR3-PRODUCTION-20261110.md)). "WNR2 (production)" below means the WNR2 container in use when these checks ran.

Paths: `verify/`, `verify/audit/`, `g7lab/`, `mutants/` are in this repo under [`research/wnr3-20261110/`](../../research/wnr3-20261110/) (scripts, logs and the small JSON results; the `.bin` samples, `.wnm2` lab models, executables and `g7lab/bin` stay in the maintainer's lab, `<TREE>\lab\wnr3-impl-20261110`). The `tools/` builder and DLL named below are `c/tools/build_nr3.py` and `c/tools/wex_nr3_tools.c` here; `fresh-fix3` / `fresh-verify` are fresh lab copies of the engine worktree (`<worktree>`) at the WNR3 review-fix commit / the WNR3 branch head.

## Riassunto (italiano)

1. **Compressione sui record reali.** Ho codificato con WNR3 103 expert V4.1 REAP-192 letti da WEX1 su U: (solo lettura): i 40 held-out dello studio 5090 e i 64 record WNR2 di produzione più grandi (uno, L39 E156, è in entrambi gli insiemi). Il modello è addestrato sugli stessi 80 expert del laboratorio. La versione non-CZ riproduce il modello lab byte per byte (id 0x299dc586).
   - **Profilo deciso (ceil-form + regola g + zero canonico + hash), 40 held-out: 0,85446 del raw**, contro 0,89944 di WNR2 (-5,00% dei byte WNR2, -4,50 punti di raw). Il target era ~0,853. Ogni record è più piccolo del suo WNR2: il guadagno va da 4,29 a 6,25 punti di raw.
   - Senza zero canonico: 0,88184 (riproduce il lab CFG, 0,88179 + il costo dell'hash).
   - **Coda (64 record WNR2 più grandi): 0,87238 del raw** contro 0,93289 di WNR2 (-6,49%). Il record più grande, L37 E150, è di **17.420.288 B** (WNR2: 18.079.744). Il record più grande di WNR2, L39 E120, scende da 18.321.408 a 17.358.848 B.
   - **Decodifica value-exact su CPU: 103/103.** Ogni record è stato decodificato di nuovo e confrontato con lo slab WEX1 riletto da U:. Le scale sono identiche. I nibble dei pesi sono identici, tranne -0 (0x8), che diventa +0 (0x0); il numero di questi nibble è uguale al conteggio dell'encoder (in media ~2,07 milioni per expert, il 5,8% dei pesi). Il non-CZ è byte-exact 103/103.
   - **Decodifica su GPU (5090, heavy.lock): 103/103 CZ e 103/103 non-CZ, ALL OK** (824 verifiche ciascuno) con l'helper WNR3.
   - Stima del container completo: **~123,4 GB (114,9 GiB)**, contro i 129,9 GB di WNR2 e i 144,4 GB di WEX1.
2. **Audit di corruzione su stream reali.**
   - GPU, flip esaustivi di ogni bit di header, primi 4 KiB della tabella delle lunghezze, tutti gli stream di scala della matrice 0 e 9 stream di pesi (primo, centrale e ultimo per matrice): 2.231.336 flip su 3 record WNR3, **0 accettati sbagliati**. Sugli stessi tipi di regione, WNR2 di produzione ne accetta sbagliati 2.081 su 489.168 (L0 E113, lo 0,43%; il 3,0% sui flip negli stream di pesi) e 1.583 su 1.245.496 (L39 E120). In L39 E120 WNR2 accetta anche 26 flip negli stream di scala.
   - Nessuna decodifica rifiutata ha toccato la destinazione.
   - Audit CPU (decoder dell'host): WNR3, 0 accettati sbagliati su 86.136 flip (esaustivi su header, lunghezze e primi stream, più 3.000 casuali sull'intero record). **WNR2 di produzione accetta sbagliati l'1,8-1,9% dei flip casuali su un record reale** (19/1000 e 18/1000), molto più dello 0,26-0,63% stimato prima.
3. **Microbench GPU (5090, stessa sessione bloccata, due giri ABAB).** p50 in ms per record: con WNR3-CZ, single 0,706-0,713 e batch-16 0,366-0,373; con WNR2 di produzione (helper distribuito, cceb4fb5) single 0,718-0,723 e batch-16 0,376-0,377. Quindi **WNR3 è alla pari o un po' più veloce per record (-1,5%)**, e i record sono più piccoli del 5%. Su record WNR2, l'helper WNR3 sta entro +1,6% / +6% al batch-16, che considero rumore (G2b: ±0,5%).
4. **Prova end-to-end dello zero canonico: preparata, non eseguita** (eseguita poi il 10/11: PASS, sezione "G7-lab result").
   - Ho aggiunto lo switch di laboratorio `V4_LAB_CANON_ZERO=1` nel worktree (la testa del branch WNR3). Trasforma ogni -0 in +0 in ogni slab di expert appena letto. Il motore compila da workspace pulito. Con lo switch spento passano test_deepseek_v4 e test_v4_wex_nr3; con lo switch acceso, test_v4_wex_nr3 fallisce dove deve (confronto byte dei pesi), e questo dimostra che lo switch è attivo.
   - Lo script `g7lab/run_g7_lab.sh` usa i bracci A1 B1 A2 B2 M1, con 4 prompt da 256 token. `compare_g7.py` fa il memcmp dei logits riga per riga. **Durata: ~55 min** (~11 min per braccio), ~20 min con `TOK=64`.
   - Il percorso `dg_moe<256>`: deepseek_v4.exe non chiama mai `dsv4_cuda_route_moe`, l'unica funzione che lo usa (e che rifiuta set con un numero di expert diverso da 256, quindi anche REAP-192). Il braccio M1 (masked + CZ) deve quindi dare logits identici, e lo script controlla che la riga "DeepGEMM routed experts active" non compaia mai.
5. **Piano disco: decisione del proprietario** (sezione 5; superato il 10/11: il container è stato costruito su U:, poi WEX1 e le copie WNR2 sono state cancellate, vedi [WNR3-PRODUCTION-20261110.md](WNR3-PRODUCTION-20261110.md)). La soluzione che consiglio: cancellare la copia WNR2 originale su E: (121 GiB, duplicato di U:) dopo un confronto sha256 con U:, e costruire WNR3 su E:. WEX1 resta su U: finché G7 non passa.

## 1. Encoding and decode of real records

Inputs: WEX1 `U:\kaira_wex\v41-reap\v4_flash_experts.wex` (read-only), WNR2 sizes from `t1_record_sizes.json` (the 7,680 records of `E:\...\v4_flash_experts.nr2.wex`). Encoder: `c/tools/build_nr3.py` + `wex_nr3_tools.dll` from `fresh-fix3` (= worktree the WNR3 review-fix commit). Each record was CPU-verified by the builder. `verify/analyze.py` then decoded it again (`wex_nr3_decode`) and compared it with the WEX1 slab read fresh from U:.

Selection (`verify/select_experts.py`): the 40 held-out experts of the 5090 study and the top 64 WNR2 records by size (cutoff 17,457,152 B, about p99). The 12 lab tail records are included. In total 103 distinct experts (L39 E156 is in both sets).

Models: trained by the builder on the lab's 80 training experts (`--train-lab80`). The non-CZ model is `0x299dc586`, byte-identical to the lab `model_CFG_rule_g.wnm2`. The CZ model is `0x27e8a1e0`. 0 gate failures; ceil mask 7 on all 103 records.

| set | n | WNR2 mean B (ratio) | WNR3 non-CZ (ratio) | **WNR3-CZ (ratio)** | CZ vs WNR2 | max WNR2 | max CZ |
|---|---|---|---|---|---|---|---|
| 40 held-out | 40 | 16,910,131 (0.89944) | 16,579,174 (0.88184) | **16,064,307 (0.85446)** | -5.00% | 17,485,824 | 16,449,536 |
| top-64 tail | 64 | 17,538,944 (0.93289) | 16,926,400 (0.90031) | **16,401,216 (0.87238)** | -6.49% | 18,321,408 | **17,420,288** (L37 E150) |
| all | 103 | 17,295,261 (0.91993) | 16,791,214 (0.89312) | 16,269,909 (0.86539) | -5.93% | 18,321,408 | 17,420,288 |

(sizes are on-disk record totals, 4096-aligned; raw = 18,800,640 B. The CZ gain over non-CZ is 2.74% of raw on the held-out set and 2.79% on the tail.)

- Canonical zero is worth 2.74 points of raw on the 40 held-out experts. The spec estimated 2.738 on the g0 basis; this is the first measurement with rule g.
- Full-container estimate: the WNR2 population mean (16,914,680 B) times the held-out CZ/WNR2 ratio 0.94998 gives 16,068,628 B per record, × 7,680 = **123.4 GB (114.9 GiB)** + model/table. Without CZ: 127.4 GB. This is conservative, because the tail ratio is better (0.935). The models are trained on 80 experts; the production model is trained on all 7,680 records.
- Value-exact CPU check: 103/103 CZ and 103/103 non-CZ (byte-exact). On CZ records: scales equal, weight nibbles equal except WEX1 0x8 -> 0x0, and the count of such nibbles equals the encoder's `cz_count` (held-out mean 2,061,030 per expert, 5.82% of weight nibbles).
- GPU (`test_decode_batch.exe` + `fresh-fix3/gpu-build/kaira_wex_cuda.dll`, sha 602e3f4d): WNR3-CZ 103/103 and non-CZ 103/103, 824 checks each, corruption statuses `[0,0,-1,0,-1,0]`, ALL OK (`verify/gpu_verify.log`).

Files: `verify/select.json`, `pairs.txt`, `encode.log`, `nr3_cz.bin` (3.6 GB) and `nr3_noncz.bin` (3.7 GB) with their `.json` reports, `model_*_lab80.wnm2`, `analyze.py`, `analyze.json` (per-record rows).

## 2. Corruption audit on real streams

The audit classifies every single-bit flip the way the engine treats it: first the owner/profile check (`wex_nr3_owner_ok` with the container flags: 97 for CZ, 33 for non-CZ), then the decoder. A flip is **rejected** (and the destination must be untouched), **harmless** (decodes to exactly the expected slab), or **WRONG** (accepted with other bytes). The WNR2 records are the production records of the same experts, read from E: (`verify/make_wnr2_bin.py` -> `wnr2_audit.bin`, model 0xab4261bd).

### 2a. GPU, exhaustive (`verify/audit/audit_gpu.cu`, helper batch API, 16 flips per call)

Regions, every bit: the record header (32 B), the first 4 KiB of the length table, every scale stream of matrix 0, and the first, middle and last weight stream of each matrix (9 streams).

| record | codec | header | lengths 4 KiB | scale streams m0 | 9 weight streams | total wrong / flips |
|---|---|---|---|---|---|---|
| L0 E113 WNR2 (prod) | 4 | 0/256 | 0/32,768 | 0/386,600 | **2,081/69,544 (2.99%)** | **2,081/489,168 (0.43%)** |
| L39 E120 WNR2 (prod) | 4 | 0/256 | 0/32,768 | **26/1,140,064** | **1,557/72,408 (2.15%)** | **1,583/1,245,496 (0.13%)** |
| L0 E113 WNR3-CZ | 5 | 0/256 | 0/32,768 | 0/369,088 | 0/66,160 | **0/468,272** |
| L39 E120 WNR3-CZ | 5 | 0/256 | 0/32,768 | 0/1,191,120 | 0/68,512 | **0/1,292,656** |
| L0 E113 WNR3 non-CZ | 5 | 0/256 | 0/32,768 | 0/369,104 | 0/68,280 | **0/470,408** |

Harmless: 0 everywhere. Rejected flips that touched the destination: 0 everywhere. WNR3 total: **0 wrong out of 2,231,336 flips** (the 23-bit hash predicts about 0.27 false accepts at this count). In WNR2, the sign flips in weight streams pass at 2-3%, and in L39 E120 (scale medians 121..123) even 26 scale flips pass. The hash closes both. Wall time: about 4-5 min per record.

### 2b. CPU (`verify/audit/audit_real.c`, the CPU decoder, 12 threads)

This runs the same decoder as the engine's host path. Two kinds of flips:
- **Exhaustive**: every bit of the header (32 B), the first 64 B of the length table, the first scale stream and the first weight stream of each matrix. (Only the first weight stream is swept: a CPU decode takes 0.6-0.8 s, so the GPU audit covers more.)
- **Random**: 1,000 flips uniform over the used bytes of the whole record (splitmix64, seed 20261110).

| record | exhaustive: wrong / flips (weight streams) | random over the whole record: wrong / flips | time |
|---|---|---|---|
| L0 E113 WNR2 (prod) | 703/25,544 (703/23,192 = 3.0%) | **19/1,000 (1.9%)** | 10.6 min |
| L39 E120 WNR2 (prod) | 301/34,336 (301/24,456 = 1.2%) | **18/1,000 (1.8%)** | 10.3 min |
| L0 E113 WNR3-CZ | **0/24,344** | **0/1,000** | 8.4 min |
| L39 E120 WNR3-CZ | **0/33,728** | **0/1,000** | 8.3 min |
| L0 E113 WNR3 non-CZ | **0/25,064** | **0/1,000** | 8.4 min |

Harmless: 0 everywhere. Every rejection left the slab untouched. CPU and GPU agree: the WNR2 weight-stream rate is about 3% on L0 E113 in both. **On whole real records, WNR2 accepts about 1.8-1.9% of single-bit flips with wrong output** (95% CI about 1.1-2.9%). That is higher than the 0.26-0.63% measured earlier on other data, because 94% of a record is weight payload, where the f[s] == f[s^8] sign flips pass. WNR3: 0 out of 86,136 CPU flips (the 23-bit hash predicts about 0.01 false accepts at this count). Output: `verify/audit/cpu_audit.jsonl`, runner `run_cpu_audit.sh`.

## 3. GPU decode microbench (RTX 5090, one heavy.lock session 02:57-03:31)

`test_decode_batch.exe` p50 in ms per record. Two ABAB rounds, in the same session as section 2a. The lock was free. No engine was running; a small lab test binary of another work package (`ws-v41gp-integ\c\deepseek_v4.exe`, 17 MB) was running.

| helper / records | round | single | batch 1 | batch 2 | batch 4 | batch 8 | batch 16 |
|---|---|---|---|---|---|---|---|
| deployed WNR2 (cceb4fb5) / WNR2 prod sample 64 | 1 | 0.723 | 0.733 | 0.529 | 0.435 | 0.404 | 0.376 |
| deployed WNR2 (cceb4fb5) / WNR2 prod sample 64 | 2 | 0.718 | 0.734 | 0.528 | 0.440 | 0.398 | 0.377 |
| WNR3 (602e3f4d) / WNR3-CZ 103 | 1 | 0.713 | 0.711 | 0.525 | 0.433 | 0.402 | 0.373 |
| WNR3 (602e3f4d) / WNR3-CZ 103 | 2 | 0.706 | 0.700 | 0.525 | 0.431 | 0.399 | 0.366 |
| WNR3 / WNR3-CZ 103 (decode pass, sect. 1) | - | 0.748 | 0.747 | 0.553 | 0.452 | 0.405 | 0.375 |
| WNR3 / WNR3 non-CZ 103 (decode pass) | - | 0.754 | 0.720 | 0.540 | 0.451 | 0.407 | 0.381 |
| WNR3 (602e3f4d) / WNR2 prod sample 64 | 1 | 0.743 | 0.755 | 0.546 | 0.447 | 0.413 | 0.399 |
| WNR3 (602e3f4d) / WNR2 prod sample 64 | 2 | 0.756 | 0.720 | 0.542 | 0.451 | 0.400 | 0.383 |

- WNR3-CZ against WNR2 in the ABAB rounds: batch-16 0.366-0.373 vs 0.376-0.377 (-1.6%), single 0.706-0.713 vs 0.718-0.723 (-1.5%). Per record it is equal or faster, and each record is 5% smaller. This is within the G4 bound (batch-16 <= WNR2 + 5%).
- The WNR3 helper on WNR2 records: batch-16 +6.1% in round 1 and +1.6% in round 2. The earlier G2b run measured -0.5..+0.5% on the same sample, so I treat this as noise. It is not settled: one more locked pair would settle it.

## 4. End-to-end canonical-zero proof: prepared, not run

*Run later on 2026-11-10: PASS, see section "G7-lab result" below.*

**Switch** (worktree `<worktree>`, branch wnr3-20261110, the WNR3 branch head): `V4_LAB_CANON_ZERO=1`. It is lab only; the application never sets it. `v4_read_expert_record` is now a thin wrapper around the unchanged reader (`v4_read_expert_record_io`). After every successful read (WEX1, WNR2 or safetensors), the wrapper runs `wex_nr3_canonical_zero` on the slab's weight bytes. Both callers go through it: the lease load and `read_private`. Every downstream use (RAM cache hits, rows16 pins, GPU mirror uploads) sees the canonical bytes. The GPU-direct fill skips the host slab, so the proof runs with `COLI_WEX_GPU_DIRECT=0` on WEX1, which has raw records only. Log: one ON line, then `experts=... nibbles=...` on the first expert and every 4096th.
- Fresh workspace `lab\wnr3-impl-20261110\fresh-verify`: the engine builds with rc 0. The only warning is the pre-existing signedness warning, now at line 18286. `test_deepseek_v4`: ok. `test_v4_wex_nr3`: ALL PASSED. With `V4_LAB_CANON_ZERO=1`, `test_v4_wex_nr3` logs `experts=1 nibbles=24721` and fails at the weight-bytes compare, as intended; this shows the switch is live on the store path.

**Script** `g7lab/run_g7_lab.sh` (with `g7lab/bin`: the fresh engine a55a4ce0, `coli_cuda_dsv4_dg.dll` b45eb5d2 from `artifacts\v41-prefix-gpufix-20261110` (= the worktree's CUDA source), and cudart/cublasLt from ws-ab):
- It uses the heavy.lock protocol (poll every 2 min, up to 3 h; it waits while any deepseek_v4.exe runs) and the RAM rule (16 GB when >= 15 GB is available, else 13, the same in every arm). The configuration is G3 arm A: WEX1 on U:, GPU tiers, the decode GPU pipeline, CPU head, GPU-direct 0.
- Arms, in this order: A1 (off), B1 (`V4_LAB_CANON_ZERO=1`), A2 (off), B2 (on), M1 (on + `DSV4_CUDA_MOE_MASKED=1`). Prompts: code1, chat_it1, reason1 and ctx2k from `ab/prompts.jsonl`, at `TOK=256`. Every arm writes `V4_LOGITS_DUMP` to `D:/kaira_wnr3_lab/g7lab`.
- `g7lab/compare_g7.py` validates every arm's log. The log must show: the WEX1 container active, 40 dense uploads, decode on the gpu-pipeline, no cpu-union fallback, no "DeepGEMM routed experts active", and the switch ON exactly in the B/M arms with experts > 0. Then: A1 == A2 byte-identical (otherwise INCONCLUSIVE), and B1, B2 and M1 == A1 byte for byte (otherwise the first differing logits row), with equal texts. Verdict: PASS / FAIL / INCONCLUSIVE / INVALID. I tested the compare logic on synthetic dumps (PASS, and FAIL with the right row).
- **Duration:** about 11 min per arm (about 20 s to ready; about 1.8-2.1 tok/s; code1 stops near 130 tokens; ctx2k adds a 2k prefill). That makes **about 55 min for the 5 arms**, plus RAM waits. With `TOK=64` it is about 20 min. Dumps: about 0.5 GB per arm on D:.
- Command: `DUMP=D:/kaira_wnr3_lab/g7lab bash lab/wnr3-impl-20261110/g7lab/run_g7_lab.sh`.

**dg_moe<256> coverage.** The masked `dg_moe<256>`/`<128>` kernels are reached only from `dsv4_cuda_route_moe` (`backend_cuda_dsv4.cu:3707`) and the 2-GPU EP2 path. `route_moe` also refuses `set->count != 256`. `deepseek_v4.exe` calls neither: in `deepseek_v4.c` the only route-MoE call is `dsv4_cuda_route_moe_ids_batch`, which uses `dg_moe_batch_contiguous`, one of the 0-difference default paths. On V4.1 REAP-192 the path is therefore unreachable. In M1, `DSV4_CUDA_MOE_MASKED=1` must change no byte, and the log check proves at run time that the route kernel never ran. A non-deterministic masked kernel can only be proven with a deterministic mode or a logprob tolerance. That only matters for the V4 256-expert model, which no WNR3 container serves.

What G7-lab does not cover: the WNR3 decoders themselves in the engine, which G4 and the engine store tests cover. It replaces the spec's G7 subset containers (2 × 12.4 GB on D:) with the switch, so no container is written.

## 5. Disk plan for the full WNR3-CZ container (~123.4 GB = 114.9 GiB): the user decides

> **Superseded 2026-11-10.** The container was built on U: (123.3 GB) and copied to C:; after the switch, WEX1 and the WNR2 copies on U:, C: and E: were deleted. The table below is the state before the build. See [WNR3-PRODUCTION-20261110.md](WNR3-PRODUCTION-20261110.md).

Free space now (2026-11-10 03:00): **C: 50.3 GiB, D: 79.9, E: 31.6, U: 10.9**. E: has dropped from about 89 since the plan was written; I did not find the cause in the folders listed below. What is on disk (GiB):

| drive | what | size |
|---|---|---|
| U: | WEX1 `kaira_wex\v41-reap\v4_flash_experts.wex` (source; keep until G7 passed and the user decides) | 134.5 |
| U: | WNR2 `v4_flash_experts.nr2.wex` (production, primary) | 121.0 |
| U: | models: REAP192-MXFP4 151.6, Qwen3.8-Flash-Next-FP8 172.8, small models 11.6 | |
| C: | WNR2 mirror `C:\kaira_wex\v41-reap` (read split with U:) | 121.0 |
| E: | WNR2 original `E:\kaira_wex\v41-reap` (duplicate of U:) | 121.0 |
| D:, E: | `DeepSeek-V4.1-Flash-1017-engram` (94.6 each; the engine reads the engram shards from C:/models and E:/models: check before deleting) | 94.6 + 94.6 |
| D: | `kaira_wnr3_lab` (G3/G4 bins, records64, subset) | 5.6 |
| C: | `lab\wnr3-impl-20261110\verify\nr3_*.bin` (this run) | 6.8 |

The builder needs the container size plus a margin on the target drive. `build_nr3.py` container mode currently asks for the raw sum + 1 GiB, which is 135.5 GiB. The spec's `--full` rule (0.90 × source + margin) would need about 122 GiB. Before the build, relax that check to about 125 GiB. `--full` is refused outright by the builder (no `--allow-full`; a full build is container mode with all 40 layers, spec P5).

Options:
1. **Recommended: build on E: after deleting the E: WNR2 original.** First `sha256sum` it against the U: copy (identical files, so nothing is lost). E: then has about 152 GiB free; after the build, about 37 GiB. Production (U: + C: mirror) does not change during the build. Afterwards WNR3 can replace WNR2 on U:/C: (copy over; U: needs the WNR2 file deleted first, because it has 10.9 GiB free).
2. Build on C: after deleting the C: WNR2 mirror (121 GiB): C: then has about 171 GiB free. Until WNR3 replaces it, the production read split loses the mirror, which costs some decode speed.
3. Build on D: after freeing about 45 GiB. The candidate is the D: engram copy (94.6 GiB), but only if it really is a duplicate the engine does not read; `COLI_MODEL_DIRS` names C:/models and E:/models. Also delete `D:\kaira_wnr3_lab` (5.6).
4. Do not touch: WEX1 on U: (until G7 passes and the user decides; WNR3-CZ cannot reproduce WEX1 bytes) and the REAP safetensors on U:.

In every case, the samples of this run can go (6.8 GiB on C:, 5.6 GiB on D:) once the user no longer needs them.

## Not done / open

- G7-lab not run (prepared only; it needs about 55 min of GPU under heavy.lock). *Done later the same day: PASS (section "G7-lab result").*
- The models were trained on 80 experts, not on all 7,680. The full container's size and max record are extrapolated from 103 records, not measured. *Measured later on the full build: 123,306,192,896 B, largest record 17,420,288 B.*
- The WNR3 helper vs WNR2 records at batch-16 differs by +1.6% / +6.1% between rounds (noise, per G2b); an extra locked pair would settle it.
- The CPU/GPU differential (`test_wex_nr3_gpu.exe`) was not rerun on the 103-record bins (it holds about 10 GB of mutated copies). It passed on the 52-record G4 bins.

## G7-lab result (2026-11-10, TOK=64)

VERDICT PASS. Arms A1, B1, A2, B2 and M1 each ran 4 prompts (code1, chat_it1, reason1, ctx2k), 256 logits rows per arm, all marked VALID. A1 and A2 are byte-identical. B1, B2 and M1 (65,536 canonical-zero experts) are byte-identical to A1, and the texts are equal. Logits hash: 19264538418be15d. M1 also shows that `DSV4_CUDA_MOE_MASKED=1` changes no byte on V4.1 REAP-192.

The B and M arms decode at about 1.0-1.16 tok/s against about 1.74-2.15 tok/s for A, and take 40 s to become ready against 17 s. That cost comes from the lab switch, which rewrites every FP4 nibble on the CPU after each read. It does not measure a WNR3 decoder. Per-arm numbers: [`research/wnr3-20261110/g7lab/RESULT-20261110.md`](../../research/wnr3-20261110/g7lab/RESULT-20261110.md).
