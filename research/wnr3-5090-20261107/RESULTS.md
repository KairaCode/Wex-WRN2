# WNR3 & DeepSeek V4.1 on NVIDIA GeForce RTX 5090 (sm_120): Final Measurements

**Date:** 2026-11-08  
**Hardware (measured 2026-11-08 02:05 via CIM + nvidia-smi):** NVIDIA GeForce RTX 5090 (Blackwell `sm_120`, 32,607 MiB VRAM, 170 SMs, PCIe gen5 x16 max, driver **621.01**); AMD Ryzen 9 7950X3D (16 cores / 32 threads); **32 GB RAM** (32 GiB of DIMMs, 31.15 GiB visible to Windows, ~15 GiB free at measurement); Windows 11 Pro 10.0.26300.  
**Toolchain:** nvcc 13.4 (V13.4.92); CuPy 14.2.0 + NVRTC 12.9 / runtime 12.9 (driver API 13.05) for the ceil-form micro-bench (same versions as the RTX 3070 run); MSYS2 gcc 16.2.0; Python 3.14.8.  
**Model:** DeepSeek V4.1 Flash REAP-192 MXFP4 (`U:\models\DeepSeek-V4.1-Flash-1017-REAP192-MXFP4`)  
**Production Archive:** `E:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex` (7,680 experts)  

---

## Riassunto Esecutivo (Italiano)

1. **Ceil-Form su RTX 5090 `sm_120` (Task 3 + Porting KAIRA `cp.async` ring kernel):**
   - **Micro-benchmark NVRTC (`ceil-gpu/ceilbench.py`):** Tutti i `138.240` stream verificati al 100% byte-exact rispetto ai safetensors originali su tutte le varianti (`A259`, `CF5`, `D`, `CFD5`, `DB2`, `CFDB2`).
     - Tempo di decodifica GPU `sm_120` per 8 esperti (`m24`, 24 matrici):
       - `A259` (baseline WNR2): `757,9 µs`.
       - `CF5` (ceil-form + deferred scale patch): **`752,8 µs`** (`0,993x`, **nessun overhead di decodifica, anzi leggermente più veloce della baseline**, con `-12,04%` di byte di scala (`1.242.888 -> 1.093.228`) e `-2,21%` di byte di peso su questo set di stream; il `30,6%` è la quota di blocchi da 32 pesi riscritti, non di codici di scala eliminati. Sul record held-out il ceil-form vale `-1,76/-1,79%` del raw).
       - `DB2` (dual-chain sign-bypass): **`591,0 µs`** (speedup `1,28x` rispetto ad `A259`), ma costa **`+0,36%` di byte di peso** (`134.946.449` contro `134.462.859`).
       - `CFDB2` (ceil-form + dual-chain sign-bypass): **`611,5 µs`** (speedup `1,24x` rispetto ad `A259`; byte di peso `-1,85%`, pesi+scale `-1,75%` del raw su questo set).
       - Rispetto alla RTX 3070 (`sm_86`): l'RTX 5090 è **`~5,5x` più veloce** (`752,8 µs` vs `4.185 µs`).
   - **Porting End-to-End nel kernel di produzione KAIRA (`ceil-kaira/wex_cuda_ceil.cuh` + `kaira_wex_cuda_ceil.dll`):**
     - Implementata l'inversione branchless `T^-1` (`wex_cuda_ceil_dbl8` + `wex_cuda_ceil_fix_scales` vettorializzato `uint4` con separazione race-free tra scale codificate e scale finali) direttamente nel kernel ring `cp.async` a 567 righe in shared memory (`wex_cuda_nr2_weights<1>`).
     - Compilato per `sm_120` con **0 spill stores, 0 spill loads** (`91` registri, `26.336 B` shared memory).
     - Sui 40 esperti held-out di test: **media record `16.578.252,8 B` (`0,88179` del raw)** contro `16.909.926,4 B` (`0,89943`) di WNR2 baseline (`-331.673,6 B/record`, **`-1,764%` del raw**).
     - Sui 12 record di coda più grandi (`big`): il massimo assoluto scende da `18.321.408 B` (produzione) / `18.485.248 B` (`Model A`) a **`17.985.536 B`** (`CFG`, `-335.872 B` vs produzione, `-499.712 B` vs `Model A`).
     - Verifica `test_decode_batch.exe` + `kaira_wex_cuda_ceil.dll`: **100% byte-exact** rispetto ai pesi e scale raw originali pre-trasformazione (`320/320` su `CF` e `CFG`, `416/416` su `CFG_all52`, più i 6 test di corruzione `[0,0,-1,0,-1,0]`), con latenza batch-16 p50 di **`0,380–0,382 ms/record`** (`0,401 ms/record` a batch-8), virtualmente indistinguibile da `kaira_wex_cuda.dll` (`0,372–0,385 ms/record`).

2. **Bit-Test Zero Canonico su `sm_120` / `sm_120a` (Task 4):**
   - Le 11 fasi della tabella di `zero/ZERO-RESULTS.md` §3 sommano **`83.841.024` output confrontati** (kernel generici, API generica default/TC/unbatched, micro-test MMA DeepGEMM, smoke MoE DeepGEMM, kernel DeepGEMM, 4 run sulle DLL distribuite; per le fasi 1 e 7 la riga riporta il conteggio per kernel, non la somma sui kernel): **0 valori differenti / 0 bit differenti** tra input con zero canonico (`+0`, codice `0x0`) e non-canonico (`-0`, codice `0x8`). Il percorso DeepGEMM masked `dg_moe<256>` (`dg_moe.json`) non è in questo conteggio: vedi la riserva sotto.
   - Validato su tutti i 16 kernel CUDA generici, sull'API pubblica C (`dsv4_cuda_expert_group`, `dsv4_cuda_moe`), sulle istruzioni Tensor Core block-scaled `mma.sync ... kind::mxf8f6f4.block_scale` (scale_vec::1X, ue8m0) che DeepGEMM usa su `sm_120a` (sm_12x non ha `tcgen05`) e sulle DLL di produzione distribuite (`coli_cuda_dsv4.dll` e `coli_cuda_dsv4_dg.dll`).
   - I controlli positivi a singolo bit hanno rilevato `39.179.124` valori differenti in totale sulle 11 fasi, quindi il test vede le differenze quando ci sono.
   - Sui percorsi di default tutti gli zero esatti in uscita sono `+0.0f`. **Eccezione:** le uscite bf16 FC1/FC2 del percorso masked `dg_moe<256>` contengono `-0` (8/8 e 26/22 output, A/B).
   - **Riserva, percorso masked `dg_moe<256>` (non di default):** non è deterministico nemmeno A contro A (da `195.779` a `852.405` valori differenti tra due run identiche: FC1 195.779, uscita instradata 540.575, FC2 852.405), quindi lì canonico contro originale non si può dimostrare (FC1 204.497, uscita 533.757, FC2 846.380 differenze, dello stesso ordine del rumore A-A). I percorsi DeepGEMM di default (`dg_moe_contiguous_fast<256>` decode, `dg_moe_batch_contiguous` prefill) danno 0 differenze e sono deterministici.
   - **Conclusione:** WNR3z (`-0 -> +0`) è bit-identico su tutti i percorsi di default testati; il percorso masked richiede una modalità deterministica o un controllo delle logprob prima di usarlo con WNR3z.

3. **Valutazione Empirica Regola `g` e Conformità Bit-a-Bit C vs Python (Task 2):**
   - **Problema di `Model B` (ancora fissa 120):** Sebbene sui 40 esperti di test risparmi `-0,194%` del raw, sulle matrici di coda (layer 36–39 con mediana 122–123) lo `srel` schiacciato contro il clamp `[-3, +3]` provocava una disastrosa espansione fino a **`+13,2%` su L39 E120** (`20,98 MB`, oltre la dimensione raw non compressa!).
   - **Successo della Regola `g` (`Model G`, coppia adiacente più densa `argmax_v (h[v] + h[v+1])`, ancora `-2`):**
     - Sul training set (80 esperti): tutte le matrici hanno coppia più densa `120`, quindi `model_G_rule_g.wnm2` è **identico byte-per-byte** a `model_B_g0_120.wnm2` (`id = 0x750c0967` = 1963723111).
     - Sui 40 esperti di test: identico risparmio di **`-0,1944%` del raw** (`-36.556,8 B/esperto`), con `Model G <= Model A` sul **100% degli esperti (`40/40`)**.
     - Sulla coda dei 12 record più grandi (`big`): **elimina totalmente l'esplosione!**
       - Record `L39_E120`: scende da `20.975.616 B` (`Model B`) a **`18.411.520 B`** (`Model G`), risultando persino **`73.728 B` (`-0,39%`) più compatto di `Model A`** (la baseline con mediana inferiore)!
       - Record massimo tra i 12 di coda: **`18.464.768 B`** (inferiore al max di `Model A` di `18.485.248 B`).
       - Delta medio sui 12 record di coda: `Model B` era peggiore del `+2,718%` rispetto ad `A`; **`Model G` è migliore di `-0,759%` (`-142.677 B`) rispetto ad `A` e di `-3,477%` (`-653.653 B`) rispetto a `B`**.
     - **Decodifica CPU e GPU 100% byte-exact** (`40/40` test + `12/12` coda = `52/52` record, `416/416` verifiche in `gpu_decode_G_all52.log`) con il decoder C e CUDA KAIRA di produzione (`kaira_wex_cuda.dll`) **completamente invariato** (legge l'ancora memorizzata nei byte `24..26`).
   - **Conformità C vs Python (OPEN-K #11 / T13):** Verificato in `t2_conformance_and_g_bins.json` che il codec C di KAIRA (`wex_nr2_encode` in `g0codec.dll`) e l'encoder Python di riferimento del Team 4 (`wnr_common.py`) producono **blob WNM2 (`59.408 B`) e tutti i `40/40` record WNR2 di test identici byte-per-byte al 100%**.

4. **Dimensioni Record V4.1 REAP-192 (Task 1):**
   - `7.680/7.680` record analizzati in `E:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex`:
     - Max: `18.321.408 B` (L39 E120, rapporto `0,9745` del raw `18.800.640 B`).
     - Media: `16.914.680 B` (`0,8997` del raw), p99: `17.453.056 B` (`0,9283`), Min: `16.809.984 B` (`0,8941`).
   - Budget a slot fissi (su max `18.321.408 B`): `+2,62%` di capacità a parità di byte, contro `+11,15%` con indicizzazione byte-packed variabile.

---

## 1. Ceil-Form GPU Benchmark Details (RTX 5090 `sm_120`)

### 1.1 Micro-Benchmark NVRTC (`ceil-gpu/bench_5090.json`, `ROUNDS=41`, `sm_120`)

| Variante | Descrizione | m1 (µs) | m3 (µs) | m24 (µs) | 8streams (µs) | Ratio m24 vs A259 | Risparmio Byte vs Baseline | Bit / Peso |
|---|---|---|---|---|---|---|---|---|
| **A259** | Baseline WNR2 (`all-in-one nt128/lb`) | 87.1 | 148.9 | 757.9 | 761.2 | 1.000x | Baseline (pesi 134.462.859 B, scale 1.242.888 B) | 3.7665 |
| **CF5** | Ceil-Form + Deferred Scale Patch | 86.8 | 147.2 | **752.8** | **758.1** | **0.993x** | **pesi -2.21%, scale -12.04%; -2.07% del raw** | 3.6927 |
| **D** | Sign-Bypass (branchless sign) | 78.4 | 134.1 | 651.5 | 659.0 | 0.860x | pesi -124 B (~0) | 3.7665 |
| **CFD5** | Ceil-Form + Sign-Bypass | 79.1 | 135.0 | 658.5 | 664.2 | 0.869x | pesi -2.21%, scale -12.04%; -2.07% del raw | 3.6927 |
| **DB2** | Dual-Chain Sign-Bypass | 72.3 | 121.5 | **591.0** | **598.4** | **0.780x (1.28x)** | **pesi +0.36% (134.946.449 B); +0.32% del raw** | 3.7665 |
| **CFDB2** | Ceil-Form + Dual-Chain Sign-Bypass | 74.2 | 125.1 | **611.5** | **619.0** | **0.807x (1.24x)** | **pesi -1.85%, scale -12.04%; -1.75% del raw** | 3.6927 |

- **Colonna byte:** da `coded_weight_bytes` / `coded_scale_bytes` di `bench_5090.json` (8 expert in-sample, 24 matrici; raw 150.405.120 B). Il `-1,76/-1,79%` del ceil-form è la misura sui record reali held-out, non su questo set.
- **Convalida Byte-Exact:** `138.240` stream su `138.240` verificati identici bit a bit contro safetensors.
- **Controllo Negativo (`CF0`):** `138.240` fallimenti rilevati (prova di sensibilità dello stream).

### 1.2 Porting End-to-End nel Kernel `cp.async` Ring di KAIRA (`ceil-kaira/`)

Portata l'inversione ceil-form `T^-1` nel decoder CUDA di produzione KAIRA ([`wex_cuda_ceil.cuh`](ceil-kaira/wex_cuda_ceil.cuh), `kaira_wex_cuda_ceil.dll` (built by `ceil-kaira/build_ceil.cmd`; not in this repo)):
- `wex_cuda_nr2_scales` decodifica le scale ceil-form in `raw + sc_aligned` (offset separato e 16-byte aligned).
- `wex_cuda_nr2_weights<1>` legge le scale ceil-form per calcolare i contesti `srel` identici all'encoder, accumula la maschera a 64 bit `ceil_emask` dei blocchi con `maximum == 5u`, applica `wex_cuda_ceil_dbl8` in-register prima della scrittura `uint4` dei pesi e invoca `wex_cuda_ceil_fix_scales` per scrivere le scale finali decrementate in `raw + plan.scale_offset[m]` senza alcuna race condition tra blocchi.

#### Compressione End-to-End (Record allineati a 4096 B, Raw = `18.800.640 B`)

| Split | Metrica | Produzione (`nr2.wex`) | `Model A` (Mediana) | `Model G` (Regola `g`) | `Model CF` (Ceil + Mediana) | `Model CFG` (Ceil + Regola `g`) |
|---|---|---|---|---|---|---|
| **40 Test Held-Out** | Media Record (B) | 16.906.752,0 | 16.909.926,4 | 16.873.369,6 | 16.578.764,8 | **16.578.252,8** |
| **40 Test Held-Out** | Rapporto / Raw | 0,89926 | 0,89943 | 0,89749 | 0,88182 | **0,88179** |
| **40 Test Held-Out** | Delta vs `Model A` | -3.174,4 B | Baseline | -36.556,8 B (-0,194%) | -331.161,6 B (-1,761%) | **-331.673,6 B (-1,764%)** |
| **12 Tail (`big`)** | Media Record (B) | 17.804.629,3 | 17.934.677,3 | 17.792.000,0 | **17.461.930,7** | 17.516.544,0 |
| **12 Tail (`big`)** | Max Record (L39 E120) | 18.321.408 | 18.485.248 | 18.411.520 | 18.038.784 | **17.985.536** |

#### Latenza di Decodifica GPU Batch (`test_decode_batch.exe` su RTX 5090 `sm_120`, `ms/record` p50)

| Configurazione (`DLL` + Dataset) | Record / Verificati | Record MiB | Single | Batch 1 | Batch 2 | Batch 4 | Batch 6 | Batch 8 | Batch 10 | Batch 16 |
|---|---|---|---|---|---|---|---|---|---|---|
| `kaira_wex_cuda.dll` + `records_A.bin` (WNR2 Mediana) | 40 / 320 (`ALL OK`) | 16.11 | 0.725 | 0.705 | 0.531 | 0.438 | 0.414 | 0.397 | 0.391 | **0.372** |
| `kaira_wex_cuda.dll` + `records_B.bin` (WNR2 g0=120) | 40 / 320 (`ALL OK`) | 16.08 | 0.722 | 0.697 | 0.535 | 0.437 | 0.411 | 0.395 | 0.391 | **0.371** |
| `kaira_wex_cuda.dll` + `records_G.bin` (WNR2 Regola `g`) | 40 / 320 (`ALL OK`) | 16.10 | 0.736 | 0.720 | 0.533 | 0.449 | 0.413 | 0.401 | 0.393 | **0.379** |
| `kaira_wex_cuda.dll` + `records_G_all52.bin` (40 test + 12 tail) | 52 / 416 (`ALL OK`) | 16.10 | 0.713 | 0.700 | 0.522 | 0.439 | 0.420 | 0.402 | 0.398 | **0.385** |
| `kaira_wex_cuda_ceil.dll` + `records_CF.bin` (Ceil + Mediana) | 40 / 320 (`ALL OK`) | 15.80 | 0.742 | 0.740 | 0.531 | 0.452 | 0.424 | 0.402 | 0.401 | **0.385** |
| `kaira_wex_cuda_ceil.dll` + `records_CFG.bin` (Ceil + Regola `g`) | 40 / 320 (`ALL OK`) | 15.80 | 0.736 | 0.708 | 0.540 | 0.442 | 0.416 | 0.401 | 0.395 | **0.380** |
| `kaira_wex_cuda_ceil.dll` + `records_CFG_all52.bin` (40 test + 12 tail) | 52 / 416 (`ALL OK`) | 15.80 | 0.733 | 0.717 | 0.529 | 0.445 | 0.428 | 0.401 | 0.389 | **0.382** |

---

## 2. Rule `g` Empirical Evaluation vs Model A & Model B (Task 2)

Eseguito su `U:\models\DeepSeek-V4.1-Flash-1017-REAP192-MXFP4` e container di produzione `E:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex`.

### 2.1 Held-Out Test Set (40 Esperti, 1 per layer)

| Metrica | Model A (Mediana Inferiore) | Model B (Ancora Fissa 120) | Model G (Regola `g`, Coppia Densa) |
|---|---|---|---|
| Model ID (FNV-1a of the WNM2 blob) | `0xfc992a63` (`4237896291`, = team 4 even.npz WNR2 model) | `0x750c0967` (`1963723111`) | `0x750c0967` (byte-identical to B: sha256 9871fa35... for model_B, model_G and model_C) |
| Dimensione Media Totale | 16.909.926,4 B | 16.873.369,6 B | **16.873.369,6 B** |
| Rapporto sul Raw (18,80 MB) | 0,89943 | 0,89749 | **0,89749** |
| Delta vs Model A (byte) | Baseline | -36.556,8 B | **-36.556,8 B** |
| Delta vs Model A (% raw) | Baseline | -0,1944% | **-0,1944%** (IC 95%: `[-0,412%, -0,026%]`) |
| Esperti con `G <= A` | - | 40 / 40 | **40 / 40 (100%)** |
| Decodifica Esatta (CPU & GPU) | 40 / 40 (`320/320`) | 40 / 40 (`320/320`) | **40 / 40 (`320/320` Byte-Exact)** |
| Conformità C vs Python (`wnr_common.py`) | **40 / 40 Byte-Identical** | - | - |

### 2.2 Tail Analysis (12 Record di Produzione più Grandi)

| Layer & Expert | Mediane Reali (`w1, w2, w3`) | Ancore Regola `g` | Model A (Byte) | Model B (Fissa 120) | Model G (Regola `g`) | Ceil+`g` (`CFG`, Byte) | Delta `G` vs `A` | Delta `G` vs `B` |
|---|---|---|---|---|---|---|---|---|
| **L39_E120** (Worst) | `[121, 123, 122]` | `[121, 122, 121]` | 18.485.248 | 20.975.616 (+13.2%) | **18.411.520** | **17.985.536** | **-73.728 B (-0.39%)** | **-2.564.096 B (-13.64%)** |
| **L37_E150** | `[121, 122, 122]` | `[120, 122, 121]` | 18.284.544 | 19.582.976 (+6.9%) | **18.464.768** | **17.973.248** | +180.224 B (+0.96%) | **-1.118.208 B (-5.95%)** |
| **L38_E191** | `[121, 122, 121]` | `[120, 121, 121]` | 18.239.488 | 18.903.040 (+4.4%) | **18.075.648** | **17.973.248** | **-163.840 B (-0.87%)** | **-827.392 B (-4.40%)** |
| **L36_E184** | `[121, 122, 122]` | `[120, 122, 121]` | 18.223.104 | 19.881.984 (+8.8%) | **18.325.504** | **17.911.808** | +102.400 B (+0.54%) | **-1.556.480 B (-8.28%)** |
| **L39_E105** | `[121, 122, 121]` | `[120, 121, 120]` | 18.046.976 | 17.739.776 | **17.752.064** | **17.879.040** (`CF`: **17.170.432**) | **-294.912 B (-1.57%)** | +12.288 B (+0.07%) |
| **L39_E191** | `[121, 122, 121]` | `[120, 121, 120]` | 17.915.904 | 17.637.376 | **17.448.960** | **17.133.568** | **-466.944 B (-2.48%)** | **-188.416 B (-1.00%)** |
| **L37_E059** | `[121, 122, 121]` | `[120, 121, 120]` | 17.883.136 | 17.432.576 | **17.285.120** | **17.059.840** | **-598.016 B (-3.18%)** | **-147.456 B (-0.78%)** |
| **L15_E029** | `[120, 121, 121]` | `[119, 120, 120]` | 17.723.392 | 17.846.272 | **17.592.320** | **17.244.160** | **-131.072 B (-0.70%)** | **-253.952 B (-1.35%)** |
| **L32_E049** | `[121, 121, 121]` | `[120, 120, 120]` | 17.612.800 | 17.182.720 | **17.182.720** | **17.125.376** | **-430.080 B (-2.29%)** | +0 B (+0.00%) |
| **L39_E060** | `[121, 121, 121]` | `[120, 120, 121]` | 17.592.320 | 18.010.112 (+1.7%) | **17.698.816** | **17.321.984** | +106.496 B (+0.57%) | **-311.296 B (-1.66%)** |
| **L37_E149** | `[121, 121, 121]` | `[120, 121, 120]` | 17.592.320 | 17.547.264 | **17.522.688** | **17.170.432** | **-69.632 B (-0.37%)** | **-24.576 B (-0.13%)** |
| **L36_E162** | `[121, 121, 121]` | `[120, 121, 121]` | 17.616.896 | 18.608.128 (+5.3%) | **17.743.872** | **17.420.288** | +126.976 B (+0.68%) | **-864.256 B (-4.60%)** |
| **MAX CODA** | - | - | **18.485.248 B** | **20.975.616 B** | **18.464.768 B** | **17.985.536 B** | **-20.480 B** | **-2.510.848 B** |
| **MEDIA CODA** | - | - | **17.934.677 B** | **18.445.653 B** | **17.792.000 B** | **17.516.544 B** | **-142.677 B (-0.759%)** | **-653.653 B (-3.477%)** |

---

## 3. Canonical Zero Bit-Test Summary (Task 4)

Vedi i documenti completi in [ZERO-RESULTS.md](zero/ZERO-RESULTS.md) e [ZERO-TABLES.md](zero/ZERO-TABLES.md).

- **Output totali confrontati:** `83.841.024` (somma delle 11 righe di fase di `ZERO-RESULTS.md` §3; le fasi 1 e 7 contano per kernel; escluso `dg_moe.json`)
- **Valori differenti (`+0` codice `0x0` vs `-0` codice `0x8`):** **0**
- **Bit differenti (`memcmp` / IEEE-754 uint32):** **0**
- **Differenze controlli positivi (`0x0 -> 0x1`):** `39.179.124` valori
- **Zero negativi in output:** `0` sui percorsi di default (kernel FP32, cuBLASLt MXFP8, `mma.sync` block-scaled `sm_120a`, DeepGEMM contiguous). Il percorso masked `dg_moe<256>` produce `-0` nelle uscite bf16 FC1/FC2.
- **Percorso masked `dg_moe<256>` (non di default):** non deterministico (A contro A: 195.779 / 540.575 / 852.405 valori differenti su FC1 / uscita / FC2), quindi canonico contro originale non è dimostrabile lì: serve una modalità deterministica o un controllo delle logprob. Dati in `zero/ZERO-TABLES.md`, sezione `dg_moe.json`.
- Nota: `ZERO-RESULTS.md` ha in cima un riquadro di correzione: i valori misurati sono 83.841.024 output (somma delle fasi) e driver 621.01, e il percorso masked `dg_moe<256>` è escluso.

---

## 4. Scripts (every path; folder `<KAIRA>\lab\wnr3-5090-20261107\`)

- **Lock and runners:** `lock.sh` (heavy.lock protocol, atomic create, 60 s poll, released on exit), `run_phase1_cpu.sh` (task 2 g0 + task 3 sample/counts/ceilenc), `run_phase2_gpu.sh` (task 2 GPU decode + ceilbench), `run_phase3.sh` (rule-g re-check + ceilbench rerun; stopped at 02:04, see section 6).
- **Task 1:** `t1_record_sizes.py` -> `t1_record_sizes.json`, `t1_record_sizes.out`, `t1_tail.out`.
- **Task 2 (WNR2-g0 / rule g):** `g0/wex_neural_rans_g0.h` (scratch copy of `src/colibri/c/wex_neural_rans.h`; only `wex_nr2_median` changed: fixed anchor when `g0_anchor >= 0`, rule g when `-2`), `g0/g0_enc.c` (trainer + encoder from the scratch copy), `g0/g0_dec.c` (UNCHANGED src decoder, separate translation unit), `g0/g0codec.dll` (MSYS2 gcc -O2), `g0/t2_g0.py` (models A/B, 40 test + 12 tail, CPU decode, GPU bins), `g0/t2_rule_g.py` -> `t2_rule_g.json` (rule g), `g0/t2_conformance_and_g_bins.py` -> `t2_conformance_and_g_bins.json` (C vs Python byte conformance, G bins), `g0/t2b_rule_g.py` (independent rule-g re-check, interrupted; its partial log `g0/t2b_rule_g.partial.log`; its `g0/model_C_rule_g.wnm2` is byte-identical to model G), GPU decode logs `g0/gpu_decode_{A,B,G,G_big,G_all52}.log` (engine helper `lab\v41-gpu-20261107\w1\build\kaira_wex_cuda.dll` sha cceb4fb5... + `test_decode_batch.exe`, source `lab\wex-prod-20261105\test_decode_batch.cu`).
- **Task 3 (ceil-form):** `t3_make_sample.py` (160-expert sample from local safetensors, sha256 vs archived manifest, KAIRA C counts), `team1/mxcount.py`, `team1/mxceil.py` (team-1 copies), `ceil-gpu/src/ceilenc.py`, `ceil-gpu/src/wnr2fmt.py`, `ceil-gpu/src/wnr2dec.cu`, `ceil-gpu/src/ceilbench.py` and `ceil-gpu/src/bench2.py` (team-4 copies; only change: `-arch` from env `CEIL_ARCH`, default `sm_120`, and nvidia-smi from PATH), other team-4 helpers in `ceil-gpu/src/` copied unchanged; venv `venv\` (cupy-cuda12x 14.2.0, cuda-toolkit[cudart,nvrtc] 12.9.2). Outputs `ceil-gpu/bench_5090.{json,log}`, `bench_5090_table.txt`, `ceilenc.log`, `encode_report_*.json`.
- **Ceil-form port into the KAIRA kernel:** `ceil-kaira/wex_cuda_ceil.cuh`, `ceil-kaira/wex_cuda_bridge_ceil.cu`, `ceil-kaira/gen_ceil_cuda.py`, `ceil-kaira/make_ceil_records.py`, `ceil-kaira/build_ceil.cmd` -> `kaira_wex_cuda_ceil.dll`, logs `gpu_decode_{CF,CFG,CFG_all52}.log`, `ceil_kaira_records.json`.
- **Task 4 (canonical zero):** `zero/src/zt_harness.cu`, `zero/src/zt_run.py`, `zero/src/zt_summary.py`, `zero/build_generic.cmd`, `zero/build_dg.cmd`, `zero/phase_*.cmd`, `zero/locked.ps1`; results `zero/out/*.json`, `zero/ZERO-RESULTS.md`, `zero/ZERO-TABLES.md`.
- Intermediate data (160-expert sample 3.0 GB, counts, encoded stream sets, record bins) live in a temporary folder (`<TMP>\data\`) and are ephemeral; the steps above regenerate them.

## 5. Not tested / open

- **End-to-end logits memcmp per step** (canonical zero on V4.1 REAP with the engine): not run. Al commit 88ddb07 (8/11) il motore V4.1 girava gli expert solo su CPU (il tier GPU rifiutava `fp8_block != 128`). Da cfb7ea0 i tier GPU eseguono V4.1 (fp8_block 32), quindi il memcmp end-to-end dei logits ora si può fare. The CPU FP4 path already decodes code 8 as +0; a canonical container would be ~130 GB; the V4.1 GPU workflow needs the GPU and heavy.lock. Do it after G2; the per-kernel bit-test above is the evidence until then.
- **MTP (DSpark) experts:** not measured (own model needed; only relevant if the engine uses DSpark).
- **Model trained on all 7680 records:** every g0 / rule-g / ceil-form size here uses models trained on 80 sampled experts; the production model is trained on 7680. The tail numbers (12 largest records) depend on how many median-121..123 matrices the training set holds.
- **Corruption audit with the hash** (team-2 cases A-J on ceil-form records with the 23-bit initial-state hash) and the TAIL+DIGEST alternative: not run.
- **Rule g / ceil+g on the full population:** measured on 40 held-out + 12 tail records, not on all 7680; the max record of a full container is not measured.
- **Vulkan paths and other drivers** for canonical zero: not tested (CUDA only).

## 6. Notes

- Anchor assertion (task 2): for every record the stored bytes 24..26 were checked equal to the anchor the model was trained with (120 for B, true lower medians for A, densest-pair anchors for G). Source check: decoders take the anchor from the record (`wex_neural_rans.h:234`, `wex_cuda.cuh:412/624`); `wex_nr2_median` is only called by the trainer and the encoder (and tests). Anchor-mismatch control (model trained at 120, byte 121): **+8.33% of raw**.
- Task 1 detail: p99.9 17,653,760 B; fixed slots at p99 give +7.72% capacity with 66 records (0.86%) overflowing; records with all three matrix medians != 120 average 0.9288 of raw (max 0.9745) vs 0.8977 (max 0.9126) with none shifted. The spec estimate max/mean 1.0113 (40 experts) is replaced by 1.0832 over all 7680.
- The main session's phase-3 run (independent rule-g re-check + ceilbench rerun) started at 02:03 after the usage-limit reset and was stopped at 02:04 at the parent session's request, before ceilbench, so it overwrote nothing: `ceil-gpu/bench_5090.*` are still the 01:28 files.
