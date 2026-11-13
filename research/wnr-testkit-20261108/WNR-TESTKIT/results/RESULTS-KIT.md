# Risultati del kit WNR su RTX 5090 (KAIRA, 2026-11-08)

## Riassunto

- **Esito:** il kit gira per intero su questa macchina (Ryzen 9 7950X3D, 32 GB, RTX 5090 sm_120, driver 621.01). Ogni numero che il kit confronta con il PC di ricerca (i7-7700K + RTX 3070) è **identico**: dimensioni dei record, modelli, sha256 dei record, conteggi dell'audit, conteggi del bit test dello zero canonico.
- **Un solo FAIL, `audit`, dovuto a un bug del codice di test e non al formato:**
  - Il caso M1 va in crash (`TypeError: 'NoneType' object is not subscriptable`) quando il bit flippato cade nei byte di padding della tabella LENC. In quel caso le lunghezze restano identiche e il record viene accettato con output identico.
  - Ho rilanciato M1 da solo con un wrapper esterno che non tocca il kit: 4000 prove per regola, **0 accettazioni silenziose con tutte le regole**. I 5 casi senza danno sono tutti flip nel byte `ltb-3` (padding di allineamento a 4 B).
  - Tutti gli altri gate dell'audit passano: HASH, HASH32 e TAILD hanno 0 accettazioni silenziose su 1.319.848 prove.
- **Altro difetto del kit, solo cosmetico:** `run_all.ps1` segna i tre runner GPU come "FAIL no RESULT line". Questi stampano la riga RESULT preceduta da un timestamp (`[ 48.9 s] RESULT ...`), mentre lo script la cerca a inizio riga. I loro `summary.json` dicono PASS.
- **Ceil-form sulla GPU:** byte-exact su tutte le varianti e su tutte le configurazioni di lancio.
  - Alla stessa configurazione della sessione precedente (nt128/lb, m24) il confronto accoppiato dà A259/CF5 = 1,017 (CF5 più veloce dell'1,7%). Il costo proprio dell'inversa è CF5/CF0 = 1,006 su m24 e 1,004 su 8 stream. Coerente con 752,8 vs 757,9 µs (0,993) del 7-8/11.
  - Sulla mediana delle 8 configurazioni l'inversa costa +2,4% su m24 e +1,9% su 8 stream: appena sopra il target T9 di +2% su m24. Sulla 3070 costava +0,3%.
  - **Anomalia m3:** su m3 (1 record) le varianti CF risultano circa 23% *più veloci* della propria base (CF5/CF0 = 0,77, sulla 3070 1,007). Lo stesso kernel CF5 su m3 fa 177 µs in gpu_bench e 304 µs in gpu_v1_profile. È quindi un artefatto di misura della taglia m3 sulla 5090, non un costo: i numeri m3 non vanno usati.
- **Zero canonico:** 0 bit diversi su 227.082.240 output (160 expert, 10 kernel colibri, righe reali e sintetiche), esattamente come sulla 3070. Anche il GEMM tensor-core FP16 (1.310.720 output) e la MMA FP4 sperimentale sm_120a (2.621.440 output) danno 0 differenze. Conferma il risultato della sessione precedente (0 differenze su 83.841.024 output sui percorsi di default (11 fasi; il percorso masked dg_moe<256> è escluso perché non deterministico)).
- **Regola g:** l'ancora fissa 120 e la coppia più densa danno lo stesso modello (id 0x750c0967, uguale al Model G/B della sessione precedente) e -0,194% del raw sui 40 held-out. Il record massimo dei 40 scende da 17.563.648 B (WNR2) a 17.055.744 B. Il kit non contiene i 12 record di coda né il container completo: il massimo assoluto di 18.321.408 B non è misurabile qui.
- **Tempi:** il run completo ha preso 21,5 minuti invece delle "alcune ore" stimate. L'audit, stimato 1-2 h, ha preso 9,3 minuti.
- **Non eseguito:**
  - i workflow di ricerca dei team 5/6/7, che aspettano il tuo OK uno per uno;
  - i percorsi GPU di KAIRA della checklist manuale.
- **`gpu_v1_profile` (sperimentale), eseguito dopo il run principale:** pesi e scale esatti e H32 corretto. HASH respinge tutte le corruzioni che passano la struttura (0 accettate su 1.382.400). Il profilo v1 costa circa 0-1% su m24 e 8 stream.

## Macchina e ambiente

| | |
|---|---|
| CPU | AMD Ryzen 9 7950X3D, 16 core / 32 thread, AVX2 sì, AVX-512F sì (IsProcessorFeaturePresent; `--cpuid` non eseguito) |
| RAM | 31,2 GB visibili, 11,7-12,1 GB liberi all'inizio |
| GPU | NVIDIA GeForce RTX 5090, cc 12.0, 170 SM, 32.607 MiB, driver 621.01 (CUDA 13.5) |
| OS | Windows 11 Pro 10.0.26300 |
| Python | 3.14.8 (riferimento: 3.14.7), venv `venv\` nella radice del kit |
| Pacchetti | numpy 2.5.4, cupy-cuda12x 14.2.0, cuda-pathfinder 1.8.6, cuda-toolkit 12.9.2.0, NVRTC 12.9.86, runtime 12.9.79: i pin esatti di `setup\requirements-gpu.txt`, installati senza errori |
| Compilatori | nvcc 13.4 presente; MSVC installato ma `cl` non nel PATH (nessun test del kit lo richiede) |
| Dati | 160 + 76 expert (4.436.951.040 B) estratti da `U:\models\DeepSeek-V4.1-Flash-1017-REAP192-MXFP4` con `fetch_data.py --local-shards`, sha256 ok 236/236. Servono 46 shard su 48 (00047 e 00048 non servono, nemmeno per i 6 MTP). Nessun download. |

## Come è stato eseguito

- **Lock:** ogni passaggio pesante è passato da `heavy.lock` (`lab\v41-gpu-20261107\heavy.lock`) con un wrapper bash. Il wrapper crea il file in modo atomico, controlla ogni 60 s e lo rimuove solo se contiene ancora la propria etichetta.

  | Passaggio | Lock | Attesa |
  |---|---|---|
  | Estrazione dati + quick | 02:36-02:42 | 8 min |
  | Run completo | 02:52-03:14 | 10 min |
  | Diagnosi M1 | 03:16-03:23 | - |
  | gpu_v1_profile | 03:46-03:49 | 23 min |

  Il lock era tenuto dal reviewer G2 / lane G del workflow V4.1.
- **Cache CuPy:** `CUPY_CACHE_DIR` puntava a `results\_cupy_cache`, così nessuna cache finisce in `%USERPROFILE%\.cupy`.
- **Scritture fuori dal kit:** controllate prima di lanciare. Gli script scrivono solo in `results\`, `data\` e `venv\`. Le eccezioni sono la cache di pip (`%LOCALAPPDATA%\pip`) e `pack_results.ps1`, che crea e poi cancella una cartella temporanea in `%TEMP%`. Nessuno script ha scritto su U:, `app\` o `kaira.json`.
- **Quick conservato:** i risultati del quick (prima del completo, che sovrascrive `results\`) sono in `results_quick_20261108\`.

## Esito per test (run completo)

| Test | Esito | Tempo | Note |
|---|---|---|---|
| check_env | PASS | 4 s | |
| verify_data | PASS | 2 s | 236/236 sha256 |
| heldout_configs | PASS | 166 s | 480/480 record esatti; tutto = riferimento |
| wnr2_g0 | PASS | 60 s | 120/120 record con sha256 = team 5 |
| chunk_sweep | PASS | 98 s | 240/240 byte-exact |
| train_curve_mtp | PASS | 94 s | tutti i check, anche quelli soft, = team 5 |
| audit | **FAIL** (bug del kit) | 558 s | crash nel caso M1; tutti gli altri gate PASS; M1 rilanciato: 0 silenziose |
| nvrtc_check | PASS (run_all scrive "FAIL no RESULT line") | 133 s | 26/26 compilazioni |
| zero_bittest | PASS (idem) | 105 s | 0 bit diversi |
| gpu_bench | PASS (idem) | 69 s | byte-exact, controlli negativi ok |
| gpu_v1_profile | DONE (sperimentale, non dà mai PASS) | 166 s | pesi e scale esatti, H32 ok; eseguito a parte (03:46-03:49) |

Righe decisive:
- `RESULT audit FAIL failed: cases_without_error; R0 A 588/138240 K 13/86400 ...; HASH A 0/138240 K 0/86400 all-cases 0/1319848; HASH32 ... 0/1319848; TAIL A 1174/138240 ...; TAILD ... 0/1319848`
- `[fg0d] case M1 ERROR TypeError: 'NoneType' object is not subscriptable` (uguale per fg0de, fg0dh, fg0dt)
- `M1DIAG none-damage trials: 5; all lengths identical: True; all n_aff_streams 0: True; byte_rel/ltb: [('fg0d', [17365], 17368), ('fg0de', [17353], 17356), ...]`

Quick pass (02:36-02:42): tutti PASS; stesso artefatto "no RESULT line" sui 3 runner GPU. Su 100 prove M1 non era stato colpito nessun byte di padding.

## Numeri contro i riferimenti

### heldout_configs (40 held-out, modelli addestrati qui sugli 80 pari)

I modelli addestrati qui sono identici a ceil-ref `even.npz` (wf, scf, etab per wnr2/f/fg0/fg0c). Tutte le 480 dimensioni dei record coincidono con `ref/heldout_ref.json`.

| config | media totale qui (B) | riferimento (B) | Δ vs WNR2 (% raw) qui | Δ riferimento | rapporto sul raw |
|---|---|---|---|---|---|
| wnr2 | 16.909.926,4 | 16.909.926,4 | 0 | 0 | 0,89943 |
| f | 16.578.764,8 | 16.578.764,8 | -1,761 (IC -1,945..-1,632) | -1,761 (-1,948..-1,632) | 0,88182 |
| fg0 | 16.573.132,8 | = | -1,791 | -1,791 | 0,88152 |
| fg0d | 16.546.611,2 | = | -1,932 | -1,932 | 0,88011 |
| fg0de | 16.548.556,8 | = | -1,922 | -1,922 | 0,88021 |
| fg0dt | 16.506.880,0 | = | -2,144 | -2,144 | 0,87800 |
| fg0c | 16.057.856,0 | = | -4,532 | -4,532 | 0,85411 |
| fg0cde | 16.033.587,2 | = | -4,661 | -4,661 | 0,85282 |
| fg0e / fg0t / fg0cd / fg0ce | 16.574.054,4 / 16.532.684,8 / 16.032.153,6 / 16.059.187,2 | = | -1,786 / -2,007 / -4,669 / -4,525 | = | |

Record massimo dei 40 (byte, rapporto): wnr2 17.563.648 (L39_E156, 0,9342); f 16.957.440 (L24_E058); fg0 16.756.736; fg0d 16.732.160; fg0dt 16.691.200; fg0cde 16.224.256 (0,8630).

### wnr2_g0 (WNR2 semplice, regola dell'ancora)

- **Modelli:** med 0xfc992a63, c120 0x750c0967, pair 0x750c0967, identici al team 5 e, per med, a ceil-ref WNR2.
- **Decodifica:** i due decoder WNR2 non modificati decodificano byte-exact 40/40 per ogni regola. Gli header portano le ancore giuste.
- **R1** (il decoder ricava la mediana): 3 rifiutati, 0 errati in silenzio. Gli sha256 dei 120 record sono identici a quelli del team 5.

| regola | media qui | riferimento | Δ vs med |
|---|---|---|---|
| med | 16.909.926,4 | 16.909.926,4 | |
| c120 | 16.873.369,6 | 16.873.369,6 | -0,194% raw (riferimento -0,194%, IC -0,410..-0,026); 6 matrici in 3 expert con ancora diversa dalla mediana |
| pair | 16.873.369,6 | = c120 su questo campione | -0,194% |

Il record massimo dei 40 passa da 17.563.648 B (med) a 17.055.744 B (c120 = pair, L24_E058, 0,9072).

### chunk_sweep

240 record su 240 decodificati byte-exact. Le righe 2048:512 sono identiche a ceil-ref held-out (160 su 160) e le righe parziali del team 6 coincidono (121 su 121).

Totale rispetto alla stessa config a 2048:512, su 40 expert:

| config | Δ |
|---|---|
| fg0de @1024:256 | +0,418% |
| fg0de @4096:1024 | -0,216% |
| fg0dt @4096:1024 | -0,103% |
| fg0de @8192:2048 | -0,325% |

### train_curve_mtp (con gli expert extra)

- **Dati:** sha256 236/236, nessun leakage (40 test, 190 nel pool, 6 MTP), ceil round-trip ok su tutte le matrici, 66 record reali byte-exact.
- **Check soft, tutti uguali al team 5:** curva even80, 25 estrazioni su 25, aggregati (differenza massima 0 B), stime MTP (20 confronti, differenza 0), record reali di spot (55 su 55).
- **Curva:** fg0 - wnr2 a N = 190 = **-1,795% del raw** (riferimento -1,795).
- **MTP (stime, rapporto medio):**

  | Opzione | Qui |
  |---|---|
  | MTP-own LOO, ancora 120 | **0,8694** |
  | MTP-own LOO, moda costante | 0,8724 |
  | MTP-own LOO, coppia più densa | 0,8733 |

  Tutti uguali al riferimento (0,8694 / 0,8724 / 0,8733). Anche le stime routed coincidono con il riferimento (differenza massima 0 su 20 confronti):

  | Opzione routed | Rapporto | Rapporto salvato | Expert salvati raw |
  |---|---|---|---|
  | WNR2 | 1,1062 | 0,9998 | 5 su 6 |
  | (i) ceil + g0 120 | 1,0927 | 1,0000 | tutti e 6 |
  | (ii) coppia più densa per matrice | 1,0520 | 0,9854 | 3 su 6 |

  Il modello routed è quindi inutile per gli MTP: serve un modello proprio.

### audit (T3, 8 expert held-out, 5 regole)

- **Record:** gli sha256 dei 40 record puliti sono identici al riferimento (40/40). Self test H32 0xB0F57EE3 ok. Decodifica dei record puliti byte-exact con entrambi i decoder. Controlli W0/S0 tutti accettati con output identico. Il cross-check `validate` concorda. Obiettivo di flip casuali raggiunto (46.080 per matrice per A, 28.800 per K).
- **Accettazioni silenziose** per caso (righe "all"); in tutto 1.319.848 prove per regola:

| caso | R0 | HASH | HASH32 | TAIL | TAILD |
|---|---|---|---|---|---|
| W0 / S0 (controlli) | 0 / 0 | 0 | 0 | 0 | 0 |
| A flip casuale (138.240) | **588** (161/227/200) | 0 | 0 | **1174** (376/417/381) | 0 |
| B flip nell'ultimo byte (72.000) | 0 | 0 | 0 | 68.901 (95,7%) | 0 |
| C flip nello stato iniziale (72.000) | 142 | 0 | 0 | 215 | 0 |
| D / E / H / J / N3 | 0 | 0 | 0 | 218 / 272 / 67 / 64 / 59 | 0 |
| F / G / M2 / M3 / N1 / N2 / N4 / N5 | 0 | 0 | 0 | 0 | 0 |
| I (contesti identici, identità diversa) | 72.000 (100%) | 0 | 0 | 72.000 | 0 |
| T (coda ricodificata) | 72.000 (per costruzione) | 0 | 0 | 72.000 | 0 |
| K flip nelle scale (86.400) | 13 | 0 | 0 | 16 | 0 |
| L flip nell'ultimo byte delle scale (34.560) | 0 | 0 | 0 | 40 | 0 |
| O record nello slot di un altro expert (16) | 16 | 0 | 0 | 16 | 0 |
| P1 / P2 / P3 | 0 / 0 / 0 | 0 | 0 | 0 / 0 / 7 su 32 | 0 |
| M1 (rilancio diagnostico, 4000) | 0 | 0 | 0 | 0 | 0 |

- **Confronto con i riferimenti** (`reference/team5_partial_cases.json` e README):

  | Riferimento | Qui |
  |---|---|
  | A con R0 588/138.240 (161/227/200), errore massimo 12 unità E2M1, nessun blockflip | identico |
  | A con TAIL 1174 (376/417/381), 330 con blockflip, errore massimo 18 unità | identico |
  | B con R0 e HASH 0/72.000 | identico |
  | C con R0 142/72.000 (solo log) | identico |
  | Smoke del team 5 su B con TAIL: 232/240 | 95,7% su 72.000 |
- **Accettazioni per caso attese:** n x 2^-23 = 0,157 (HASH) e 0,0003 (TAILD); osservate 0.
- **Danno con R0 e TAIL non limitato:** errore massimo silenzioso su tutti i casi 1,36e39 unità (R0) e 2,04e39 (TAIL), per i casi I/O/T (expert o slot sbagliato).

### nvrtc_check

26 compilazioni su 26, 0 errori (86, 120, compute e sm, più sm_120a per mxf4). NVRTC supporta fino a sm_121. Nessuno spill nei kernel bench e zero. I kernel sperimentali v1 `sdec3<256,...>` hanno spill di 4 B su sm_120.

### zero_bittest (160 expert)

- **10 kernel reali:** 0 bit diversi. Ogni kernel confronta 14.008.320 output (56.033.280 per i due kernel a 4 token): stessi conteggi e stessi zero in output (3.295.547 / 14.556.082) della 3070.
- **Righe sintetiche a zero:** 0 differenze. In totale 227.082.240 output confrontati, 0 differenze.
- **Controlli:** il positivo scatta in tutti i kernel (1, o 4 per i kernel a 4 token, come sulla 3070); il negativo scatta (10.240 differenze di solo segno); le riesecuzioni sono deterministiche e tutti gli output sono scritti.
- **GEMM tensor-core FP16:** 1.310.720 output, 0 bit diversi, controllo positivo 8.
- **mma f16/bf16:** 0 D diversi su 20.000 tile; A tutta a zero dà D = +0 in 2.560.000 su 2.560.000.
- **MMA FP4 sperimentale sm_120a:** 2.621.440 output, 0 bit diversi, 0 NaN.
- **Avviso innocuo:** `RuntimeWarning: overflow encountered in ldexp` in `run_zero_bittest.py:90`. Viene dal codice di scala E8M0 255 che diventa inf.

### gpu_bench (8 expert, 41 round, nt 32/64/128/256 x lb/out4, sm_120)

- **Correttezza:**
  - Sha256 degli input ok.
  - Encoder identico al team 4 per ogni array e dimensione.
  - Round-trip ceil su CPU ok.
  - Scale byte-exact.
  - Tutte le varianti byte-exact contro i byte originali in ogni configurazione e in ogni modo di lancio: singolo, 8 stream, parziale, pipeline.
  - I controlli negativi CF0 / CFD0 / CFDB20 non sono byte-exact, come previsto.
- **Dimensioni (8 record):** A259 0,90227 del raw, CF 0,88155, CFD 0,88156 (= riferimento), DB2 0,90548, CFDB2 0,88477.
- **Trasformazione ceil:** 8.847.360 blocchi, 29,93% pari, 0 sotto 6, 0 overflow di scala.
- **Clock GPU:** 1980 MHz all'inizio, 3037 MHz alla fine (93% di utilizzo). Le mediane assolute sono rumorose (A259 m24 a nt128/lb: 949,9 µs, ma il rapporto accoppiato con CF5 è 1,017). Per decidere contano i rapporti accoppiati.

Miglior configurazione per variante (µs; 3070 da `ref/bench_3070/best_final.txt`):

| variante | m1 | m3 | m24 | 8 stream | 3070 m24 | 3070/qui m24 |
|---|---|---|---|---|---|---|
| A259 | 169,0 | 261,3 | 802,8 (nt256/lb) | 786,4 | 4171,8 | 5,20 |
| CF5 | 167,1 | 177,3 | 735,3 (nt128/lb) | 789,0 | 4185,1 | 5,69 |
| D | 162,9 | 238,8 | 651,3 | 708,6 | 3835,9 | 5,89 |
| CFD5 | 163,0 | 173,2 | 650,3 | 737,5 | 3848,2 | 5,92 |
| DB2 | 163,0 | 244,9 | 569,4 | 643,6 | 3230,7 | 5,67 |
| CFDB2 | 167,1 | 179,4 | 599,2 | 659,1 | 3307,5 | 5,52 |

Rapporti accoppiati nella stessa configurazione (mediana sui round; min / mediana / max sulle 8 configurazioni):

| coppia | m1 | m3 | m24 | 8 stream | 3070 m24 (mediana) |
|---|---|---|---|---|---|
| CF5/CF0 (costo dell'inversa) | 1,043 | **0,766** | 0,993 / **1,024** / 1,048 | 0,949 / 1,019 / 1,036 | 1,003 |
| CFD5/CFD0 | 1,038 | 0,788 | 1,019 | 1,012 | 1,000 |
| CFDB2/CFDB20 | 0,999 | 0,723 | 1,023 | 1,006 | 1,003 |
| CF5/A259 | 1,042 | 0,768 | 1,022 | 1,023 | 0,988 |
| CF0/A259 | 1,000 | 1,000 | 0,999 | 0,998 | 0,998 |

Alla configurazione nt128/lb della sessione precedente:

| Misura | Qui | 7-8/11 |
|---|---|---|
| A259/CF5 m24 | 1,0168 | |
| A259/CF5 8 stream | 0,9996 | |
| CF5/CF0 m24 ("own") | 1,006 | |
| CF5/CF0 8 stream | 1,004 | |
| CF5 m24 | 735,3 µs | 752,8 µs |
| A259 m24, stimato da CF5 x 1,0168 | circa 748 µs | 757,9 µs |
| DB2 m24 | 581,6 µs | 591,0 µs |
| CFDB2 m24 | 610,3 µs | 611,5 µs |

- **Target T9:**
  - Costo proprio dell'inversa ≤ +2%: rispettato alla configurazione migliore (+0,6%), di poco mancato sulla mediana delle configurazioni a m24 (+2,4%).
  - WNR3 ≤ WNR2 + 5%: rispettato in tutte le configurazioni.
- **Anomalia m3:** con un solo record (3 matrici) le varianti CF vanno quanto m1 (CF5 m3 177 µs contro m1 167 µs), mentre A259/CF0/D salgono a 239-261 µs. Sulla 3070 i rapporti m3 erano circa 1,0. Non l'ho indagata: il README dice di non toccare i kernel. Va capita prima di citare numeri m3.

### gpu_v1_profile (SPERIMENTALE, eseguito dopo il run principale)

Lock dalle 03:46 alle 03:49 (in attesa dalle 03:23 per lane G). Run completo: 41 round, nt 32-256 x lb/out4, sm_120, 166 s.

- **Esito:** `RESULT gpu_v1_profile DONE EXPERIMENTAL untested: weight_all_exact True, scales exact True, h32 True`. È DONE per costruzione: questo runner non dà mai PASS.
- **Correttezza del profilo v1** (CEIL|LENC|HASH decodificato nel kernel): tutti i pesi e le scale sono esatti. Il vettore H32 è corretto sulla GPU. I byte originali degli 8 expert e le scale codificate sono uguali a quelli dell'encoder del team 4.
- **Corruzione sulla GPU** (1.382.400 flip casuali per variante):

  | Variante | Accettati con output sbagliato | Note |
  |---|---|---|
  | G5, senza HASH | 6249 (0,452%) | 6245 differiscono solo in un nibble di segno; coerente con R0 dell'audit CPU, 0,425% |
  | G5H | 0 | 6111 passano la struttura e li respinge l'hash |
  | G5HL | 0 | 5578 passano la struttura e li respinge l'hash |

  - **Flip nell'ultimo byte:** 0 accettati in tutte le varianti.
  - **Scale:** SG accetta 74 su 172.800; SH e SHL 0.
  - **Corruzione della sezione LENC** (6400 prove per variante): 0 accettati con output sbagliato; 2 accettati con output identico.
- **Costi accoppiati** (mediana sulle 8 configurazioni; tra parentesi la 3070, versione di codice precedente):

  | Costo | m24 | 8 stream |
  |---|---|---|
  | HASH | 1,005 (1,007) | 1,008 (1,011) |
  | LENC | 1,011 (1,000) | 1,005 (0,997) |
  | HASH+LENC | 1,010 (1,001) | 0,996 (1,001) |
  | v1 vs kernel CF5 del team 4, stesso modello | 0,999 (v1 vs CF5: 1,007) | 1,002 (1,006) |

  Su m24 e 8 stream il profilo v1 costa quindi circa 0-1% sulla 5090. Controllo A/A: 0,999 / 1,002.
- **Anomalia m3 di nuovo presente, ma su varianti diverse:** qui G5, G5L e CF5g su m3 vanno quanto m1 (circa 222 µs), mentre CF5 sale a 304 µs. Nel gpu_bench lo stesso CF5 su m3 faceva 177 µs. Che lo stesso kernel dia valori m3 così diversi in due bench indica un artefatto di misura della taglia m3 sulla 5090, non un costo reale. 540 blocchi a nt32 stanno in un'onda sola sui 170 SM. Per questo HASH su m3 risulta 1,338. **Non usare i numeri m3.**
- **Spill:** `nvrtc_check` segnala 4 B di spill nei kernel `sdec3<256,...>` (scale, nt=256) su sm_120. Nei kernel dei pesi v1 non ci sono spill (l0 in tutte le configurazioni).

## Confronto con `lab\wnr3-5090-20261107\RESULTS.md`

| misura | RESULTS.md (7-8/11) | kit, oggi | |
|---|---|---|---|
| ceil-form, bench m24 | CF5 752,8 vs A259 757,9 µs (0,993) | nt128/lb: CF5 735,3 µs, A259/CF5 accoppiato 1,017; mediana delle configurazioni CF5/A259 1,022 | coerente alla stessa configurazione |
| byte-exact ceil-form GPU | 138.240/138.240 stream | tutte le varianti, tutte le configurazioni e i modi di lancio | ok |
| DB2 / CFDB2 m24 | 591,0 / 611,5 µs | 581,6 / 610,3 µs (nt128/lb) | ok |
| zero canonico | 0 differenze su 83.841.024 output sui percorsi di default (11 fasi; il percorso masked dg_moe<256> è escluso perché non deterministico) | 0 su 227.082.240 output dei kernel colibri + 1,31 M (FP16 TC) + 2,62 M (MMA FP4 sm_120a) | ok |
| regola g vs ancora fissa 120 sui 40 held-out | -0,1944% raw, Model G = Model B (id 0x750c0967) | c120 = pair = -0,194%, id 0x750c0967, IC -0,410..-0,026 | identico |
| WNR2 med (Model A) | 16.909.926,4 B (0,89943), id 0xfc992a63 | identico | identico |
| ceil + mediana (CF), 40 held-out | 16.578.764,8 B (-1,761%) | f = 16.578.764,8 B (-1,761%) | identico |
| ceil + g (CFG) | 16.578.252,8 B (C KAIRA, regola g) | il kit misura fg0 = ceil + ancora fissa 120: 16.573.132,8 B (-1,791%) | non la stessa config |
| record massimo | 18.321.408 B (container di produzione, 7680 record); coda: G 18.464.768, CFG 17.985.536 | solo i 40 held-out: WNR2 17.563.648, c120/pair 17.055.744, f 16.957.440, fg0d 16.732.160 | il kit non ha la coda né il container |

## Difetti del kit trovati (non corretti: il kit è rimasto com'è)

1. **`tests\audit\cases.py:600-602` (`run_rcase`).** Se `decode_r` accetta un record senza stream toccati (`n_aff_streams == 0`), restituisce `damage=None` e l'espressione `x['damage']['coded_sym_diff']` va in crash. Succede con M1 quando il flip cade nel padding della tabella LENC (`pack_groups` riempie fino al byte e poi a 4 B). Correzioni possibili:
   - trattare `damage is None` come danno zero (output identico, accettazione non silenziosa);
   - oppure limitare i flip di M1 ai bit usati della tabella.

   Con la prima correzione, applicata dal wrapper `m1_diag.py` senza toccare il kit: M1 4000/4000 per regola, 0 silenziose, 1 accettata con output identico.
2. **`setup\run_all.ps1`.** Cerca `'^RESULT '`, ma i runner GPU stampano `[   x s] RESULT ...`, quindi run_all scrive "FAIL no RESULT line (exit 0)". `collect_results` legge i summary e conta bene.
3. **`tests\gpu\run_zero_bittest.py:90`.** Avviso di overflow in `np.ldexp` (E8M0 255), solo cosmetico.
4. **`tests\gpu\gpu_bench.log`.** Il log viene accodato tra quick e completo, quindi contiene entrambe le esecuzioni.
5. **Stime di durata.** Su questa macchina il kit è 5-10 volte più veloce del previsto (il completo 21,5 min invece di ore).

## Cosa non è stato eseguito e perché

- **Workflow di ricerca dei team 5/6/7** (`workflows\`): sono lunghi e multi-agente. Per tua richiesta partono solo uno alla volta, dopo il tuo OK su portata e dimensioni.
- **Checklist dei percorsi GPU di KAIRA** (`tests\gpu\CHECKLIST-kaira-gpu-paths.md`: DeepGEMM sm120 FP4, cuBLASLt MXFP8, Vulkan, memcmp end-to-end dei logit): è manuale e richiede il build di KAIRA, quindi non fa parte di questo compito. DeepGEMM e cuBLASLt sono già coperti dal bit test della sessione del 7-8/11; Vulkan e i logit end-to-end restano aperti.
- **`check_env --cpuid`:** non eseguito, perché lancia codice macchina da una pagina W^X. Il default con IsProcessorFeaturePresent basta.
- **Coda dei 12 record più grandi e container completo:** non sono nel kit.

## File

- `results\`: run completo (`ALL_RESULTS.json`, `_runlogs\`, una cartella per test).
- `results_quick_20261108\`: copia del quick pass, più `quick_run.log`.
- `results\audit_M1_diag\`: rilancio diagnostico di M1. Contiene `summary.json` (DONE), `audit.log`, `m1_none_damage_trials.json` con i 5 casi accettati con output identico, il wrapper `m1_diag.py` e il suo log. È incluso nel pacchetto (nel repo Wex-WRN2 manca `case_M1.json`, 4,5 MB: i totali sono in `summary.json`).
- `results\gpu_v1_profile\`: profilo v1 (la cache `v1enc` è esclusa dal pacchetto).
- `results_KAY_<data>.zip`: pacchetto da `setup\pack_results.ps1`. È circa 6,6 MB, contiene anche questo file (`results\RESULTS-KIT.md`) ed esclude le cache rigenerabili `_cache`, `gpu_bench\enc` (961 MB) e `gpu_v1_profile\v1enc` (557 MB). La cartella temporanea del pack era dentro il kit (`_packtmp`, rimossa).
- `data\`: 236 expert (4,44 GB), non inclusi nel pacchetto.
- Log dei lanci sotto lock: `results\_runlogs\full_run.log`, `results\_runlogs\v1_run.log`, `results_quick_20261108\quick_run.log`.
