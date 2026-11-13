# A/B sullo stesso drive: WEX1 contro WNR2 (V4.1 REAP-192, GPU, U:), 2026-11-08 20:33-21:44

> **Superato da W3** (2026-11-09, 16 GB, build integrata, `research/v41-w3-20261109/results.md`): WNR2 -1,80% ± 0,65% rispetto a WEX1. Il +2,0% qui veniva per circa 2/3 dalla head letta da U: a `--memory-gb 13`. Il verdetto "WEX1 rimane il formato di produzione" è superato anche lui: dal 9/11 KAIRA usa WNR2 col mirror U:+C:, per scelta dell'utente.

## Riassunto (italiano)

- **Verdetto: WNR2 resta un formato di capacità, non di velocità.** WNR2 è più veloce di WEX1 in modo misurabile, **+2,0% tok/s in decode** (IC 95% da +1,2% a +2,8%, 8 coppie). L'intervallo esclude lo 0, ma resta sotto la soglia del +3% del PLAN (3.2). WEX1 rimane il formato di produzione per la velocità. WNR2 occupa il 10% di disco in meno (129,9 contro 144,4 GB).
- **Token identici in tutte le esecuzioni.** I dump dei logits delle 4 esecuzioni ABBA hanno lo stesso sha256 (`4300486e...`), e lo stesso vale per i due smoke (`c6e4044f...`). Quindi i logits sono identici bit per bit, oltre ai token greedy, su tutti i 5 prompt.
- **WNR2 accorcia il TTFT del 4-7%** (media delle due esecuzioni). Su ctx2k (2.276 token) passa da 524 a 501 s, cioè -4,4%. Sui prompt brevi il guadagno va dal -5% al -7%.
- **Da dove viene il guadagno** (~18 ms/token in media):
  - **circa 2/3 (~11 ms/token) vengono dalla head.** A 13 GB la head bf16 viene letta da U: a ogni token, e con WNR2 trova meno contesa sullo stesso drive: 313 contro 325 ms/token;
  - il resto (~7 ms/token) è MoE più attenzione:
    - il calcolo MoE sulla GPU scende di 16-20 ms/token, perché gli expert caricati direttamente in VRAM non vanno caricati da RAM;
    - l'attesa dei pesi sale invece di 8-12 ms/token, perché la decodifica rANS sta nel percorso di caricamento;
    - l'attenzione sale di ~7 ms/token;
  - dal disco si leggono il 5% di byte in meno (1,08 contro 1,14 GB per forward) con un 5% di letture in più, perché i record sono più piccoli.
- **RAM: `--memory-gb 13` in tutte le esecuzioni.** Alla presa del lock erano disponibili 13,5 GB (< 15). Con 13 GB la head bf16 non sta in RAM e viene letta dal disco a ogni token: ~0,32 s/token, il 40% del tempo di decode in entrambi i bracci. Per questo i tok/s assoluti sono bassi: 1,15 contro i ~2,0 di G2 a 16 GB.
  - Stima [I]: a 16 GB la head è residente e la quota di guadagno che viene dalla head sparisce. Restano ~7 ms su ~0,5 s/token, cioè circa +1,5%, ancora più lontano dalla soglia del 3%.
- Le due esecuzioni dello stesso braccio differiscono di meno dello 0,5%: misura ripetibile, macchina quieta.

## Setup

- **Build (G2 review 3, la stessa di `g2/locked_g2r3.out`).** Nessuna build e nessuna modifica ai sorgenti. Hash in `hashes.txt`:

| file | sha256 |
|---|---|
| `lab\v41-port-20261106\ws-g\g2bin\deepseek_v4-fix3.exe` | `6493e16edd7b8482bf770e39e481b1ed6edb0fb7647690be8e40e0d25948c9d1` |
| `ws-g\new-out3\coli_cuda_dsv4_dg.dll` (caricata, `COLI_DSV4_DLL`) | `3182bcd1cac78e9b1aeb27f29b19a2e5390427fd0ee90cf42fb9812b18f9e8a6` |
| `ws-g\new-out3\coli_cuda_dsv4.dll` (non caricata) | `99cb288f54550ffe5c1e0367e100fe56d2db3c01683d01c289ba1a702c9a59ab` |
| `lab\v41-gpu-20261107\w1\build\kaira_wex_cuda.dll` (helper) | `cceb4fb573b3b3e7b71843bf85a2b87cf9efdd94155bfc5deec717cce949e782` |
| `ws-g\g2bin\cudart64_13.dll` | `d26fd1c2629594759af569a5252f94e4568d4ebeca67dc323834deb141aa278a` |
| `ws-g\g2bin\cublasLt64_13.dll` | `cad63434448e7141629e240ea093ad596a7ef6a0f67b468ba9bc1df6e1eeee33` |

- **Container (entrambi su U:, SK hynix, 6,58 GB/s):**
  - A = `U:\kaira_wex\v41-reap\v4_flash_experts.wex` (WEX1, "Container v1", 144.389.472.256 B);
  - B = `U:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex` (WNR2, "Container v3", 129.905.360.896 B, sha256 `3F703FA4...`).
  - Il log di ogni esecuzione mostra il container atteso (controllo automatico in `run_ab.sh`).
- **Env.** I bracci differiscono **solo** in `COLI_WEX_PATH`. Il resto (`env.txt`) è l'ambiente di `g2/run_g2.sh` (la macchina di decode di kaira.json), con `COLI_WEX_GPU_DIRECT=2` in entrambi:
  `COLI_V4_SAVE_USAGE=0 RAM_GB=13 CTX=4096 V4_SKIP_COLD=0 V4_MTP=0 DSV4_DECODE_PROF=1 V4_PREDICT_STATS=1 COLI_WEX_CUDA_LIB=<helper> DSV4_CUDA=1 COLI_DSV4_DLL=<dg dll> DSV4_CUDA_VRAM_RESERVE_MB=4096 DSV4_CUDA_EXPERT_MIRRORS=5500 V4_COPY_WORKER=1 V4_PROMOTE_VRAM=2 V4_ASYNC_ATTACH=1 DSV4_CUDA_STREAM_PRIORITY=1 V4_VRAM_FIRST=1 V4_GPU_HOT_CACHE=1 COLI_V4_TIER_PREWARM=1 V4_LOADER_LANES=8 COLI_DIRECT_OVERLAPPED=1 V4_NEXT_PREFETCH=1 V4_NEXT_PREFETCH_K=8 V4_NEXT_PREFETCH_DEPTH=2 V4_NEXT_PREFETCH_K2=4 V4_NEXT_PREFETCH_THREADS=8 V4_NEXT_PREFETCH_KEEP=1 V4_NEXT_PREFETCH_TOKSTART=1 V4_NEXT_PREFETCH_AVX2=1 V4_PREFETCH_NO_CREDIT=1 V4_ROUTER_AVX2=1 V4_NP_ONLINE=1 COLI_WEX_GPU_DIRECT=2`, `COLI_MODEL_DIRS=C:/models/DeepSeek-V4.1-Flash-1017-engram;E:/models/DeepSeek-V4.1-Flash-1017-engram`, modello `U:\models\DeepSeek-V4.1-Flash-1017-REAP192-MXFP4`. Campionamento greedy (`SUBMIT ... 0 1`, temperatura 0).
- **Differenze rispetto al G2 B (scelte per avere RAM uguale, letto dal codice di `ws-g\c\deepseek_v4.c`):**
  - `COLI_WEX_GPU_DIRECT=2` è attivo anche sul braccio WEX1. Lì non ha effetto: `coli_v4_expert_store_wex_entry` accetta solo record Neural-rANS, e il guard dei mirror di V4.1 è già sempre attivo. Il log di WEX1 mostra 0 letture dirette e pool=0.
  - **Nessun `COLI_WEX_RAM_CACHE_GB`.** Quella cache dei record compressi esiste solo per i record Neural-rANS (`v4_wex_crc`), quindi darebbe a WNR2 4 GB di RAM in più di WEX1. In più non ci starebbe: 12,7 GiB proiettati + 4 GB contro i 13,5 GB disponibili.
  - **RAM uguale, verificata.** Entrambi i bracci hanno `ram_tiers available=13.00GiB target_slots=7 target_cache=4.90GiB head=streamed-bf16 projected=12.74GiB`, TIERS ram=280 slot, e 982-1027 mirror in VRAM.
- **Regola RAM.** Disponibili 13,5 GB alle 20:33:54 → `--memory-gb 13`, deciso una volta sola e uguale in tutti i bracci. Dopo è salita a 16-18,6 GB (vedi la colonna "RAM disp."), ma il valore è rimasto fisso.
- **Macchina.** Nessun'altra app di calcolo sulla GPU (1,6-1,7 GiB usati e 0-6% di utilizzo all'avvio di ogni esecuzione).
  - Un'altra lane (G3, `locked_g3.ps1`) era in attesa di 18 GB **senza** il lock e non ha mai girato.
  - Piano energetico "Bilanciato": non l'ho cambiato (impostazione di sistema).
  - Esclusioni di Defender non verificate.
- **heavy.lock.** Preso alle 20:33:53 con il protocollo di `wnr3-5090-20261107\lock.sh`, etichetta "A/B same-drive WEX1 vs WNR2 (main session)". Rilasciato alle 21:44:39; il trap toglie solo un lock con la nostra etichetta.
- **Nessuna scrittura su U:.** I mtime della cartella del modello e dei container non sono cambiati (`.coli_usage` è ancora del 2026-11-07).

## Smoke (w0, 16 token)

| run | READY s | tok/s | TTFT s | container nel log | note |
|---|---|---|---|---|---|
| ab-smoke-A-wex1 | 16,8 | 1,305 | 4,53 | Container v1 active: U:\...\v4_flash_experts.wex | 4.719 letture host, 0 dirette |
| ab-smoke-B-wnr2 | 18,3 | 1,370 | 4,86 | Container v3 active: U:\...\v4_flash_experts.nr2.wex | 1.974 letture dirette → VRAM, 0 fallimenti |

Logits w0 identici bit per bit (sha256 `c6e4044f...`).

## ABBA: tabelle per esecuzione (128 token greedy, w0 escluso dalle statistiche)

Legenda delle colonne:
- `tok/s`: decode dal primo token (DONE STAT).
- `TTFT`: client, da SUBMIT al primo DATA.
- `non-VRAM/tok`: usi di expert instradati non serviti dalla VRAM al momento dell'attach, cioè 240 − `vram_hits` di `moeprof` (decode). Su WNR2 con GPU-direct un miss caricato disco→VRAM dalla lane conta come hit VRAM, quindi il miss si vede nel `wait`, non qui.
- `wait`: attesa dei pesi degli expert per token (`moeprof wait`, media sul decode). È la metrica "NVMe wait ms/token".
- `gpu`: `moeprof gpu`.
- `hit host`: hit rate dello store RAM del turno.

Le righe PROF/TIERS complete e le medie di `decprof`/`moeprof` per prompt sono in `summary.json`; i log grezzi sono in `runs\`.

**ab-A1-wex1** (20:35:32, RAM disp. 16,3 GB; READY 15,8 s; 176.749 letture, 3.323 GB, 60,8 letture e 1,14 GB per forward)

| prompt | tok/s | TTFT s | non-VRAM/tok | wait ms/tok | gpu ms/tok | hit host % | PROF (wall prompt gen disk wait matmul attn head fwd) | TIERS |
|---|---|---|---|---|---|---|---|---|
| code1 | 1,259 | 10,48 | 75,6 | 267,4 | 88,8 | 44,9 | 112,138 29 128 347,078 37,619 4,281 28,336 41,890 156 | 1003 280 6397 18,86 5,26 |
| chat_it1 | 1,077 | 9,71 | 100,9 | 382,0 | 94,2 | 43,1 | 128,504 30 128 493,789 51,689 3,941 30,234 42,627 157 | 1020 280 6380 19,18 5,26 |
| reason1 | 1,199 | 12,21 | 85,9 | 301,5 | 93,5 | 45,0 | 118,986 41 128 394,157 41,681 4,901 30,872 41,518 168 | 1020 280 6380 19,18 5,26 |
| ctx2k | 1,044 | 527,50 | 94,2 | 359,2 | 94,8 | 16,3 | 650,103 2276 128 1393,536 86,525 210,401 311,998 41,065 2403 | 1020 280 6380 19,18 5,26 |

**ab-B1-wnr2** (20:53:09, RAM disp. 17,6 GB; READY 17,3 s; 105.838 letture host + 80.462 dirette, 3.152 GB, 64,1 letture e 1,08 GB per forward)

| prompt | tok/s | TTFT s | non-VRAM/tok | wait ms/tok | gpu ms/tok | hit host % | PROF | TIERS |
|---|---|---|---|---|---|---|---|---|
| code1 | 1,281 | 9,91 | 15,8 | 278,6 | 75,3 | 47,4 | 109,856 29 128 348,594 37,365 5,347 27,152 39,980 156 | 1001 280 6399 18,82 5,26 |
| chat_it1 | 1,116 | 9,43 | 19,6 | 389,4 | 73,4 | 47,8 | 124,131 30 128 491,064 51,451 4,848 27,744 40,076 157 | 1001 280 6399 18,82 5,26 |
| reason1 | 1,226 | 11,56 | 18,8 | 313,1 | 73,7 | 47,9 | 115,999 41 128 394,131 42,211 5,601 28,401 39,774 168 | 1001 280 6399 18,82 5,26 |
| ctx2k | 1,066 | 500,35 | 18,6 | 363,2 | 73,5 | 7,8 | 620,393 2276 128 1269,979 76,207 219,382 284,611 40,079 2403 | 1001 280 6399 18,82 5,26 |

**ab-B2-wnr2** (21:10:07, RAM disp. 18,6 GB; READY 17,9 s; 105.753 letture host + 80.206 dirette, 3.146 GB, 63,9 letture e 1,08 GB per forward)

| prompt | tok/s | TTFT s | non-VRAM/tok | wait ms/tok | gpu ms/tok | hit host % | PROF | TIERS |
|---|---|---|---|---|---|---|---|---|
| code1 | 1,278 | 9,91 | 15,7 | 278,0 | 75,7 | 47,5 | 110,037 29 128 345,282 37,358 5,247 27,210 40,210 156 | 1008 280 6392 18,95 5,26 |
| chat_it1 | 1,118 | 8,73 | 19,6 | 385,3 | 75,4 | 48,1 | 123,220 30 128 487,173 51,140 4,124 27,733 40,212 157 | 1008 280 6392 18,95 5,26 |
| reason1 | 1,211 | 11,58 | 18,6 | 313,5 | 73,5 | 48,3 | 117,281 41 128 392,319 42,365 5,488 28,583 40,832 168 | 1008 280 6392 18,95 5,26 |
| ctx2k | 1,061 | 502,21 | 18,3 | 362,8 | 74,3 | 7,7 | 622,787 2276 128 1273,063 77,171 221,044 284,000 40,460 2403 | 1008 280 6392 18,95 5,26 |

**ab-A2-wex1** (21:27:10, RAM disp. 18,5 GB; READY 15,9 s; 176.257 letture, 3.314 GB, 60,6 letture e 1,14 GB per forward)

| prompt | tok/s | TTFT s | non-VRAM/tok | wait ms/tok | gpu ms/tok | hit host % | PROF | TIERS |
|---|---|---|---|---|---|---|---|---|
| code1 | 1,267 | 10,81 | 74,7 | 262,5 | 90,9 | 45,0 | 111,809 29 128 341,414 37,074 4,259 29,303 41,161 156 | 1027 280 6373 19,31 5,26 |
| chat_it1 | 1,092 | 9,75 | 100,5 | 380,1 | 92,8 | 43,1 | 126,973 30 128 492,059 51,452 3,891 30,442 41,175 157 | 1027 280 6373 19,31 5,26 |
| reason1 | 1,203 | 12,15 | 85,6 | 301,5 | 91,0 | 45,1 | 118,526 41 128 391,323 41,796 4,812 30,871 41,035 168 | 1027 280 6373 19,31 5,26 |
| ctx2k | 1,033 | 521,13 | 93,7 | 355,1 | 93,4 | 16,1 | 645,060 2276 128 1414,777 89,271 205,866 307,835 41,984 2403 | 1027 280 6373 19,31 5,26 |

## Statistiche

| braccio | n (esecuzione × prompt) | media tok/s | IC 95% (t, df=7) |
|---|---|---|---|
| WEX1 (A1, A2) | 8 | 1,147 | ± 0,080 (1,067-1,227) |
| WNR2 (B1, B2) | 8 | 1,170 | ± 0,076 (1,094-1,245) |

**Delta di WNR2 rispetto a WEX1.** È la differenza appaiata per prompt sulle coppie ABBA (A1,B1) e (A2,B2): **+2,04% ± 0,80% (IC 95%: +1,24% … +2,85%)**, con n=8 e t di Student a df=7. Rapporto delle medie dei bracci: +1,99%. Le differenze per coppia:

| prompt | A1→B1 | A2→B2 |
|---|---|---|
| code1 | +1,75% | +0,87% |
| chat_it1 | +3,62% | +2,38% |
| reason1 | +2,25% | +0,67% |
| ctx2k | +2,11% | +2,71% |

**TTFT** (media delle due esecuzioni): code1 10,65 → 9,91 s (-6,9%); chat_it1 9,73 → 9,08 s (-6,7%); reason1 12,18 → 11,57 s (-5,0%); ctx2k 524,3 → 501,3 s (-4,4%).

**Identità dei token.** I 5 prompt hanno token identici in tutte e 4 le esecuzioni ABBA (`tokens.json`, argmax per passo di decode). I file di logits completi sono identici bit per bit:
- ABBA, tutte e 4: sha256 `4300486e5ccf5d9cf2f28e4f3c7ae5bafcba1a9c4d136c2c89d808139c18a69b`;
- smoke, entrambi: `c6e4044f8f956e66661fafc5ce45fbfb713f3d84c4b20fae00ebd7a8ba29393b`.

I dump sono fuori da OneDrive, in `<local scratch dir>\abdump\`.

## Verdetto (regola PLAN 3.2)

WNR2 diventa il formato di velocità solo se valgono tutte e tre le condizioni:

| condizione | esito |
|---|---|
| ≥ +3% tok/s | **no** (+2,04%) |
| IC 95% che esclude lo 0 | sì |
| token identici | sì |

→ **WNR2 = formato di capacità** (-10% di disco sul container). WEX1 resta il formato di velocità in produzione.

## Dove va il tempo (decode, per token, medie delle due esecuzioni)

| voce | WEX1 | WNR2 | delta |
|---|---|---|---|
| `moeprof wait` (attesa pesi) | 263-382 ms | 278-389 ms | +8-12 ms (la decodifica rANS sulla GPU è nel percorso della lane) |
| `moeprof gpu` | 89-95 ms | 73-75 ms | -16-20 ms (meno attach da RAM con upload: `attach` 4,6-7,4 → 0,7-0,9 ms) |
| `decprof attn` | 59-66 ms (ctx2k 132-139) | 67-73 ms (ctx2k 141-143) | +6-9 ms (le decodifiche sulla GPU si sovrappongono alle proiezioni) |
| head (`PROF head` / token generati, letta da U: a 13 GB) | 321-333 ms | 311-319 ms | -8-14 ms (meno contesa su U:) |
| tempo per token (1 / tok/s), media delle 2 esecuzioni | code1 792, chat_it1 922, reason1 833, ctx2k 963 ms | 782, 895, 821, 940 ms | -10, -27, -12, -23 ms (di cui head -12, -14, -8, -10) |
| byte dal disco per forward (run intera, prefill incluso) | 1,14 GB (60,7 letture × 18,8 MB) | 1,08 GB (64,0 letture, ~16,9 MB l'una) | -5% di byte, +5% di letture |

Il risparmio teorico del formato è -10% di byte per record. Il risparmio reale sul disco è solo -5%, perché il percorso GPU-direct rilegge di più: i record caricati in VRAM non entrano nel tier RAM (pool di 32 mirror, `pool_hits` 35.638 su 61.682 riempimenti in B2).

## Comandi

```
DUMP='<local scratch dir>\abdump' \
  sh lab/v41-gpu-20261107/ab-samedrive-20261108/run_ab.sh > run_ab.out 2>&1
# per esecuzione (run_ab.sh, funzione arm):
env $COMMON COLI_WEX_PATH=<WEX1|WNR2> V4_LOGITS_DUMP=$DUMP/<run>.logits \
  C:/Python314/python.exe -I lab/v41-gpu-20261107/m0/serve_bench.py <exe> U:\models\DeepSeek-V4.1-Flash-1017-REAP192-MXFP4 \
  lab/v41-gpu-20261107/ab/prompts.jsonl runs/<run> 128 w0,code1,chat_it1,reason1,ctx2k
C:/Python314/python.exe -I lab/v41-gpu-20261107/ab-samedrive-20261108/analyze_ab.py <DUMP>
```

File in questa cartella:
- `run_ab.sh`, `run_ab.out` (log del driver, con RAM e GPU all'avvio di ogni esecuzione);
- `env.txt`, `hashes.txt`;
- `analyze_ab.py`, `summary.json`, `tokens.json`;
- `runs\*.jsonl` / `*.stderr.log` / `*.meta.json` (log grezzi).

## Limiti

- **Una sola ripetizione ABBA (4 esecuzioni).** Il PLAN 3.1 chiedeva ABBA ×2 e 8 prompt a 256 token. Il compito chiedeva 4 prompt a 128 token. La varianza tra le esecuzioni dello stesso braccio è < 0,5%.
- **`--memory-gb 13` per la regola RAM.** A 16 GB (head residente, 9 slot per layer) la parte di guadagno che viene dalla head (~2/3) sparirebbe e il delta atteso sarebbe ~+1,5% [I]. Solo una ripetizione a 16 GB lo può confermare; è improbabile che porti il delta oltre il +3%.
- **Questa build non ha i contatori `wexprof`/`npprof` (W3) né `V4_WEX_BATCH` (W2).** Le letture dal disco per prompt non sono separabili: le letture/byte riportati sono per esecuzione, prefill incluso. Il batching W2 potrebbe ridurre l'aumento del `wait` di WNR2 [I].
