# W3: WEX1 contro WNR2 a RAM uguale sui tier GPU di V4.1 (motore integrato), 2026-11-09 01:50-05:00

## Riassunto

- **Verdetto: WNR2 non diventa il container di V4.1 in questa configurazione. WEX1 resta in produzione e WNR2 resta un formato di capacità** (-10% di disco: 129,9 contro 144,4 GB).
  - La regola dell'utente (PLAN, aggiornamento 2026-11-08 21:15) adotta WNR2 se valgono due condizioni: token identici in tutto l'ABBA e WNR2 non più lento.
  - Token identici: **sì**.
  - Non più lento: **no**. Il decode di WNR2 è più lento dell'**1,80%**: IC 95% da -2,44% a -1,15% (4 coppie ABBA, t con df=3). Sui singoli prompt il risultato è -1,80% ± 1,40% (n=32).
  - Anche la regola originale del PLAN 3.2 (+3% con IC che esclude lo 0) dà "formato di capacità".
- **Token e logits identici bit per bit** in tutte le 8 esecuzioni ABBA, su tutti i 9 prompt (sha256 dei dump di logits `070fff1105833d65…` in tutte e 8).
  - I due smoke hanno sha256 `c6e4044f8f956e66…`. È lo stesso prefisso dello smoke del 2026-11-08: sul prompt w0 la build integrata riproduce i logits della build G2.
- **Rispetto all'A/B dell'8/11 (+2,0% per WNR2) il segno si è invertito, come quel report aveva previsto.**
  - Quel guadagno veniva per 2/3 dalla head, che a 13 GB era riletta da U: a ogni token.
  - Qui la RAM è a 16 GB, la head è residente e la proiezione della head gira sulla GPU (G3): quel guadagno sparisce.
  - Resta solo il costo della decodifica di WNR2 sulla GPU.
- **Dove va il tempo** (decode, media per token su 32 coppie esecuzione×prompt per braccio): 534,1 ms con WEX1 contro 543,7 ms con WNR2, cioè +9,6 ms.
  - **Attenzione: +8,5 ms** (73,6 → 82,1). Le decodifiche rANS di WNR2 girano sulla stessa GPU e sull'unico copy engine delle proiezioni dense, che fanno andata e ritorno con l'host a ogni chiamata.
  - **MoE in pareggio** (`decprof moe` 422,9 → 423,7 ms):
    - il calcolo sulla GPU scende di 15,9 ms (94,4 → 78,5), perché i record caricati direttamente disco→VRAM non hanno bisogno dell'attach da RAM (`attach` 5,7 → 0,9 ms);
    - l'attesa dei pesi sale di 10,6 ms (312,2 → 322,7), perché la decodifica sta nel percorso di caricamento.
  - mHC uguale: 21,2 contro 21,1 ms.
- **Disco: stessi byte per forward (1,52 GB in entrambi i bracci).** WNR2 fa l'11% di letture in più (90,0 contro 80,7 record per forward) di record più piccoli del 10%.
  - Motivo: i record caricati in GPU-direct (circa il 72% delle letture di WNR2) non passano dal tier RAM, quindi un mirror tolto dalla VRAM si rilegge dal disco. `ram_skips` (~104.000 per esecuzione) conta il caso opposto: candidati GPU-direct che la RAM aveva già o stava caricando, e che hanno preso il percorso host.
  - Il 10% di capacità di WNR2 non arriva mai alla RAM, perché gli slot RAM tengono record decodificati. Lo farebbe solo la cache dei record compressi (`COLI_WEX_RAM_CACHE_GB`), spenta in entrambi i bracci per avere RAM uguale; vedi "Limiti".
- **TTFT** (media delle 4 esecuzioni per braccio):
  - prompt brevi con WNR2 dallo -0,8% al -6,4%;
  - ctx2k +1,9% (400,6 → 408,1 s);
  - w0 +7,2%: è il primo prompt dopo il caricamento.
  - Per prompt l'IC 95% include sempre lo 0.
- **Confound del drive risolto:** entrambi i container sono su U:, nelle copie verificate (marker `.verified`). La differenza D:/E: misurata in M0 qui non entra. Non ho scritto nuove copie. Gli shard engram (47 su C:, 48 su E:) sono gli stessi nei due bracci.
- **RAM uguale, verificata** in tutte le 8 esecuzioni: `ram_tiers available=16.00GiB dense=resident(6.65GiB) target_slots=9 target_cache=6.30GiB head=resident-bf16 projected=15.37GiB`, 360 slot RAM, 934-980 mirror in VRAM.
  - Problema trovato e corretto nel driver: il motore usa come budget min(`--memory-gb`, RAM disponibile per l'OS) all'apertura. Lo smoke A partì con 15,49 GB disponibili e ricevette 6 slot per layer invece di 9.
  - Da allora il driver aspetta di avere almeno `RAM_GB + 0.3` GB disponibili prima di ogni esecuzione e controlla la riga `ram_tiers` (`ram_budget_full=1` in tutte le 8).

## Tabelle

**tok/s di decode per prompt** (media di 4 esecuzioni per braccio; nella colonna delle coppie, i delta di WNR2 rispetto a WEX1 per A1-B1, A2-B2, A3-B3, A4-B4):

| prompt | token | WEX1 | WNR2 | delta medio | coppie | TTFT WEX1 s | TTFT WNR2 s |
|---|---|---|---|---|---|---|---|
| code1 | 130 (EOS) | 2,268 | 2,156 | -4,92% | -3,6 / -5,8 / -4,6 / -5,6 | 10,56 | 9,88 |
| code2 | 256 | 2,146 | 2,044 | -4,74% | -4,1 / -5,1 / -4,4 / -5,3 | 10,80 | 10,59 |
| chat_it1 | 256 | 1,673 | 1,648 | -1,49% | -1,0 / -2,4 / -1,5 / -1,1 | 9,71 | 9,38 |
| chat_it2 | 256 | 1,865 | 1,996 | +7,01% | +7,9 / +6,1 / +7,3 / +6,7 | 10,28 | 10,00 |
| reason1 | 217 (EOS) | 1,881 | 1,798 | -4,37% | -4,0 / -4,7 / -4,5 / -4,4 | 11,95 | 11,64 |
| reason2 | 240 (EOS) | 1,869 | 1,784 | -4,52% | -4,4 / -4,3 / -4,5 / -4,9 | 11,19 | 11,10 |
| ctx2k (2.276 tok) | 256 | 1,537 | 1,534 | -0,16% | +0,5 / +0,7 / -0,1 / -1,7 | 400,64 | 408,09 |
| short1 | 8 (EOS) | 1,945 | 1,921 | -1,20% | -1,7 / -1,0 / -1,0 / -1,1 | 4,38 | 4,25 |

- **Media per coppia:** -1,29 / -2,06 / -1,65 / -2,18%, quindi **-1,80% ± 0,65%** (df=3).
- Senza short1, che ha solo 8 token: -1,88% ± 1,60% (n=28).
- Media dei bracci: 1,898 contro 1,860 tok/s, cioè -1,98%.
- Il delta di ogni prompt è stabile tra le coppie, entro circa 1 punto: l'effetto dipende dal prompt e non è rumore.
  - chat_it2 è l'unico prompt dove WNR2 vince. Lì l'hit rate dello store host è 62% contro 41%.

**Latenza per token** (media per prompt, dai timestamp di `decprof`): p50 538,9 → 543,9 ms (+0,9%); p90 638,5 → 617,0 ms (-3,4%). Con WNR2 la coda è un po' più corta.

**Contatori per esecuzione** (prefill di ctx2k incluso; `wexprof` stampa i totali solo all'uscita):

| esecuzione | READY s | letture/forward | GB/forward | letture host + dirette | wexprof domanda: record, ms lettura/rec, ms decodifica/rec | wexprof prefetch: record, ms lettura/rec, ms decodifica/rec |
|---|---|---|---|---|---|---|
| A1 / A2 / A3 / A4 | 15,8 / 15,4 / 16,9 / 16,0 | 81,1 / 80,8 / 80,7 / 80,1 | 1,52 / 1,52 / 1,52 / 1,51 | 335k / 334k / 333k / 331k + 0 | — | — |
| B1 | 17,9 | 90,0 | 1,52 | 105k + 267k | 53.729 (1,58/batch), 7,02, 0,640 | 210.199 (1,96/batch), 9,63, 0,702 |
| B2 | 18,0 | 90,4 | 1,53 | 105k + 268k | 54.015, 7,08, 0,633 | 211.394, 9,67, 0,691 |
| B3 | 19,3 | 90,0 | 1,52 | 105k + 267k | 54.018, 7,08, 0,625 | 209.908, 9,70, 0,679 |
| B4 | 18,1 | 89,3 | 1,51 | 104k + 265k | 53.336, 7,06, 0,633 | 208.542, 9,70, 0,695 |

- **WNR2.** Il batch W2 copre quasi tutto: il percorso per record resta a 9-14 caricamenti su domanda e 2.700-3.100 di prefetch per esecuzione. GPU-direct: 0 fallimenti in tutte le esecuzioni.
- **Il prefetch domina:** circa l'80% dei caricamenti GPU-direct viene dal prefetch.
- **Precisione del predittore:** recall 76,4% a k=8. La precisione a profondità 1 è il 57,3% (284.369 su 496.000 ipotesi); il dettaglio è nella sezione F1.

## Setup

- **Build fresca da `src` in `ws-ab`.**
  - `src` coincide con `ws-w/c`, cioè con W2 finale (mtime di `deepseek_v4.c`: 01:19).
  - Comandi: `python -I ws.py sync ws-ab`, `ws.py make ws-ab build -f Makefile.deepseek-v4 deepseek-v4`, poi le DLL con le ricette `build-dll.cmd` e `build-dg-dll.cmd` (copiate in `ws-ab\dll-out\`).
  - L'exe differisce da quello misurato in W2 (`ws-w\bin`, compilato alle 01:01, prima dell'ultima modifica di W2 alle 01:19). Questa build contiene quella modifica.

| file | sha256 |
|---|---|
| `ws-ab\bin\deepseek_v4.exe` | `7a6f3e3f30e608c900af35f7d2ea13a0d81a6432c295693e465f6351a9b7459f` |
| `ws-ab\dll-out\coli_cuda_dsv4_dg.dll` (caricata, `COLI_DSV4_DLL`) | `b1b7cdb518d2f309371602efbe58c48736574962caa1ef9b7a12af81f12e404b` |
| `ws-ab\dll-out\coli_cuda_dsv4.dll` (non caricata) | `5483b6a774c2bab892809d7561a434936d1bc3803ac46e9877c03e3b5320fc52` |
| `w1\build\kaira_wex_cuda.dll` (helper) | `cceb4fb573b3b3e7b71843bf85a2b87cf9efdd94155bfc5deec717cce949e782` |
| `ws-ab\bin\cudart64_13.dll` / `cublasLt64_13.dll` | `d26fd1c2…` / `cad63434…` (identiche a quelle del toolkit 13.4) |

- **Gate sulla build:** `tests\test_v41_on_v4_gpu_refusal.exe ... up` con entrambe le DLL nuove: **PASS** (tier fp8_block 32 attivo, reserve/publish dei mirror di W2, rifiuto a runtime della head GPU). Log in `ws-ab\gates.log`.
  - Non ho rifatto i gate completi del 5.1. La lane W2 li ha passati alle 01:22 sullo stesso sorgente (`ws-w\gates3.log`, ALL PASS).
  - Nota: il test carica la DLL con un path assoluto, quindi `cudart64_13.dll`/`cublasLt64_13.dll` devono stare accanto all'exe del test (come in `ws-w\c\tests`).
- **Bracci.** Differiscono solo in `COLI_WEX_PATH`:
  - A = `U:\kaira_wex\v41-reap\v4_flash_experts.wex` (WEX1, Container v1);
  - B = `U:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex` (WNR2, Container v3).
  - Ogni log mostra il container atteso (controllato da `run_w3.sh`).
- **Ambiente** (`env.txt`): quello di W2, cioè le impostazioni di decode di kaira.json, più:
  - `COLI_WEX_GPU_DIRECT=2`, `V4_WEX_BATCH=1`, `V4_GPU_HEAD=1`, `RAM_GB=16`, `DSV4_CUDA_VRAM_RESERVE_MB=4096`;
  - impostazioni lossless: `V4_SKIP_COLD=0`, `V4_MTP=0`, `V4_TOPK` non impostato, greedy;
  - F1 ai valori di default (spento).
- **Deviazione dal PLAN 3.2: nessun `COLI_WEX_RAM_CACHE_GB`.** Quella cache tiene solo record Neural-rANS e si aggiunge a `--memory-gb` (`v4_wex_crc`, deepseek_v4.c:14745). Con 4 GB in entrambi i bracci, WNR2 avrebbe 20 GB contro 16.
- **Protocollo 3.1 completo:**
  - prompt: w0 di riscaldamento più gli 8 prompt di `prompts.jsonl`, a 256 token;
  - ordine ABBA ×2: A1 B1 B2 A2 A3 B3 B4 A4;
  - ogni esecuzione è un processo nuovo (cache fredde);
  - controlli automatici per esecuzione: READY, container, 40 upload densi, dense=gpu, prefetch, GPU-direct, batch W2, MoE in pipeline GPU, budget RAM pieno.
- **heavy.lock** (protocollo PowerShell `New-Item`, etichetta "W3 A/B ..."), con la macchina quieta (nessun nvcc/cl/make/gcc/robocopy/python tranne il server MCP):
  - smoke: lock dalle 01:50:04 alle 01:51:34;
  - ABBA: lock dalle 01:52:16 alle 04:59:47.
- **Macchina:**
  - RAM disponibile all'avvio delle esecuzioni: 17,5-21,1 GB;
  - GPU prima di ogni esecuzione: 1,06-1,46 GiB usati, 0-7% di utilizzo;
  - stessa sessione di boot;
  - piano energetico "Bilanciato" (non l'ho cambiato);
  - esclusioni di Defender non verificate.
- **Logger di temperatura.** Durante B4 e A4 girava un logger di temperatura NVMe che **non ho avviato io** (`ab\nvme_temp.py` e `nvme_temps.log`, comparsi alle 04:14, campionamento ogni 30 s).
  - Il sensore più caldo di U: ha raggiunto 84 °C in B4 e 82 °C in A4 (soglia di warning 86 °C).
  - La coppia A4/B4 (-2,18%) è in linea con le altre tre: nessun segno che abbia spostato il risultato.

## F1: scelta offline delle soglie (PLAN 3.3, passo 1)

Dati: istogramma `v4_next_predict_conf` di `V4_PREDICT_STATS=1`, identico in tutte le 8 esecuzioni perché il routing è deterministico. Le soglie si leggono per profondità.

| profondità | precisione totale | precisione per fascia di confidenza (0,05 l'una, da 0) | τ=0,10 blocca | τ=0,15 blocca |
|---|---|---|---|---|
| d1 | 57,3% | 27, 33, 46, 71, 95, 98, 99, 99, 99, 99% | il 19,8% delle ipotesi e l'11,4% di quelle usate (precisione restante 63,3%) | il 61,3% delle ipotesi e il 44,7% di quelle usate (precisione restante 82,0%) |
| d2 | 73,4% | 37, 52, 58, 68, 91, 97, 98, 99, 99, 99% | 2,9% e 2,0% | 26,7% e 20,8% |

Le soglie scelte sono τ1 = 0,10 e τ2 = 0,15:
- **τ1 = 0,10** blocca le fasce con precisione sotto il ~35%;
- **τ2 = 0,15** blocca quelle sotto il ~50%. La fascia 0,15-0,20 ha già il 71%, quindi i punti al 50% e al 70% cadono entrambi a 0,15.
- Con τ2 si perde quasi metà dei prefetch utili a d1. Per questo mi aspetto che τ2 rallenti, e che solo τ1 possa guadagnare [I].

**La griglia F1 non è stata misurata.**
- Script pronti: `f1\run_f1.sh` con i bracci base, t10, t15, m287 (16 record WEX1 per forward), m144 (8 record) e base2, su w0 + 3 prompt a 128 token. `ab\locked_pilots.ps1` lancia prima il pilota C e poi la griglia.
- Li ho annullati prima che partissero. Alle 05:00:36 la sessione principale ha preso heavy.lock per spostare file su U: e C: (cancellazione di WEX1 da D:, shard engram 47 da C: a D:, WNR2 da E: a C:). Quegli spostamenti cambiano l'ambiente di esecuzione (`COLI_MODEL_DIRS`), quindi una misura fatta dopo non sarebbe confrontabile con questa.
- Non è partita nessuna esecuzione e il lock non è mai stato mio.

## Limiti e prossimi passi

1. **Il braccio "cache compressa a byte uguali" non è stato misurato.** È l'unica configurazione in cui la capacità di WNR2 arriva in RAM (pilota C: WEX1 a 16 GB contro WNR2 a `RAM_GB=12` + `COLI_WEX_RAM_CACHE_GB=4`).
   - Lo script è pronto: `ab\run_w3c.sh` (ABBA corto pA1 pC1 pC2 pA2, con analisi in `ab\analyze_pilots.py`).
   - Va rilanciato dopo gli spostamenti di file, aggiornando `COLI_MODEL_DIRS` se lo shard 47 non è più su C:.
   - Se C batte A, il verdetto andrebbe riaperto per la combinazione WNR2 + cache.
2. **Il costo di WNR2 è nella contesa sulla GPU con l'attenzione** (+8,5 ms/token), non nel disco.
   - Leve [I]: decodificare in batch più grandi o fuori dalla finestra delle proiezioni dense.
   - Leva definitiva: la catena device-resident (G5/G6/G7), che toglie le andate e ritorno con l'host per ogni chiamata.
   - Un'alternativa: fare passare i record GPU-direct anche dal tier RAM (oggi lo saltano, da cui le riletture +11%).
3. **`wexprof` stampa solo i totali all'uscita.** I conteggi di WNR2 per prompt e per token non sono separabili; la riga `wexprof` per token chiesta dal PLAN (W3) non esiste. Non ho modificato sorgenti in questo package.
4. **I byte su disco per esecuzione includono il prefill** (ctx2k da solo legge centinaia di GB).
5. **I dump dei logits** (6,4 GB) sono fuori da OneDrive: `<local scratch dir>\w3dump\`.

## Comandi e file

```
powershell -NoProfile -ExecutionPolicy Bypass -File lab\v41-gpu-20261107\ab\locked_w3.ps1 <DUMP>   # heavy.lock + quiete + run_w3.sh
PYTHONUTF8=1 python -I lab\v41-gpu-20261107\ab\analyze_w3.py <DUMP>                              # -> summary.json, tokens.json
```

- **Driver:**
  - `run_w3.sh`: regola RAM, attesa della RAM disponibile prima di ogni esecuzione, controlli per esecuzione, file `STOP` per fermarsi dopo un ABBA;
  - `locked_w3.ps1`: lock, controllo di quiete, rilascio solo del nostro lock;
  - output in `locked_w3-smoke.out` e `locked_w3.out`.
- **Dati:**
  - `runs\w3-*.jsonl`, `*.stderr.log`, `*.meta.json` (log grezzi);
  - `summary.json` (tutti i numeri per esecuzione e prompt, delta, identità dei token, sha256);
  - `tokens.json` (id greedy per esecuzione e prompt);
  - `analyze_w3.out`;
  - `env.txt`, `hashes.txt`.
- **Piloti non eseguiti:** `run_w3c.sh`, `locked_pilots.ps1`, `analyze_pilots.py`, `..\f1\run_f1.sh`.
