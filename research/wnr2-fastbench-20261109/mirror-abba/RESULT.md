# Mirror ABBA: WNR2 su U:+C: contro WEX1 su U: (2026-11-09) — NON ESEGUITO (misurato poi il 10/11, vedi la nota sotto)

## Riassunto

- **Verdetto: NOT_RUN.** Nessuna misura. La domanda resta aperta.
- Ho aspettato 4 ore, dalle 19:11 alle 23:12. La macchina non è mai stata libera abbastanza a lungo.
- Cosa ha bloccato (minuti su 227):
  - RAM disponibile sotto 13,3 GB: 74 min. un'app di texturing 3D usava 14-15 GB (52 GB di commit), editor di giochi altri 4 GB.
  - Altri motori o test sulla GPU (deepseek_v4 di altre sessioni): 56 min.
  - Compilazioni (gcc, make, nvcc, cl): 41 min.
  - app 3D aperto: 29 min, dalle 22:49 fino alla scadenza.
  - GPU sopra il 10%: 24 min.
- Ho preso heavy.lock due volte, alle 19:40 e alle 21:18. Entrambe le volte la RAM è crollata subito dopo: da 16 GB a 3-13 GB in pochi minuti. Anche la VRAM usata da altri era 29 GB su 32.
- In quei casi ho fermato il mio smoke prima che partisse il motore e ho rilasciato il lock. Il lock è stato tenuto 1 min la prima volta e 11 min la seconda.
- Nessun motore è partito. Nessuna lettura dai container. Nessun processo altrui fermato.
- Senza esecuzioni non ci sono tok/s, TTFT, attesa MoE, latenza per lettura, quota per drive, identità dei token né temperature NVMe sotto carico.
- Per la produzione resta valido il dato precedente: WEX1 su U: resta il container finché il mirror non viene misurato.
- *Aggiornamento 2026-11-10: il mirror è stato misurato con `short-abba` (+49%, [NOTE-5](../NOTE-5-mirror-ab-20261110.md)); poi KAIRA è passato a WNR3 con mirror e WEX1 è stato cancellato ([WNR3-PRODUCTION-20261110.md](../../../docs/reports/WNR3-PRODUCTION-20261110.md)).*

## Cosa serve per rilanciare

- La macchina deve restare libera per circa 45 minuti:
  - un'app di texturing 3D, editor di giochi e app 3D chiusi, oppure almeno 13,3 GB di RAM liberi in modo stabile;
  - nessuna compilazione e nessun altro motore in esecuzione;
  - KAIRA senza motore attivo.
- Comando (aspetta da solo la macchina libera e il lock, al massimo `WAIT_HOURS` ore, default 10):

  ```
  bash <KAIRA>/lab/wnr2-fastbench-20261109/mirror-abba/run_mirror_abba.sh
  ```

- ~~Il guard dello strumento non conosce un'app di texturing 3D e editor di giochi.~~ Dopo la revisione 2 (VALIDATION.md) c'è un gate VRAM: prima del lock e prima di ogni corsa serve `memory.used` < 4 GB (`VRAM_MAX_GB`) senza motori in esecuzione. È questo che vede Painter, editor di giochi e app 3D (app C+G su WDDM), oltre alla RAM. Durante le corse un campionatore invalida il set se la RAM o la VRAM di altri cresce.
- Regola RAM: 16 solo con ≥ 16,3 GB disponibili, altrimenti 13. Se prima di una corsa il gate resta chiuso 15 min, il set diventa INVALID senza avviare il motore.

## Setup preparato (pronto, non eseguito)

- **Driver:** `mirror-abba/run_mirror_abba.sh`. Esegue questi passi:
  1. aspetta che `short_abba.sh --check` dica "quiet" e che la RAM disponibile sia almeno 13,3 GB per 5 controlli di fila (1 al minuto);
  2. prende heavy.lock una volta sola per smoke e ABBA; lo rilascia solo se il file contiene ancora la sua riga;
  3. lancia lo smoke del braccio M (short1, 16 token, `--seq B1`). Il controllo della corsa in `short_abba.sh` richiede "Matching mirror container active", "[v4_wex] read split over 2 drives" e almeno il 10% dei byte letti (contatori OS) su ciascun drive. Se manca qualcosa si ferma (exit 6) e non misura;
  4. lancia l'ABBA A1 B1 B2 A2: w0 di riscaldamento, poi 7 prompt × 64 token; l'analisi la fa `analyze_short.py`;
  5. registra le temperature NVMe di C: (disco 3) e U: (disco 5) ogni 10 s in `nvme_temps.log` di ogni passo.
- **Motore:** il motore installato, copiato in `mirror-abba/bin` con sha256 uguale alla copia in app:
  - deepseek_v4.exe `7a6f3e3f…`
  - coli_cuda_dsv4_dg.dll `b1b7cdb5…`
  - coli_cuda_dsv4.dll `5483b6a7…`
  - kaira_wex_cuda.dll `cceb4fb5…`
  - cudart64_12.dll `c2c9a9c2…`
  - Il PATH parte da CUDA v13.4 bin/x64, come nel profilo.
- **Env:** le chiavi del profilo v41gpu in `app\kaira.json`, letto e non modificato. Coincidono con COMMON di `short_abba.sh`. In più il contesto del profilo, `CTX=32768`.
  - Braccio A: `COLI_WEX_PATH=U:\kaira_wex\v41-reap\v4_flash_experts.wex`, senza mirror e senza pesi.
  - Braccio M: `COLI_WEX_PATH=U:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex`, `COLI_WEX_EXTRA_MIRRORS=C:\kaira_wex\v41-reap`, `COLI_WEX_WEIGHTS=38,62`.
  - `V4_WEX_BATCH=1` e `COLI_WEX_GPU_DIRECT=2` in entrambi i bracci.
  - Il motore cerca `coli_cuda_dsv4_dg.dll` per prima, quindi `COLI_DSV4_DLL` sulla copia dg è lo stesso default dell'app.
- **Differenze note rispetto a KAIRA:**
  - KAIRA avvia il motore tramite `coli serve`, che aggiunge `OMP_WAIT_POLICY=active`, `GOMP_SPINCOUNT=200000` e `OMP_DYNAMIC=FALSE`. Lo short ABBA non le imposta (come W3). Non sono nel profilo.
  - KAIRA usa un budget RAM automatico. Qui vale la regola dell'utente: 16 GB oppure 13 GB, uguale nei due bracci.
- **Mirror su C:**: la copia ha la stessa dimensione dell'originale su U: (129.905.360.896 byte) e ha il marker `.verified`. Non ho riverificato lo sha256.
- **Codice del motore** (src, solo lettura): con `COLI_WEX_WEIGHTS` e un mirror, `v4_wex_pick` divide le letture per hash di (layer, esperto). Le letture GPU-direct passano dallo stesso `v4_wex_pick`, quindi anche i record WNR2 caricati direttamente in VRAM vengono divisi tra i due drive.

## Registro dei tentativi

| ora | evento |
|---|---|
| 19:11 | primo driver. Attesa: compilazioni, lock di altre sessioni, motori di test sulla GPU |
| 19:40 | lock preso. RAM 9,9 GB < 15: la regola aspetta fino a 10 min. Visto un'app di texturing 3D con 14 GB, ho fermato lo smoke prima del motore. Lock rilasciato alle 19:41 |
| 19:41 | secondo driver, con controllo RAM ≥ 13,3 GB prima del lock |
| 21:18 | lock preso con 16 GB liberi. Regola RAM: 16. Poi la RAM disponibile è scesa a 3-13 GB (Painter + editor di giochi) e la VRAM usata da altri a 29 GB. Lo smoke aspettava 16,3 GB. Ho fermato lo smoke prima del motore. Lock rilasciato alle 21:30 |
| 21:30 | terzo driver: RAM ≥ 13,3 GB per 5 controlli di fila prima del lock. Scadenza invariata (23:11) |
| 22:49 | app 3D aperto, GPU 11-16% |
| 23:12 | NOT_RUN: scadute le 4 ore. Ultimo blocco: app 3D, GPU 11% |

Nota: alle 21:28, nel secondo tentativo, il primo segnale di stop ha colpito per errore il campionatore `quiet.log` del mio stesso script invece dello script stesso. Ho poi fermato lo script giusto. Erano solo processi miei e nessun motore era partito.

## File

- `mirror-abba/run_mirror_abba.sh`: il driver.
- `mirror-abba/runs/driver-*.log`: i tre driver, con tutto quello che hanno aspettato.
- `mirror-abba/runs/smoke-20261109-191155/` e `smoke-20261109-194157/`: gli smoke fermati prima del motore. Contengono solo log, env, hash, `quiet.log` e temperature a riposo; nessuna esecuzione.
- `mirror-abba/bin/`: le copie del motore installato, verificate con sha256.
