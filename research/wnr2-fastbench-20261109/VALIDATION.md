# Validazione dei banchi veloci WEX1/WNR2 (trace replay e short ABBA), 2026-11-09

## Riassunto (aggiornato dopo la revisione critica, 2026-11-09 sera)

- **Lo short ABBA è l'A/B di sviluppo, da usare a macchina quieta.** Dopo la revisione lo strumento è stato corretto (vedi "Revisione").
  - **Nuovo set di prompt:** w0 di riscaldamento, poi 7 prompt a 64 token, in quest'ordine: code1, code2, chat_it1, chat_it2, reason1, reason2, short1. Sono tutti i prompt di W3 tranne ctx2k.
    - Il vecchio set (code1, chat_it1, reason1 a 128 token) era spostato di 0,7-1 punto rispetto alla media larga di W3. Tra un prompt e l'altro la differenza ha SD ~3,8 punti, cioè circa 10 volte il rumore tra coppie.
    - Controllo con `validate/cut_tokens.py` su W3 tagliato a 64 token, 7 prompt: **-1,89%** (± 0,70 sulle medie di coppia, df=3). A 256 token: -1,99%.
  - **Esecuzione delle 05:40** (16 GB, vecchio set di 3 prompt), rianalizzata con l'IC nuovo (prompt come effetti fissi, df=3): WNR2 **-2,69% ± 0,38%**.
    - Prima l'IC era ± 1,85%, perché trattava i 6 delta come indipendenti.
    - Verifica incrociata con le medie per coppia: ± 1,55% (df=1).
    - Validità: la SD tra i prompt è 1,95 punti, quindi un altro set può spostare la media di circa 1,1 punti.
    - Token e logits identici bit per bit (sha256 `0a3eb3359c844836`). Verdetto: **NOT ADOPTED**, come W3.
  - **Rilancio serale a 13 GB:** ora dà **INCONCLUSIVE** e non più "KEEP B". Le cause: stalli in chat_it1 (B2, token da 4,0 s) e in reason1 (A2 25 s, B2 28 s), e spread di code1 tra le esecuzioni dello stesso braccio (A 4,6%, B 3,9%).
  - **Due cautele emerse dalla revisione:**
    - **Regola dello spread.** Lo spread max-min cresce con il numero di esecuzioni per braccio. Su W3 (4 esecuzioni per braccio) la regola "> 2%" fa scattare INCONCLUSIVE (A code1 3,13%). Con 2 esecuzioni per braccio l'esecuzione delle 05:40 resta sotto (massimo 1,08%).
    - **IC a effetti fissi.** Su W3 a 64 token l'IC a effetti fissi è ± 0,31, contro ± 0,70 sulle medie di coppia. C'è quindi un piccolo effetto per esecuzione che l'IC a effetti fissi non vede. Un verdetto con l'IC vicino allo 0 va confermato con più coppie o con W3.
- **Il replay è solo un esercitatore del drive con verifica dei byte.** Non va usato per classificare WEX1 contro WNR2, né per mettere a punto il percorso WNR2.
  - **Differenze dal motore:**
    - buffer privati invece dell'anello da 16 slot;
    - batch circa 2 volte più grandi;
    - nessun percorso host per il 31% delle letture;
    - nessun drain dello stream;
    - domanda limitata a 1 per layer.
  - **`B_pad` è circolare:** è calibrato sul contatore di letture del motore, quindi il segno "giusto" che dà è costruito, non predetto.
  - Resta utile per tre cose: ripetibilità del drive (spread 0,1-0,3%, uguale a 12 ore di distanza), verifica byte per byte di WNR2 (64/64 record identici a WEX1) e GB/s del drive.
- **Perché WNR2 aspetta di più nel motore: è un'ipotesi, non una causa dimostrata.** La frase precedente ("WNR2 aspetta di più perché legge più record, il percorso di lettura non è più lento") è ritirata.
  - Nel motore (PROF `edisk`, short ABBA delle 05:40) il percorso di lettura di WNR2 *è* più lento:
    - latenza totale di lettura **+30%** (1.292-1.324 s contro 1.010-1.012 s) per **+0,3-0,9% di byte** (1.392-1.398 contro 1.385-1.388 GB);
    - **15,7-16,0 ms per lettura contro 13,7**;
    - **5,5-5,7 letture in volo contro 4,4** (legge di Little: `edisk` / wall).
  - WNR2 fa l'11-12% di letture in più (82,3-82,6 mila contro 73,7-73,8 mila), di record più piccoli. Nel motore le sue letture fanno più coda e sono più lente.
  - Il replay diceva il contrario (latenza su domanda 11 ms contro 19 ms): un'altra conferma che il replay non riproduce il percorso WNR2 del motore.
  - **Hit in RAM:** scendono da 42,7 a 20,7 per token (trace registrati). Questo è il dato da citare.
  - **Lettura sbagliata di `ram_skips`, corretta:** `v4_wex_direct_ram_skips` non conta i record che saltano la RAM. Conta le volte in cui il GPU-direct è stato rifiutato perché la RAM aveva già il record, cioè gli *usi* del tier RAM.
  - **Cause concorrenti da provare con short ABBA che cambiano solo l'env:**
    - `V4_NEXT_PREFETCH_THREADS` 4 contro 8;
    - `COLI_WEX_GPU_DIRECT_PREFETCH_RANK`;
    - `COLI_WEX_RAM_CACHE_GB` > 0.
- **Raccomandazione:**
  - **short ABBA** (nuovo default, stima ~22-25 min a 16 GB, non ancora misurata) per ogni decisione di container o di politica;
  - la domanda di produzione (WNR2 con mirror U:+C: contro WEX1 su U:) si misura con `short_abba.sh --app-engine`: l'esempio è nell'intestazione dello script, il braccio B è il mirror M;
  - **replay** solo per la salute del drive e la verifica dei byte.

## Revisione (2026-11-09 sera): cosa è cambiato negli strumenti

Copia `.bak` di ogni file prima della prima modifica: `short-abba/{short_abba.sh,analyze_short.py,serve_bench.py}.bak`, `validate/cut_tokens.py.bak` e `VALIDATION.md.bak`. Nessuna esecuzione pesante: tutto è stato verificato sui log salvati.

| # | file | modifica |
|---|---|---|
| 1 | `short_abba.sh` | Default: w0 + 7 prompt a 64 token, in ordine fisso e documentato (vedi sopra). |
| 2 | `analyze_short.py` | IC con i prompt come effetti fissi e residui tra coppie (df = coppie × prompt − prompt). La SD tra prompt è stampata a parte come nota di validità, insieme a una verifica incrociata sulle medie di coppia. Niente più IC per braccio: restano solo le medie. Verdetto: **KEEP B** solo con media ≥ 0 e token id identici; **NOT ADOPTED** con tutto l'IC sotto lo 0 (o token id diversi); altrimenti **INCONCLUSIVE**. Su un set non valido stampa **INVALID** e mai un verdetto. |
| 3 | `analyze_short.py` | Controllo degli stalli: un prompt è invalido se un suo token di decode ha `decprof` hc+attn+moe > max(2 s, 3 × mediana dell'esecuzione) oppure `moeprof` gpu > 1 s. Il prompt esce da tutte le coppie e il set diventa INCONCLUSIVE. INCONCLUSIVE anche quando, nello stesso braccio, lo spread di tok/s di un prompt supera il 2%. |
| 4 | `short_abba.sh` | Controllo "macchina quieta", anche con `--no-lock`: prima di partire attende (poll ogni 60 s, fino a `--quiet-max`, default 4 h) finché sulla GPU non resta nessun processo estraneo di solo calcolo (tipo C in `nvidia-smi -q -d PIDS`: con WDDM le app desktop sono C+G), l'utilizzo GPU è < 10% e non gira nessun cc1/gcc/make/nvcc/cl/app 3D né motore KAIRA. Il controllo si ripete dopo aver preso il lock. Durante le esecuzioni un campionatore scrive `quiet.log` ogni 10 s: i PID del nostro motore (`<run>.pid`, scritto da `serve_bench.py`) sono esclusi, e ogni hit va in `OUT/INVALID` e ferma l'ABBA. Nel log resta scritto cosa ha atteso. |
| 5 | `short_abba.sh` | Regola RAM con il ciclo di `record_trace.sh`: attende fino a 10 min di avere ≥ 15 GB, altrimenti sceglie 13. La decisione è registrata in `env.txt` e nel log. |
| 6 | `short_abba.sh` | Rilascio del lock: salva la riga esatta scritta da `take_lock` e cancella il lock solo se il file è ancora uguale a quella riga, come `run_validate.sh`. |
| 7 | `analyze_short.py` | Percorso di lettura per esecuzione e per braccio: somma di PROF `edisk` / contatore di letture all'uscita (`v4_direct` + `v4_wex_gpu_direct`) = ms per lettura; `edisk` / wall = letture in volo. Quota per drive dalla riga `v4_mirror` del motore, quando c'è, e dai contatori dei byte letti dall'OS (`<run>.disk.txt`, salvati da `short_abba.sh`). Con un container solo-mirror il motore non conta le letture per drive. |
| 8 | `validate/cut_tokens.py` | Tabella T975 fino a df=30, poi 2,0. Stampa tre IC (effetti fissi, medie di coppia, iid). Default: i 7 prompt a 64 token. |
| 9 | `short_abba.sh` | Dopo il confronto di token e logits restano solo gli sha (nel log, in `summary.json` e in `tokens.json`, con gli id argmax per rianalizzare). I dump creati da quell'esecuzione vengono cancellati; nient'altro viene toccato. `analyze_short.py` senza `--dump` legge `tokens.json`. |
| 10 | `VALIDATION.md` | Questo riassunto: causa declassata a ipotesi, `ram_skips` corretto, limiti del replay. |
| 11 | `short_abba.sh` | `--a-env` e `--b-env` passano l'env di ciascun braccio (per esempio `COLI_WEX_EXTRA_MIRRORS` e `COLI_WEX_WEIGHTS`). `--app-engine` copia l'engine installato (`deepseek_v4.exe`, `coli_cuda_dsv4_dg.dll`, `coli_cuda_dsv4.dll`, `kaira_wex_cuda.dll`, `cudart64_12.dll`) in `mirror-abba/bin`, controlla lo sha256 di ogni file contro la copia in app\ e lo esegue da lì, con il CUDA v13.4 del profilo in testa al PATH. Con un mirror nell'env, il controllo dell'esecuzione richiede: "Matching mirror container active", "read split over 2 drives" e almeno il 10% dei byte letti dall'OS su ciascuno dei due drive. `--check` stampa cosa vedono adesso i controlli "macchina quieta" e RAM, senza attendere. |

**Controlli eseguiti (solo log salvati):**
- `analyze_short.py runs/20261109-054051 --dump …`:
  - **-2,69 ± 0,38%** (df=3), medie di coppia ± 1,55;
  - percorso di lettura A 13,70-13,71 ms/lettura e 4,38 in volo, B 15,70-16,02 ms e 5,54-5,67;
  - token identici. Verdetto **NOT ADOPTED**.
- `analyze_short.py runs/20261109-175141 --dump …`: **INCONCLUSIVE**. Gli stalli sono:
  - B2/reason1, token da 28,4 s (il `moeprof` gpu 23,5 s delle 18:11);
  - A2/reason1, app 3D (fino a 25 s per token);
  - B2/chat_it1, un token da 4,0 s.
  - Resta solo code1 (-0,62 ± 4,78%, df=1), con spread A 4,6% e B 3,9%.
- Con un file `INVALID` nella cartella stampa **INVALID**. Rianalisi senza dump da `tokens.json`: stesso risultato.
- W3 completo (8 prompt, 256 token), solo come prova: -1,80 ± 0,25% a effetti fissi (results.md: ± 0,65 sulle medie di coppia). Verdetto INCONCLUSIVE per la regola dello spread (A code1 3,13%).
- `cut_tokens.py` su W3 con i 7 prompt:
  - a 64 token: **-1,89%**, ± 0,31 a effetti fissi, ± 0,70 sulle medie di coppia, ± 1,30 iid;
  - a 256 token: **-1,99%**, ± 0,28 / ± 0,63 / ± 1,55.
- `bash -n` su `short_abba.sh`, `run_validate.sh` e `record_trace.sh`: OK.
- `short_abba.sh --check` e `--check --app-engine` con il braccio mirror:
  - copie in `mirror-abba/bin` con sha256 uguale ad app\;
  - l'engine installato è identico a ws-ab (`7a6f3e3f…`, `b1b7cdb5…`, `cceb4fb5…`).

## Revisione 2 (2026-11-09 notte): VRAM, banda morta della RAM, spread, verdetto, dump

Copie prima di questa revisione: `*.bak2` accanto a `short_abba.sh`, `analyze_short.py`, `cut_tokens.py`, `run_mirror_abba.sh`, `VALIDATION.md` e `RESULT.md` (i `.bak` restano quelli della prima revisione). Nessuna esecuzione pesante: solo log salvati, `--check` (sola lettura) e due controlli senza motore. Sostituisce i punti 3 (spread), 5 (RAM) e 9 (dump) della tabella sopra.

| # | file | modifica |
|---|---|---|
| 1 | `short_abba.sh` | **Gate VRAM.** Prima del lock (anche con `--no-lock`) e prima di ogni esecuzione serve `memory.used` < `--vram-max-gb` (default 4, env `VRAM_MAX_GB`; 0 o `DSV4_CUDA=0` lo spengono), senza nessun motore KAIRA in esecuzione, il nostro compreso. È l'unico controllo che vede un'app di texturing 3D, editor di giochi e l'interfaccia di app 3D (app C+G su WDDM: 29 GB su 32 il 2026-11-09). La lettura va in `env.txt` e in `<run>.meta.json` (`vram_used_mib`, `vram_gate_mib`). |
| 1 | `short_abba.sh` | **Campionatore** ogni 10 s: utilizzo GPU, `memory.used`, RAM disponibile e working set del nostro motore in `quiet.log`. Hit (→ `INVALID`): un processo estraneo (come prima); mentre il nostro motore gira, RAM disponibile + suo working set < RAM_GB − 1; `memory.used` più di 3 GB sopra la base della corsa. |
| 1 | `analyze_short.py` | Legge `v4_tier_prewarm vram_experts=` e `target_slots`. **INVALID** se `target_slots` differisce tra le corse, o se `vram_experts` manca in qualcuna o differisce di più del 5%. |
| 2 | `short_abba.sh` | **Banda morta della RAM tolta.** 16 solo con ≥ 16,3 GB disponibili (RAM_GB + 0,3, quello che chiede il gate di partenza), attesa fino a 10 min, altrimenti 13. Se il gate prima di una corsa resta chiuso per 15 min scrive `OUT/INVALID` e si ferma **prima** di avviare il motore (prima partiva lo stesso, con il budget che l'OS dava). |
| 3 | `analyze_short.py` | **Spread sulle medie di corsa.** La regola del 2% vale per la media di tok/s su tutti i prompt usati, una per corsa e per braccio. Lo spread del singolo prompt è solo un avviso stampato, sopra il 3,5%. |
| 4 | `analyze_short.py` | **KEEP B** richiede anche ogni media di coppia ≥ 0; **NOT ADOPTED** basato sull'IC richiede anche ogni media di coppia < 0. Altrimenti INCONCLUSIVE, con il motivo. NOT ADOPTED per token diversi resta senza condizioni. |
| 5 | `short_abba.sh` | **Dump.** Cancellati all'uscita solo se `OUT/tokens.json` contiene ogni dump di questa esecuzione con lo stesso sha256; altrimenti restano e il log dice dove. Se `$DUMP/<run>.logits` esiste già, lo script rifiuta di partire (prima dei wait, e di nuovo prima della corsa) e non lo tocca. |
| 6 | `analyze_short.py` | Un set senza coppie A/B stampa **INVALID: no A&lt;i&gt;/B&lt;i&gt; pair** invece del `ZeroDivisionError` (ogni smoke `--seq B1`). |
| 7 | `run_mirror_abba.sh` | Il wait prima del lock include il gate VRAM (tramite `--check`; `VRAM_MAX_GB` passa a `short_abba.sh`). Attesa totale configurabile con `WAIT_HOURS` (default 10 h; `DEADLINE_EPOCH` resta). Lo smoke resta com'era: riga "Matching mirror container active", riga "read split over 2 drives", ≥ 10% dei byte letti dall'OS su ciascun drive. |
| - | `cut_tokens.py` | `--pairs 1,3` tiene solo quelle coppie; stampa lo spread delle medie di corsa, lo spread massimo per prompt e le medie di coppia. |
| - | `check_analyze.py`, `check_guards.sh` | Nuovi controlli senza motore (vedi sotto). |

**Tre scelte diverse dalla lettera della revisione, con i dati:**
- **RAM nel campionatore.** "RAM disponibile < RAM_GB − 1" letto alla lettera scatta sempre: la macchina ha 31,2 GB e il motore a 16 GB ne tiene ~15, quindi durante una corsa l'OS ne dà ~3-5. Conta quindi come disponibile anche il working set del nostro motore (`tasklist`): l'hit scatta quando un altro processo prende più di (disponibile all'avvio − 15) GB.
- **Base della VRAM.** "La lettura all'avvio del motore" non può essere quella prima del motore (il nostro motore riempie la VRAM fino alla riserva: A ~28 GB, B ~31 GB). Nemmeno quella a READY: nella corsa a 13 GB, B2 passa da 28,6 GB (READY + 3 s) a 31,9 GB, cioè +3,3 GB solo suoi. La base è il primo campione dopo il primo prompt finito (w0); il render di app 3D in A2 (28,2 → 31,9 GB, +3,7) resta un hit. Limite: con la VRAM piena (bracci WNR2 ~31 GB su 32) WDDM può spostare la memoria di un'app estranea, che allora conta meno della sua dimensione.
- **Tolleranza 5% su `vram_experts`.** L'uguaglianza esatta renderebbe INVALID anche i set buoni: 05:40 963-965, W3 934-966 (3,3%), 13 GB 1015-1029. 1 GB di VRAM in meno vale ~60 expert (~6%). `target_slots` invece deve essere identico.

**Controlli eseguiti:**
- **W3 a 64 token, sottoinsiemi di 2 coppie** (`cut_tokens.py --pairs`):

| coppie | B vs A, IC effetti fissi | spread medie di corsa A / B | spread max per prompt | medie di coppia |
|---|---|---|---|---|
| 1,2 | -1,82 ± 0,66 | 0,59 / 0,39 | 1,98 (A code1) | -1,36 / -2,29 |
| 1,3 | -1,79 ± 0,57 | 0,86 / 0,04 | 1,62 (A code1) | -1,36 / -2,22 |
| 1,4 | -1,52 ± 0,45 | 1,37 / 1,00 | **2,93** (A code1) | -1,36 / -1,68 |
| 2,3 | -2,25 ± 0,50 | 0,27 / 0,43 | 1,41 (B code1) | -2,29 / -2,22 |
| 2,4 | -1,99 ± 0,34 | 0,77 / 1,39 | **2,71** (B reason2) | -2,29 / -1,68 |
| 3,4 | -1,95 ± 0,42 | 0,50 / 0,96 | **2,49** (B reason1) | -2,22 / -1,68 |
| tutte e 4 | -1,89 ± 0,31 | 1,37 / 1,39 | 2,93 | tutte < 0 |

  - La vecchia regola per prompt (2%) scattava in 3 sottoinsiemi su 6; quella sulle medie di corsa in 0 su 6 (massimo 1,39%). L'avviso al 3,5% non scatta mai (massimo 2,93%). Tutti gli IC escludono lo 0 e tutte le medie di coppia sono < 0: il verdetto resta NOT ADOPTED.
- **`analyze_short.py` sui set salvati:**
  - 05:40: invariato, **NOT ADOPTED** -2,69 ± 0,38%, medie di coppia -2,82 / -2,57, `vram_experts` 963-965, 9 slot in tutte;
  - 13 GB (175141): **INCONCLUSIVE** (stalli; spread delle medie di corsa A 4,61%, B 3,85%, con il solo code1), più l'avviso per prompt;
  - W3 completo (256 token): ora **NOT ADOPTED** -1,80 ± 0,25%, tutte le medie di coppia < 0 (prima era INCONCLUSIVE per lo spread di A code1 3,13%); spread delle medie di corsa A 1,85%, B 1,34%;
  - un set con solo B1: **INVALID: no A&lt;i&gt;/B&lt;i&gt; pair** (la versione precedente andava in `ZeroDivisionError`).
- **`short-abba/check_analyze.py`**: 10/10 casi sintetici (KEEP B; media ≥ 0 con una coppia < 0 → INCONCLUSIVE; NOT ADOPTED; IC sotto 0 con una coppia ≥ 0 → INCONCLUSIVE; spread delle medie di corsa; avviso per prompt con KEEP B; `vram_experts` -6,7% → INVALID e -3,6% → valido; `target_slots` diversi → INVALID; nessuna coppia → INVALID).
- **`short-abba/check_guards.sh`**: 17/17, con le funzioni prese da `short_abba.sh` e `nvidia-smi`, `tasklist` e RAM finti: gate VRAM nel controllo "quieto"; gate di partenza scaduto → `INVALID`, nessun `meta.json`, motore non avviato (per RAM 10 < 16,3 e per VRAM 6000 MiB); dump esistente → rifiuto, file intatto; `meta.json` con `vram_used_mib`; campionatore senza hit prima della base, +3000 MiB nessun hit, +3100 MiB hit VRAM, 0,5 + 14,0 GB < 15 hit RAM; il controllo dei dump (`COVERED`) dice sì solo con tutti gli sha uguali.
- `bash -n` sui tre script; `short_abba.sh --check` in sola lettura (gate a 4 GB: quieto con 2,3 GB; a 1 GB: "GPU memory used 2252 MiB >= 1024 MiB (VRAM gate)"); il parsing del driver su quell'output vede il blocco VRAM.

## Cosa è stato eseguito

| quando | cosa | lock | note |
|---|---|---|---|
| 05:20:35-05:24:52 | `record_trace.sh` ARM=wex1, w0 + code1, chat_it1, reason1 × 128 token, 16 GB | heavy.lock | traccia `replay/traces/wex1-3x128.*` |
| 05:25-05:36 | `ab_replay.sh` (verifica 64 + A1 B1 B2 A2, `--skip 15`) | heavy.lock | `replay/runs/ab1` |
| 05:36:31-05:40:50 | `record_trace.sh` ARM=wnr2, stessi prompt | heavy.lock | `replay/traces/wnr2-3x128.*`; poi `validate/pad_trace.py` → `.pad12887.trace` |
| 05:40:51-05:58:31 | `short_abba.sh`, 16 GB (19,9 GB disponibili) | heavy.lock | `short-abba/runs/20261109-054051` |
| 05:58-06:00 | `variants.sh` | heavy.lock | interrotto: alle 06:00 l'utente ha messo in pausa i test (`moves.log`) |
| 17:17:51-18:27:01 | `validate/run_validate.sh` (un solo lock) | heavy.lock | ab2, 8 varianti, short ABBA a 13 GB |

- La prima tornata (05:20-06:00) è di questo stesso passo di validazione, interrotto dalla pausa. I suoi dati sono completi e li riuso: stessa build, stesso env, stessi container. Gli spostamenti di file di NOTE-3 erano già finiti alle 05:20:12.
- Build e helper uguali a W3 in tutte le esecuzioni:
  - `ws-ab\bin\deepseek_v4.exe` `7a6f3e3f…`;
  - `coli_cuda_dsv4_dg.dll` `b1b7cdb5…`;
  - `w1\build\kaira_wex_cuda.dll` `cceb4fb5…`.
- Env = `ab/run_w3.sh` (GPU-direct 2, batch WNR2, head GPU). Container su U: con marker `.verified`.
- Il lock delle 17:17 è stato preso libero (l'ultimo detentore era "wf-team5 gpu-v1", 17:15:55). Il trap l'ha rilasciato alle 18:27:01 dopo aver controllato che fosse ancora il nostro. Alle 18:27:04 l'ha preso "colibri-pr v4-union-lease-cap" (il timestamp 18:25:34 nell'etichetta è quello del suo avvio).

## Passo 1: trace reale (WEX1, 16 GB)

`replay/traces/wex1-3x128.trace.json`. Il gemello WNR2 è `wnr2-3x128.trace.json`.

| | WEX1 | WNR2 |
|---|---|---|
| token di decode (forward) | 396 | 396 |
| letture/token (domanda + prefetch usati + prefetch sprecati) | 12,78 + 65,40 + 67,84 = 146,0 | 17,81 + 77,91 + 40,27 = 136,0 |
| hit in RAM/token | 42,7 | **20,7** |
| letture del trace / contatore del motore all'uscita (prefill incluso) | 57.826 / 73.692 = **0,785** | 53.847 / 82.600 = **0,652** |
| VRAM miss/token: trace (confine di token) / moeprof del motore | 120,9 / 87,2 | 116,4 / 21,3 |
| attesa MoE del motore, ms/token (media, p50, p95) | 270,5, 259,3, 437,5 | 287,6, 274,4, 426,0 |
| recall del predittore (= `pf_share`) | 0,767 | 0,767 |
| pacchetti persi dal tap | 0 | 0 |

- **VRAM miss:** non confrontabili 1:1. Il motore conta come hit anche gli expert promossi in VRAM durante il token (`V4_PROMOTE_VRAM=2`, copy worker), e per WNR2 anche ogni record GPU-direct. Il trace vede solo la residenza al confine di token. Per il replay contano le letture.
- **Letture, WEX1:** il 21,5% mancante è il prefill dei 4 prompt più le riletture dentro il token.
- **Letture, WNR2:** il trace legge *meno* di WEX1 (136 contro 146 per token), mentre il motore legge l'11-12% *in più*. I record GPU-direct entrano ed escono dalla VRAM dentro il token senza passare dalla RAM, e il RESID non li vede. È il limite principale del registratore.
- **Calibrazione `pad_trace.py`:** aggiunge (82.600 − 53.847) − (73.692 − 57.826) = **12.887** letture sprecate, sparse in modo uniforme. Ipotesi: prefill e riletture invisibili sono uguali nei due bracci. Così i byte di decode tornano col contatore: 66.734 × 16,9 MB ≈ 1.128 GB, contro i 1.397 − 268 GB del motore.
- **Ripetibilità del set di letture del motore** (stessi prompt, esecuzioni diverse): WEX1 73.692 / 73.823 / 73.670; WNR2 82.600 / 82.641 / 82.304. Spread ≤ 0,4%.

## Passo 2: replay ABBA WEX1 contro WNR2 sullo stesso trace

`replay/runs/ab1` (05:25, `--skip 15`, 381 token) e `replay/runs/ab2` (17:17, `--skip 0`, 396 token). A = WEX1 a lane per record, B = WNR2 `--batch`, 8 lane su domanda e 8 thread di prefetch.

| | ab1 A | ab1 B | ab2 A | ab2 B | B vs A (ab2) |
|---|---|---|---|---|---|
| ms/token, media (esecuzione 1 / esecuzione 2) | 424,75 / 424,52 | 385,22 / 386,57 | 427,02 / 425,67 | 385,55 / 386,14 | **-9,5%** (ab1: -9,1%) |
| ms/token p50 / p95 | 420,3 / 575,8 | 383,1 / 518,9 | 422,1 / 575,1 | 383,1 / 517,8 | |
| attesa dei layer, ms/token | 424,3 | 381,8 | 425,9 | 381,4 | -10,5% |
| MB/token | 2.748 | 2.473 | 2.745 | 2.471 | -10,0% |
| lettura GB/s (wall) | 6,47 | 6,41 | 6,44 | 6,40 | -0,5% |
| latenza di lettura su domanda p50 / p95, ms | 19,3 / 25,8 | 11,3 / 19,6 | 19,3 / 25,8 | 11,1 / 19,5 | -42% |
| decodifica (B) o H2D (A), ms/record | 0,61 | 0,52 | 0,73 / 0,64 | 0,54 | |
| quota della decodifica o H2D coperta dalle letture | 93% | 90% | 92-94% | 89% | |
| spread tra esecuzioni | 0,1% | 0,3% | 0,3% | 0,2% | |

- **Verifica:** 64/64 record WNR2 decodificati dall'helper (API singola e batch) identici byte per byte a WEX1, in entrambe le tornate. Fallimenti: 0 in tutte le esecuzioni.
- **Ripetibilità tra sessioni** (12 h, macchina diversamente carica):
  - B 385,89 contro 385,85 ms/token;
  - A 424,6 contro 426,4 ms/token. Lo 0,4% include i 15 token di w0, che ab2 conta e ab1 no.
- **L'H2D di WEX1 varia** tra 0,61 e 0,79 ms/record da un'esecuzione all'altra (stato di GPU e PCIe), ma resta sotto le letture: sui ms/token pesa meno dello 0,5%.

## Passo 3: replay contro esecuzione reale (stesso braccio)

Reale = `validate/real_cost.py --ids w0,code1,chat_it1,reason1`: medie per token di decode di moeprof (wait, attach, gpu) e decprof (hc, attn, moe). Le fonti sono le esecuzioni di registrazione e lo short ABBA delle 05:40 a 16 GB.

| braccio | reale wait | reale wait+attach+gpu | reale token (hc+attn+moe) | replay | replay contro reale |
|---|---|---|---|---|---|
| WEX1, registrazione | 270,5 | 370,1 | 465,3 | A senza calcolo: 426,4 ms/token | **+15,2%** su wait+attach+gpu |
| WEX1, short A1/A2 | 279,1 | 377,8 | 464,4 | stesso | **+12,9%** |
| WEX1 con `--layer-ms 4.9` (A_l) | | | | attesa 284,9; token 482,2 | attesa **+5,3% / +2,1%**; token **+3,6%** |
| WNR2, registrazione | 287,6 | 367,9 | 477,4 | B sul trace WEX1: 385,9 | +4,9% |
| WNR2, short B1/B2 | 290,0 | 369,8 | 476,2 | B_pad: 440,4 | +19,7% / +19,1% |
| WNR2 con `--layer-ms 4.9` (B_padl) | | | | attesa 255,2; token 460,3 | attesa **-11 / -12%**; token -3,6% |

- **Entro il 20%: sì,** per tutte le varianti. Il caso peggiore è B_pad, al limite (+19,7%).
- `--layer-ms 4.9` = (decprof totale − attesa moeprof) / 40 dell'esecuzione di registrazione. Per WEX1 porta l'attesa simulata al 2-5% di quella reale.
- **Segno della differenza WNR2 − WEX1:**

| fonte | attesa MoE, ms/token | tempo per token |
|---|---|---|
| reale, registrazione | **+17,1** | +12,1 ms (+2,6%) |
| reale, short ABBA 16 GB (appaiato) | **+11,1 ± 7,6** | tok/s -2,69% |
| reale, W3 (results.md) | **+10,5** | tok/s -1,80% |
| replay, stesso trace (ab1/ab2) | -44,6 | -40,5 ms (**-9,5%**): segno sbagliato |
| replay B_own (trace registrato su WNR2) | -75,3 | -69,8 ms: segno sbagliato |
| **replay B_pad (trace WNR2 calibrato)** | **+8,3** | **+14,0 ms (+3,3%): segno giusto** |
| replay B_padl contro A_l (con `--layer-ms`) | -29,7 | -21,9 ms: segno sbagliato |

**Lettura.**
- In un regime limitato dal disco (il motore legge a ~5,6 GB/s medi su tutta l'esecuzione, il drive ne dà 6,5) il tempo per token segue i byte letti.
- Il trace calibrato di WNR2 ha +15% di letture × record -10% = +3,8% di byte, e il replay dà +3,3%.
- `--layer-ms` è uno spin sulla CPU, quindi la decodifica rANS del replay ha la GPU tutta per sé. Nel motore la decodifica divide la GPU e l'unico copy engine con attenzione e MoE:
  - attenzione 60,8 → 66,1 ms/token in registrazione, 56-57 → 66-67 nello short ABBA, 73,6 → 82,1 in W3;
  - calcolo MoE GPU 94 → 79 ms/token (niente attach).
- Il replay non ha questa parte, perciò con `--layer-ms` WNR2 sembra più veloce di quanto sia.
- **Dopo la revisione:** il "segno giusto" di B_pad non è una prova. B_pad è calibrato sul contatore del motore (circolare). Il replay differisce dal motore in buffer, batch, percorso host, drain dello stream e domanda per layer (vedi Riassunto). Questa tabella descrive il replay, non il motore.

## Passo 4: short ABBA contro W3

| misura | B vs A tok/s (IC 95%) | attesa B − A, ms/token | token | durata |
|---|---|---|---|---|
| **short ABBA 05:40, 16 GB** (`short-abba/runs/20261109-054051`) | **-2,69 ± 1,85%** (n=6) | +11,1 ± 7,6 | identici, logits bit per bit (`0a3eb335…`) in A1 B1 B2 A2 | **17 min 40 s** |
| W3 tagliato: code1, chat_it1, reason1, primi 128 token (`validate/cut_tokens.py`) | -2,44 ± 1,50% (n=12) | +15,1 ± 6,0 | identici | (3 h 07 min per tutto W3) |
| W3, stessi 3 prompt a lunghezza piena (256 o EOS) | -3,59 ± 1,08% (n=12) | +20,5 ± 6,1 | identici | |
| W3 completo, 8 prompt (results.md) | -1,80 ± 0,65% (4 coppie) | +10,5 | identici | 3 h 07 min |
| **Dopo la revisione** (IC con i prompt come effetti fissi): short ABBA 05:40 | **-2,69 ± 0,38%** (df=3; medie di coppia ± 1,55) | +11,1 ± 7,9 | identici | |
| W3, nuovo set (7 prompt, primi 64 token, `cut_tokens.py`) | **-1,89%**: ± 0,31 effetti fissi, ± 0,70 medie di coppia | +14,0 ± 2,4 | identici | |
| W3, stessi 7 prompt, primi 256 token | -1,99%: ± 0,28 effetti fissi, ± 0,63 medie di coppia | +12,8 ± 2,1 | identici | |

Gli IC "± … (n=…)" delle prime righe trattano ogni delta come indipendente. Sono stati sostituiti: i delta di uno stesso prompt condividono il livello di quel prompt.

**Per prompt (primi 128 token)**, delta per coppia:

| prompt | short ABBA | W3 |
|---|---|---|
| code1 | -4,84 / -5,08 | -3,5 / -5,8 / -4,5 / -5,6 |
| reason1 | -1,62 / -1,05 | -2,1 / -3,1 / -3,3 / -2,8 |
| chat_it1 | -1,95 / -1,59 | +0,7 / -0,6 / +0,3 / +1,1 |

- chat_it1 è l'unico prompt che differisce di circa 2 punti. In W3 parte dopo code2 (ordine w0, code1, code2, chat_it1, …), quindi con una cache calda diversa.
- **Stesso verdetto di W3:** WNR2 più lento, IC sotto lo 0, token identici → WNR2 non adottato.
- **Come screen, lo short ABBA è più pessimista del W3 completo** (-2,7 contro -1,8%), perché manca chat_it2, dove WNR2 vince del 7%. Un risultato al limite, con l'IC che tocca lo 0, va confermato con W3.
- `cut_tokens.py` sulle esecuzioni dello short ABBA ridà esattamente `analyze_short.py` (-2,69 ± 1,87%): il taglio a 128 token è coerente.

**Rilancio serale, 13 GB** (`short-abba/runs/20261109-175141`, 17:51-18:27, **35 min 18 s**). Non è una misura valida di ripetibilità:
- **RAM:** 14,4 GB disponibili alla decisione, quindi `RAM_GB=13` per tutte le esecuzioni (`head=streamed-bf16`, 7 slot per layer). Da B1 in poi erano disponibili 18-20 GB.
- **Macchina occupata.** Campioni ogni 20 s in `validate/busy-shortabba-13gb.log`:
  - gcc/make di `ws.py make ws-ramfix-test` dalle 17:52 alle 17:57 (durante A1), e `ws.py run ws-prefix … gates.sh quick` dalle 17:37;
  - B2/reason1: due token fermi 28 s e 16 s nel MoE, verso le 18:11:20 (causa non identificata: attesa I/O);
  - A2/reason1: render di app 3D (partito alle 18:17:23) con la GPU al 90-100%: 126 token su 127 sopra 1 s, 0,238 tok/s.
- **Risultato grezzo:** +43% ± 111% (n=6) e verdetto "KEEP B". È falso: viene da un outlier di +259% su reason1.
- **Senza reason1** (prompt puliti, n=4): **-0,41% ± 2,61%**, attesa +12,1 ms/token. Con reason1 della coppia 1 (+1,28%), la media su n=5 è -0,07%.
- **Con il nuovo `analyze_short.py`:** **INCONCLUSIVE**. Il controllo degli stalli toglie reason1 (A2 fino a 25 s per token, B2 28,4 s) e anche chat_it1 (un token da 4,0 s in B2). Resta code1: -0,62 ± 4,78% (df=1), con spread nello stesso braccio del 4,6% (A) e del 3,9% (B).
- Per confronto, ab-samedrive (13 GB, build G2 più vecchia) sugli stessi 3 prompt a 128 token dà +1,92% ± 1,15% (n=6). A 13 GB il vantaggio di WNR2 sulla head letta dal disco e il suo costo sulla GPU della build integrata sembrano compensarsi [I]. Con n=4 e la macchina occupata non è conclusivo.
- **Token e logits identici** anche qui: lo stesso sha `0a3eb335…` delle esecuzioni a 16 GB. I logits non dipendono dal tier di RAM.

## Passo 5: varianti del replay

Dal passo 2, il percorso di I/O di WNR2 *nel replay* non è il colpevole:
- latenza su domanda -42%;
- GB/s uguali;
- decodifica coperta dalle letture all'89-94%.

(Dopo la revisione: nel motore invece lo è. PROF `edisk` dà 15,7-16,0 contro 13,7 ms per lettura, con 5,5-5,7 contro 4,4 letture in volo. La premessa di questo passo vale solo per il replay.)

Ho quindi provato 3 varianti sul **set di letture** e sul **calcolo**, non su lane, `--split` o allineamento. Due esecuzioni ciascuna, interlacciate (A_l1 B_own1 B_pad1 B_padl1 B_padl2 B_pad2 B_own2 A_l2, 17:28-17:51), `--skip 0`, file in `validate/runs-variants/`.

| variante | trace | letture/token (D / P usati / sprecati) | MB/token | ms/token (es. 1 / es. 2) | attesa | rispetto a | Δ |
|---|---|---|---|---|---|---|---|
| A (ab2) | wex1 | 12,78 / 65,40 / 67,84 | 2.745 | 427,0 / 425,7 | 425,9 | | |
| B_own | wnr2 così come registrato | 17,81 / 77,91 / 40,27 | 2.301 | 356,6 / 356,6 | 350,7 | A | -16,4% |
| **B_pad** | wnr2 + 12.887 sprecate | 17,81 / 77,91 / 72,81 | 2.851 | 440,4 / 440,3 | 434,2 | A | **+3,3%, attesa +8,3 ms** |
| A_l | wex1, `--layer-ms 4.9` | come A | 2.745 | 483,2 / 481,2 | 284,9 | | |
| B_padl | B_pad, `--layer-ms 4.9` | come B_pad | 2.851 | 463,2 / 457,3 | 255,2 | A_l | -4,5%, attesa -29,7 ms |

- **B_pad dà lo stesso segno del reale.** Anche la grandezza è vicina: +8 contro +11/+17 ms/token di attesa, +14 contro +12 ms/token.
  - ~~Causa riprodotta: WNR2 legge più record (i GPU-direct saltano la RAM), non record più lenti.~~ **Ritirato dopo la revisione.** B_pad è calibrato sul contatore di letture del motore, quindi il segno è costruito: è circolare. Nel motore le letture di WNR2 sono anche più lente (+2-2,3 ms per lettura, coda più lunga).
  - ~~La correzione va nella politica del motore … Non nel formato né nelle code.~~ **Ritirato.** La causa resta un'ipotesi. Le cause concorrenti, da provare con short ABBA che cambiano solo l'env, sono `V4_NEXT_PREFETCH_THREADS` 4 contro 8, `COLI_WEX_GPU_DIRECT_PREFETCH_RANK` e `COLI_WEX_RAM_CACHE_GB` > 0. Il calo degli hit in RAM (42,7 → 20,7 per token) resta un indizio. `ram_skips` invece no: conta gli usi del tier RAM, non i record che lo saltano.
- **B_own non basta:** il registratore perde le riletture dentro il token, quindi un trace registrato su WNR2 va calibrato sul contatore del motore.
- **B_padl mostra il limite del replay:** senza contesa sulla GPU, la decodifica sembra gratis.
- Ripetibilità delle varianti: B_own e B_pad 0,01%, A_l 0,4%, B_padl 1,3%.

## Tempi (wall)

| strumento | durata |
|---|---|
| `record_trace.sh` (3 prompt × 128 + w0, 16 GB) | 4 min 17 s (WEX1), 4 min 19 s (WNR2) |
| `ab_replay.sh`: verifica 64 record + ABBA | 10 min 47 s (ab2), circa 10 min 45 s (ab1) |
| una esecuzione di `replay.exe` | 2 min 22 s - 2 min 55 s senza calcolo; 3 min 02 s - 3 min 11 s con `--layer-ms 4.9` |
| 8 varianti | 23 min |
| `short_abba.sh` a 16 GB | **17 min 40 s** (esecuzioni 17 min) |
| `short_abba.sh` a 13 GB | 35 min 18 s (una esecuzione disturbata da 823 s; pulito sarebbe circa 28 min) |
| W3 | 3 h 07 min (solo ABBA) |

## Problemi trovati

1. **Il registratore sottostima le letture di WNR2** (0,652 del contatore del motore contro 0,785 per WEX1). Il RESID al confine di token non vede i record GPU-direct che entrano ed escono dalla VRAM dentro il token. Senza calibrazione (`pad_trace.py`), un trace registrato su WNR2 dice che WNR2 legge meno di WEX1, mentre il motore legge il 12% in più.
2. **Il replay su un trace condiviso misura solo il formato.** Dà WNR2 -9,5% e il segno sbagliato rispetto a W3. Non modella né la politica dei tier (RAM saltata dal GPU-direct) né la contesa sulla GPU (attenzione +5/+10 ms/token).
3. **`--layer-ms` gira sulla CPU,** quindi favorisce WNR2 (decodifica senza concorrenza sulla GPU).
4. **`analyze_short.py` non aveva un controllo per gli outlier.** Il rilancio a 13 GB, con uno stallo da 28 s e un render di app 3D, ha dato "KEEP B" a +43%. **Corretto:** controllo degli stalli e dello spread, verdetto INVALID/INCONCLUSIVE (Revisione, punti 2-3).
5. **`short_abba.sh` non controllava che la macchina fosse quieta.**
   - Senza `--no-lock` controllava solo che non girasse un altro `deepseek_v4.exe`.
   - heavy.lock non copre build, gate e lavoro esterno a KAIRA (app 3D, gcc).
   - Qui gli ho passato io `--no-lock`, perché il lock lo teneva `run_validate.sh`: nessun controllo, quindi.
   - **Corretto:** attesa della macchina quieta prima di partire e campionatore ogni 10 s durante le esecuzioni, anche con `--no-lock` (Revisione, punto 4).
6. **La regola RAM si decideva una volta, all'avvio, senza attesa.** Un calo temporaneo di RAM (14,4 GB alle 17:51; 18-20 GB pochi minuti dopo) manda tutto l'ABBA a 13 GB, che è un'altra configurazione: head letta dal disco, ~1,2 contro ~2,1 tok/s. **Corretto:** attesa fino a 10 min di avere ≥ 15 GB, come `record_trace.sh`, con la decisione registrata (Revisione, punto 5).
7. **Default diverso di `--skip`:** ab1 era a `--skip 15` (solo i 3 prompt), ab2 e le varianti a `--skip 0` (w0 incluso). Ho confrontato ogni gruppo con il reale calcolato sullo stesso insieme di token. L'effetto di w0 sulle medie è sotto lo 0,5%.
8. `<TMP>\cut128.py` è sparito durante il lavoro (cartella temporanea condivisa). Lo strumento ora è `validate/cut_tokens.py`.

## Limiti

- Lo short ABBA a 16 GB è stato eseguito **una volta** (05:40). Il rilancio serale non è confrontabile (13 GB, macchina occupata). La sua ripetibilità è stimata dalle coppie interne (-2,82 / -2,57%) e dai due ABBA dentro W3 (-2,39 / -2,48%), non da una seconda sessione a 16 GB.
- **La calibrazione di B_pad usa il contatore di uscita del motore, che include il prefill.** L'ipotesi "prefill e riletture uguali nei due bracci" non è misurata. Una riga `wexprof` per token o per prompt nel motore la renderebbe misurabile; non ho toccato i sorgenti.
- La quota domanda/prefetch del trace è un modello (`pf_share` = recall 0,767), non una misura.
- Le NOTE-2/3/4 della sessione principale (braccio safetensors, mirror U:+C:, M+R) non fanno parte di questo passo e **non sono state eseguite**.
  - Gli script le supportano: `short_abba.sh --a/--b`, e per il mirror `--app-engine` con `--b-env 'COLI_WEX_EXTRA_MIRRORS=C:\kaira_wex\v41-reap COLI_WEX_WEIGHTS=38,62'`.
  - Correzione: la ripartizione viene da `COLI_WEX_WEIGHTS`, non da `COLI_DISK_WEIGHTS`. Senza `COLI_WEX_WEIGHTS` tutte le letture vanno su U:.
- **Dump dei logits** in `%LOCALAPPDATA%\Temp\short-abba\` (1,6 GB, esecuzioni 054051 e 175141): ancora presenti, perché sono serviti a rianalizzare queste due esecuzioni. Non li ho cancellati io. Da ora `short_abba.sh` cancella i dump che crea, dopo l'analisi.
- **Lo short ABBA nuovo (7 prompt × 64 token) non è ancora stato eseguito.** Durata, ripetibilità e spread tra esecuzioni dello stesso braccio, sotto la soglia del 2%, sono da misurare. short1 finisce in EOS dopo 8 token, quindi il suo tok/s è il più rumoroso.
- Nessuna modifica a `src/`, `deepseek_v4.c`, `app/` o `kaira.json`. Nessun commit. Nessuna scrittura su U:.

## File

- `validate/run_validate.sh`: driver della tornata serale (un lock: replay ABBA, 8 varianti, short ABBA). Log in `validate/run_validate.log`.
- `validate/real_cost.py`: costo reale per token dai log del motore. `validate/cut_tokens.py`: delta appaiato sui primi K token (confronto con W3).
- `validate/pad_trace.py`: calibrazione del trace WNR2. `validate/variants.sh`: versione della prima tornata, interrotta.
- `validate/runs-variants/*.json|txt`, `replay/runs/ab1`, `replay/runs/ab2`, `replay/traces/*`.
- `short-abba/runs/20261109-054051` (16 GB, valido) e `short-abba/runs/20261109-175141` (13 GB, contaminato).
- `validate/busy-shortabba-13gb.log`: campioni di build e GPU durante il rilancio.
- `short-abba/short_abba.sh`, `analyze_short.py`, `serve_bench.py` (scrive `<run>.pid`), `validate/cut_tokens.py`: versioni dopo la revisione. Le precedenti sono in `*.bak`, accanto, e in `VALIDATION.md.bak`.
- `mirror-abba/bin/`: copie dell'engine installato, con sha256 uguale ad app\ (scritte da `short_abba.sh --app-engine`).
