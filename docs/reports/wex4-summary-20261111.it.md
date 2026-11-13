# WEX4: rapporto breve (2026-11-11)

> **Note of 2026-11-13.** This report records 2026-11-11. Its word "Privato" (private) below is of that date: since 2026-11-13 the repository is licensed under the Apache License 2.0, to be made public. The current specification is [WEX4-SPEC.md](../WEX4-SPEC.md).

> **Dated report (2026-11-11).** Its "Stato" line was written before the install; WEX4 is in production since that day, as the end of "Cosa serve per il deploy in KAIRA" says. Superseded by [WEX4-SPEC.md](../WEX4-SPEC.md) and, for what runs today, by [STATUS.md](../../STATUS.md).

Formato successivo del container KAIRA. Privato: **niente di questo va su JustVugg/colibri**.

**Stato:** spec + prototipo CPU su un branch di lavoro di questo repo (unito a `main` con 54e688b). In KAIRA non è cambiato niente: motore, helper CUDA, `app\kaira.json` e container di produzione sono intatti. Il deploy richiede una decisione esplicita.

- Spec: [WEX4-SPEC.md](../WEX4-SPEC.md)
- Prototipo: `c/wex4.h`, `wex4/wex4_store.h` (loader lato motore), `wex4/wex4_tools.c`, `wex4/wex4.py`, `wex4/wex4_convert.py`
- Test: `sh wex4/build_and_test.sh` (meno di 2 minuti, senza GPU e senza modello reale)
- Misure e ricerca: `wex4/research/` (script e risultati in ogni cartella; verifiche in `wex4/research/VERIFICATION.md`). Revisione avversaria: `wex4/review/REVIEW.md`

## In una frase

WEX4 è un container auto-descrittivo come safetensors (nomi, dtype, shape di qualsiasi tensore), ma compresso senza perdita, con hash su tutto, allineato a 4096 byte e con un indice binario che il lettore C valida sul posto senza parser JSON. I record WNR3 restano **identici byte per byte**: stessa lettura, stesso decoder GPU; velocità attesa uguale, non ancora misurata nel motore.

## Cosa è meglio di safetensors

| | safetensors | WEX4 |
|---|---|---|
| Integrità | nessun checksum: un bit corrotto viene caricato in silenzio | hash per stream nel decoder, SHA-256 per record, SHA-256 su indice e header (una radice di 32 byte identifica tutto il file) |
| Compressione | nessuna | codec per record: WNR3 per gli expert MXFP4, WBP1 per FP8/BF16/F32, raw come ripiego; un record non è mai più grande del raw arrotondato a 4096 byte |
| Header | fino a 100 MB di JSON da analizzare | pagina fissa da 4096 byte + indice binario a righe fisse, validato sul posto, senza campi facoltativi né ordine libero |
| Versione | nessun campo | major, minor, feature obbligatorie e facoltative, con regole per i lettori |
| Allineamento | 8 byte | ogni record su 4096 byte (letture non bufferizzate, mirror su più dischi) |
| File poliglotti / dati nascosti | corretto dopo un audit | dimensione esatta nell'header coperto da hash, nessun byte fuori da un'area sotto hash, niente dati in coda (restano aree libere: metadati, blob sconosciuti, dati dei tensori) |
| Ritorno a safetensors | - | `unpack` ricostruisce i file sorgente con lo **stesso SHA-256** (non per i container canonical-zero, che sono esatti nei valori e non nei byte) |

Cosa resta meglio in safetensors: è uno standard pubblico con molti lettori e i tensori si possono usare con mmap senza decodifica.

## Cosa è meglio di WEX3/WNR3

- **Generale:** non solo expert. Ogni tensore ha nome, dtype e shape; un expert è un record con 6 tensori.
- **Più sicuro:** indice e header sotto hash; il record ripete owner, id del modello e flag value-exact, e l'indice li ricontrolla; limiti su ogni conteggio; nessuna allocazione guidata dal file.
- **Mirror:** due copie sono lo stesso container se hanno la stessa pagina di header (4096 byte), non serve confrontare tutta la tabella.
- **Migrazione senza riscrivere 123 GB:** upgrade sul posto, reversibile (vedi sotto).
- **Decoder CPU 6-30 volte più veloce**, senza cambiare formato.

## Numeri misurati

Su dati reali, in sola lettura, sul desktop, salvo dove indicato. Dettagli e limiti nella spec, sezioni 11, 12 e 15.

| Cosa | Risultato |
|---|---|
| Indice WEX4 del container di produzione (7.680 record, 46.080 tensori) | 6,7 MB: **+0,005 %** su 123,3 GB. Validazione in C: **27 ms** |
| Tabella degli expert ricostruita dall'indice | identica byte per byte a quella di produzione |
| 4 expert impacchettati dai safetensors con il modello di produzione | record **identici** a quelli in produzione |
| Shard BF16 reale (970 MB) | **0,6625** del raw; file ricostruito con SHA-256 identico |
| Tensori non-expert di un layer (FP8 + F32 + BF16, 168 MB) | **0,817**; ricostruzione identica |
| Tutti i tensori non-expert del modello (11,1 GB, oggi non compressi) | **circa 8,5 GB, -2,6 GB** (stima da campione, verificata) |
| Decode CPU di un record WNR3, riferimento | 505 ms su 1 thread |
| Decode CPU, nuovo decoder (stessi byte, stessi verdetti) | **72-90 ms** su 1 thread (6-7x), **12-16 ms** su 8 thread; dentro il wrapper attuale 17-18 ms (circa 30x) |
| Lo stesso sul portatile (1 record, non riverificato) | 1.077 ms -> 129 ms (1 thread), 38 ms (8 thread) |
| Decode WBP1 (BF16), C scalare, 1 thread | 161 MB/s |
| Lettura + SHA-256 di tutto il container | circa 2 minuti (stima da 64 record) |
| Fuzzing | **112.399 file mutati (1.105 mutazioni nulle), 0 corruzioni silenziose, 0 crash**; 46 attacchi scritti a mano tutti rifiutati; lettore C e lettore Python d'accordo su 2.042 file mutati |

## Compressione: dove siamo e cosa resta

Richiesta: comprimere ancora di più. Risposta misurata:

- **Expert MXFP4: WNR3-CZ è al limite della sua famiglia di modelli.** Media reale del container 0,85398 del raw. Nessun contesto statico in più sposta più di circa lo 0,05 % (i nibble sono quasi indipendenti una volta nota la classe di scala; il segno è una moneta esatta). Lo "scheletro" del record (lunghezze e stato iniziale degli stream) vale in tutto lo 0,576 % del raw.
- **Un record "WNR4"** (tabella lunghezze codificata + stato iniziale corto + allineamento a 64 byte) farebbe risparmiare **circa 0,22 GB (0,18 %)**. Non giustifica un codec nuovo con nuovo codice GPU e nuovi gate. Non incluso.
- **Chunk più grandi (4096:1024)**: -0,4 GB senza codec nuovo, ma dimezzano gli stream; sui record V4 il decode a record singolo era 30-65 % più lento. Su V4.1 non è misurato.
- **Il guadagno vero è fuori dagli expert:** i tensori non-expert (FP8, BF16, F32) oggi non sono compressi; con WBP1 scendono di 2,6 GB. È spazio su disco, non velocità: si leggono una volta all'avvio.

## Velocità di decode: cosa migliora e cosa no

- **CPU:** il nuovo decoder (`wex4/research/fast-cpu-decode/wex_fast_decode.h`) è il miglioramento concreto. Serve al ripiego CPU, agli strumenti e alla verifica completa: decodificare tutti i 7.680 record passerebbe da circa 65 minuti a 1-2 minuti (stima dai tempi per record, nessun passaggio completo eseguito).
- **GPU:** niente nel formato la rende più veloce. Il decode a batch 16 (0,37 ms/record) è già vicino al tempo di copia PCIe degli stessi byte (circa 0,32 ms), e l'attesa del disco resta la parte più grande del token.
- **Un'idea aperta:** un modello dei pesi più semplice (contesto = sola classe di scala, segno come bit grezzo) decodifica 1,75-2 volte più veloce di un decoder AVX2 semplice, con +0,06-0,08 % di dimensione; ma non è stato confrontato con il nuovo decoder veloce, che sul formato attuale è già più rapido. L'effetto su GPU non è misurato; sarebbe un codec nuovo.

## Cosa serve per il deploy in KAIRA

1. **Store del motore** (`v4_wex_load_manifest`, `expert_store_wex.c`): se il magic è WEX4, validare header e indice con `c/wex4.h`, caricare il modello WNM2 dal blob, riempire la tabella di `WexEntry` con `wex4_expert_entry()`. Tutto il resto (cache, split di lettura, mirror, GPU-direct, batch) vede gli stessi dati di oggi. Il loader esiste come prototipo provato (`wex4/wex4_store.h`: radice fissata, tabella, flag, modello, SHA-256 al primo uso); resta da chiamarlo dal motore, niente nel percorso caldo.
2. **Helper CUDA:** nessuna modifica (stessa `WexEntry`, stessi byte, feature 16).
3. **`app\kaira.json`:** una chiave facoltativa con la radice attesa (64 cifre esadecimali). Nient'altro.
4. **Upgrade del container** (`python -I wex4/wex4_convert.py upgrade FILE`): controlla prima che il risultato sia valido, legge e fa l'hash di tutti i record (circa 2 minuti, stima), aggiunge in coda la vecchia pagina di header e l'indice (6,7 MB), poi riscrive una sola pagina da 4096 byte. I record non si spostano. `downgrade` riporta il file identico a prima, anche se la pagina di header è andata distrutta a metà dell'operazione (provato).
5. **Gate** (spec 13.5): test del motore su un container piccolo aggiornato; upgrade di U: con KAIRA fermo, `verify` completo, stessi token e stessi tok/s della copia WEX3 su C:; poi C:.

Fatto il 2026-11-11: gate M1 e M2 su un worktree KAIRA (branch `wex4-20261111`, commit 75932c1, non unito a master e non installato; patch in `integration/kaira-wex4-loader-75932c1.patch`). Lo store del motore apre lo stesso container in forma WEX4 e serve gli stessi byte; radice fissata con `COLI_WEX_ROOT`, header o indice rovinati, record corrotti o scambiati e percorso masked si comportano come con WEX3. I test esistenti restano verdi. Fatto anche M3 sul container di produzione su U: (dettagli in `research/wex4-m3-20261111/RESULT.md`): upgrade in 94 secondi, verifica completa in 8 minuti (ogni hash, tutti i 7.680 record decodificati su CPU), motore reale con GPU: stessi token e stessi logit, 2,077 contro 2,070 tok/s (+0,3 %, dentro il rumore). Poi `downgrade`: il file ha di nuovo lo SHA-256 originale. Fatto anche M4 e, su decisione dell'utente, **WEX4 è in produzione dal 2026-11-11**: U: e mirror C: in forma WEX4 con la stessa radice, mirror riconosciuto, 3,05 tok/s, logit identici, motore 75932c1 installato (vecchio exe salvato in `<KAIRA>\lab\app-before-wex4-20261111`). Poi (commit 4ba5568, patch `integration/kaira-wex4-first-read-4ba5568.patch`, installato su decisione dell'utente): il motore controlla lo SHA-256 di ogni record alla prima lettura (`COLI_WEX_VERIFY`, acceso di default quando la radice è fissata con `COLI_WEX_ROOT`); un record che non torna viene rifiutato e letto dai safetensors. Misura con mirror e radice fissata, una coppia valida: stessi logit; tempo totale di generazione 212,8 s spento contro 213,0 s acceso (+0,1 %); il costo sta all'avvio (pronto in 26,4 s invece di 20,8 s) e nel primo prompt (+4,9 s), poi nessuna differenza misurabile. Radice fissata nel profilo `v41gpu` di `kaira.json`; vecchio exe e vecchio `kaira.json` in `<KAIRA>\lab\app-before-wex4-verify-20261111`. ABBA completo e avvio da KAIRA non ancora fatti.

## Domande aperte

1. Solo expert nel container (come oggi) o tutto il modello? I non-expert valgono -2,6 GB ma richiedono un secondo percorso di caricamento nel motore.
2. Dove fissare la radice (chiave in `kaira.json`) e se firmarla.
3. Costo a runtime della verifica SHA-256 al primo uso: misurato su una coppia, circa +5 s all'avvio e +5 s sul primo prompt, poi niente. Da confermare con un ABBA completo.
4. Chunk 4096:1024: vale una misura GPU su V4.1?
5. Il modello dei pesi più semplice e più veloce: vale una misura GPU?
6. Gli expert MTP (384, fuori dal container oggi): metterli dentro?
7. Altri motori (qwen38 FP8 e simili): il formato li copre, ma ogni motore ha bisogno del suo store e, per gli expert letti a ogni token, di un kernel GPU per WBP1.

## Incidente durante il lavoro

Una misura ha creato per errore un file temporaneo da 32 GB che ha riempito il disco C: per pochi minuti; è stato cancellato subito e lo spazio è tornato a 32 GB liberi. Nessun file del modello, del container o di KAIRA è stato toccato. La misura è stata rifatta in memoria.
