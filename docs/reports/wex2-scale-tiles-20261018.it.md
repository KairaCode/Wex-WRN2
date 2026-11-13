# WEX v2: compressione senza perdita delle scale MXFP4

> **Dated report (2026-10-18).** WEX2 was written and tested but never built for a whole model ([CHANGELOG.md](../../CHANGELOG.md)). What runs today: [STATUS.md](../../STATUS.md).

Implementazione e verifica locale: **18 ottobre 2026**.

WEX2 mantiene invariati i pesi FP4 E2M1 e comprime le scale UE8M0 in tile di 256 valori. Il lettore ricostruisce lo stesso layout in RAM usato da WEX1 prima di esporlo ai kernel CUDA. Non cambia la quantizzazione e non richiede modifiche ai kernel.

## Risultato verificato

Due esperti reali del layer 10 di DeepSeek V4 Flash, rimappati in una fixture autonoma:

| Contenitore | Dimensione, inclusi header/tabella/allineamenti |
|---|---:|
| WEX1 | 26.746.880 byte |
| WEX2 | 25.415.680 byte |
| Riduzione | **1.331.200 byte, 4,977%** |

Il confronto Python e quello del percorso HOT del motore verificano **ogni byte delle scale e dei pesi** contro i safetensors sorgenti. WEX2 legge 25.407.488 byte di record contro 26.738.688 di WEX1; la differenza rispetto alla dimensione del file è la pagina header più la tabella.

È una misura su due esperti, non una conversione dell'intero modello e non un benchmark di token/s. Il decoder espande le scale prima dell'uso: **il risparmio riguarda file e trasferimenti dal disco, non automaticamente RAM o VRAM residenti**. La latenza aggiuntiva del decoder va misurata nel carico reale prima di attivare V2 come default.

Risultati e comandi ripetibili: KAIRA `artifacts\wex-v2-20261018\` (non in questo repo): `real-sample-results.json` (campione reale), `native-checks.log` (test nativi), `python-tests.log` (test Python).

## Decoder CPU ottimizzato

La seconda iterazione conserva esattamente il formato V1/V2 e i file già prodotti. Riduce il costo della lettura V2 in entrambi i lettori, standalone e HOT:

- Se `base + (2^bits - 1) <= 255`, tutti i residui rappresentabili sono validi: la prima passata verifica header, dimensioni e padding del tile senza riesaminare ogni scala. Le basi più alte conservano il controllo individuale dell'overflow.
- L'espansione usa gruppi di otto valori con larghezze note al compilatore, da 1 a 7 bit; a 8 bit e base zero usa una copia. Il tile finale parziale resta limitato ai suoi byte effettivi.
- La validazione completa precede ancora ogni modifica dello slab. Lo scratch contiene soltanto le scale compresse; i pesi non vengono ricopiati. Il codice è C senza intrinseche o nuove dipendenze.

Misura su Ryzen 9 7950X3D, GCC 16.2, `-O3 -march=x86-64-v3`, sugli stessi due esperti reali:

| Decoder | Mediana del tempo CPU per esperto |
|---|---:|
| Prima | 1.119,932 µs |
| Ottimizzato | 63,789 µs |
| Accelerazione | **17,6×** |

Sono quattro coppie di prove in ordine alternato, tre round per prova, 1.000 iterazioni per ciascun esperto e round. La mediana considera le dodici misure per decoder. Il tempo include ripristino del prefisso compresso, allocazione, validazione ed espansione; **esclude l'I/O disco e l'inferenza**. Non misura un aumento dei token/s. Ogni prova confronta tutte le scale e tutti i pesi con WEX1.

Verifica aggiuntiva: **9.216 combinazioni** di larghezza/base/lunghezza con encoder di test indipendente, più **30.000 stream** validi e malformati confrontati con il decoder precedente, senza differenze di accettazione o dei byte decodificati. Anche un overflow nell'ultimo tile viene rifiutato prima di modificare lo slab. Store, percorso HOT, fallback per corruzione e suite DeepSeek V4 sono stati rieseguiti con successo.

Il [benchmark C](../../c/tools/bench_wex_decode.c) legge al massimo due record compressi e accetta `RAW.wex COMPRESSED.wex [iterations] [rounds]`. Per ripetere il confronto locale con la versione precedente conservata:

```powershell
C:\Python314\python.exe artifacts\wex-decode-20261018\verify_decode.py
```

Risultati completi e hash dei decoder in KAIRA `artifacts\wex-decode-20261018\` (non in questo repo): `decode-results.json`, `native-checks.log` (test nativi), `standalone-tests.log` (test standalone), `fallback-results.json` (fallback).

## Compatibilità

- Il lettore aggiornato accetta WEX1 e WEX2. Il writer continua a produrre V1 senza opzioni aggiuntive.
- WEX2 è esplicito: `--format-version 2`. Il nome predefinito è `v4_flash_experts.v2.wex`, separato dal file V1.
- V2 usa un **magic differente**: i vecchi lettori ignoravano versione e flag, quindi cambiare soltanto il numero di versione avrebbe consentito una lettura errata dei dati compressi.
- I vecchi lettori rifiutano il nuovo magic; il percorso planned può ripiegare sui safetensors. Per usare effettivamente V2 occorre il runtime aggiornato e `COLI_WEX_PATH` impostato al nuovo file.
- Il binario completo con il decoder ottimizzato è stato costruito separatamente in KAIRA `artifacts\wex-decode-20261018\runtime\deepseek_v4.exe` (non in questo repo). La prova di avvio senza modello è in `launcher-smoke.log` nella stessa cartella. Non è stato distribuito al posto del runtime installato.

## Formato su disco

Gli interi sono little-endian. Header e tabella iniziano a offset multipli di 4096; anche offset e dimensione di ciascun record sono allineati a 4096.

L'header strutturato rimane di **40 byte**, dentro una pagina di 4096. Magic/versione sono `0x57455831/1` o `0x57455832/2`. `aligned_expert_bytes` descrive la massima capacità dello slab **decodificato**. V2 usa il bit 0 dei flag header per il codec scale e conserva un UUID di generazione ai byte 40–55 della pagina. L'UUID identifica la copia per la verifica delle mirror; non è un checksum dei pesi.

Le entry rimangono di **72 byte**:

- `total_bytes`: dimensione allineata del record **su disco**;
- `scale_bytes`, `weight_bytes`, shape e offset: layout **decodificato**, con scale w1/w2/w3 seguite dai pesi w1/w2/w3;
- `flags=0`: record raw, come V1;
- `flags=(scale_pages << 8) | 1`: record con scale compresse; `scale_pages` indica la dimensione del prefisso su disco in pagine da 4096.

Un record compresso contiene:

```text
[prefix 12 byte][stream scale][padding fino a scale_pages*4096]
[pesi FP4 originali][padding finale a 4096]
```

Il prefix è `<III>`: magic `0x31534357`, numero di scale decodificate, numero di byte dello stream. Ogni tile contiene `base:uint8`, `bits:uint8` e i residui `scala-base` impacchettati LSB-first. Sono supportati 0–8 bit; 0 descrive un tile costante. L'ultimo tile può avere meno di 256 valori.

Si usa il codec soltanto quando le scale raw occupano un numero intero di pagine e la dimensione allineata del record diminuisce realmente. Gli altri record restano raw, anche all'interno di un file V2.

## Lettura senza ricopiare i pesi

La destinazione della singola lettura Direct I/O è:

```text
slab + scale_bytes_raw - scale_pages*4096
```

La destinazione resta allineata. I pesi arrivano direttamente al loro offset finale; il decoder salva in scratch soltanto lo stream delle scale e lo espande all'inizio dello slab. Evita così un `memmove` dell'intero payload dei pesi. I contatori del percorso HOT riportano i byte del record compresso effettivamente letto; la capacità della cache resta quella decodificata.

## Scrittura e verifica

Il repacker controlla dtype, shape, dimensioni dei tensori, duplicati, intervalli sovrapposti, letture corte e metadata WEX. Il file nuovo viene scritto in un temporaneo univoco nella stessa directory, sincronizzato, validato e pubblicato con `os.replace`. Non cancella anticipatamente il container precedente. Anche la mirror viene pubblicata tramite un temporaneo; il suo nome coincide con quello del file primario.

I lettori controllano magic/versione, flag, dimensioni, layout canonico, capacità e limiti del file prima delle letture. Le mirror devono avere stessa dimensione, intera pagina header e tabella identiche. I tile malformati vengono rifiutati prima di espandere lo slab. Il percorso HOT ripiega sui safetensors se il manifest WEX è incompatibile o la decodifica fallisce.

## Uso

Esempio sul piccolo campione già verificato, senza scrivere file:

```powershell
C:\Python314\python.exe app\tools\repack_v4_wex.py --model-dir artifacts\wex-v2-20261018\real-sample --format-version 2 --dry-run
```

Per il modello completo usare `--model-dir` e `--output` espliciti, con un volume che possa contenere **l'intero nuovo file contemporaneamente al precedente**. `--dry-run` legge tutte le scale ma nessun payload dei pesi; sul modello completo non è una scansione dei soli header. Le disponibilità di spazio rilevate durante lo sviluppo non consentivano una seconda copia completa su C: o U:, quindi non è stata avviata una conversione completa.

Il modulo autorevole è [`c/tools/repack_v4_wex.py`](../../c/tools/repack_v4_wex.py).

## Verifiche eseguite

- **19 test Python:** codec, roundtrip V1/V2, record misti/raw fallback, dry-run, nomi mirror e preservazione del file precedente in caso di errore.
- **Codec C:** larghezze 0–8, tile parziali, byte noti, overflow, metadata e stream malformati, pesi invariati.
- **Store standalone:** lookup, cache, lease, eviction, prefetch e verifica mirror su file sintetici V1/V2.
- **Percorso HOT reale:** due esperti reali, V1/V2, confronto completo dei byte, contatori I/O, cache/prefetch; fallback per header e stream corrotti in KAIRA `artifacts\wex-v2-20261018\fallback-results.json` (non in questo repo).
- **Suite DeepSeek V4 esistente:** superata, comprendendo store, prewarm, pool di prefill, layout rows16, KV, layer e matematica.
- **Build Windows completa isolata:** 31 oggetti compilati dai sorgenti correnti, incluso il decoder ottimizzato; manifest in KAIRA `artifacts\wex-decode-20261018\runtime\build-result.json` (non in questo repo).

Il compendio fornito dall'utente è stato usato come riferimento per distinguere compressione su disco e rappresentazione residente, e per mantenere i trasferimenti allineati. Questa implementazione usa Direct I/O Windows; non introduce un percorso Microsoft DirectStorage o un decoder GPU.
