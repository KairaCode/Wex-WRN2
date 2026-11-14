# WNR2 → WNR3: ricerca e sviluppo sul formato (DeepSeek V4.1 REAP-192 MXFP4)

> **Dated report (2026-11-07).** Research done before the WNR3 decisions. The format that was built took only part of it: superseded by [WNR3-SPEC.md](../WNR3-SPEC.md), whose section 0 lists what was left out.

Stato al 2026-11-07. Dati reali: 160 expert V4.1 (4 per layer × 40 layer, 3,0 GB) scaricati con HTTP Range dal repo pinnato
`patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4 @ 4ab649b2` in `<RESEARCH-PC>\v41-sample`. Tutte le misure sono
held-out (modello addestrato su metà degli expert, costo misurato sull'altra metà) salvo dove indicato. Numeri grezzi e script:
[wnr3-research-measurements-20261107.md](wnr3-research-measurements-20261107.md). Il lavoro è diviso in quattro team:

- team 1: ricerca online su 7 fronti, 60 candidati, verifica delle fonti e verifica tecnica;
- team 2: sicurezza dello zero, audit delle misure, prototipo GPU;
- team 3: spec WNR3, implementazione di riferimento, modello tok/s;
- team 4: validazione del ceil-form.

A questi si aggiungono le mie misure.

Legenda: **misurato** = numero ottenuto sui dati reali; **letto** = da codice o fonte citata; **stima** = derivato o modellato.

La bozza di spec aggiornata è [WNR3-DRAFT-v2.1.md](../../research/wnr3-20261107/team4/spec-update/WNR3-DRAFT-v2.1.md) (DRAFT 0.2.1; l'originale è in `<RESEARCH-PC>\team4\spec-update\`). Contiene il ceil-form, l'ancora fissa, (a) e (b) spostati in appendice e il profilo dello zero canonico. Il §0.3 della spec elenca tutte le modifiche e il §0.4 dice quali regole sono state davvero eseguite in un codec.

La 0.2.1 corregge tutti gli errata del critic del team 4 (12 punti più 4 extra), e poi 40 problemi trovati da tre revisori indipendenti, tutti di gravità media o bassa:
- il default è ora l'hash per stream (CEIL|LENC|HASH, flag 0x1C, 0,88021 del raw); TAIL+DIGEST è un profilo opzionale, ammesso solo dopo l'implementazione del digest e un audit di corruzione;
- i tok/s vengono dalla simulazione di capacity-update, con l'ipotesi G2+W2 dichiarata;
- c'è WNR2-g0 come passo zero;
- le regole mai eseguite sono marcate *[I, untested]*;
- la dipendenza da un'unica linea di codice è dichiarata, con un test di cross-decode bloccante.

## In sintesi

- **Ceil-form.** Il 30,6% dei blocchi V4.1 ha i codici raddoppiati: è un artefatto dell'export di DeepSeek (quantizzazione ceil e poi scala ricavata di nuovo). Riscriverli in forma canonica è byte-exact.
  - Record rANS reali held-out: −1,79% del raw, ratio 0,8815 contro 0,8994.
  - GPU: byte-exact anche dentro un decoder GPU (RTX 3070), con tempo di decodifica pari o migliore grazie ai byte in meno.
  - È generale: lo stesso artefatto c'è in tutti i DeepSeek V4.x e in gpt-oss.
- **Contesti di modello.** Quasi tutto il guadagno trovato prima nei contesti (bit di block max, K modelli, pm2) imitava questo artefatto o il salto dell'ancora srel 120/121. Sopra il ceil-form vale ~0, e il bit di block max peggiora.
- **Profilo byte-exact proposto.** Ceil-form + ancora fissa + tabella lunghezze codificata + hash: **0,8802 del raw** contro 0,8994, cioè +2,2% di record nello stesso spazio (+13,6% contro WEX1 invece di +11,2%).
- **Profilo value-exact.** Con lo zero canonico sopra si arriva a **0,8528 del raw**, cioè +5,5% di record rispetto a WNR2 (+17,3% contro WEX1).
  - Il −0 → +0 non cambia nessun bit di output nei kernel CPU e CUDA di default (0 bit diversi su decine di milioni di output, sm_86).
  - Mancano i percorsi tensor-core della 5090 e la decisione del proprietario sulla regola delle logprob.
- **Integrità.** Oggi lo 0,26–0,63% dei bit-flip singoli passa inosservato e diventa un peso con il segno invertito. Un hash nello stato iniziale rANS lo chiude per +0,007% di byte.
- **tok/s** (modello, non misura).
  - Il profilo byte-exact vale +1,4% (U:) / +2,3% (E:) rispetto a WNR2: non è dimostrabile con un A/B.
  - Il profilo value-exact vale +3,5% / +5,9%.
  - W2 e la scelta del drive contano di più del formato.
  - Spazio: nessun formato entra nei 122,6 GB liberi su U: riportati dal reference. A WNR3z mancano 0,61 GB.

## Classifica delle modifiche

Le dimensioni sono record rANS **reali**, codificati e decodificati: team 4, ceil-ref, 40 expert held-out (uno per layer), modelli addestrati su 80 expert, IC bootstrap al 95%. Le percentuali sono del record raw (18.800.640 B). Base WNR2 = 0,89943.

| # | Modifica | Tipo | Effetto | Costo di decodifica | Stato |
|---|---|---|---|---|---|
| 1 | **Ceil-form (f) + ancora fissa 120 (g0)** | byte-exact | **−1,79%** (IC −1,99…−1,64), ratio 0,8815 | +0…2% kernel (3070, patch differita); 0 B di shared | validato su CPU (736 record) e GPU (3070); mancano la 5090 e il codec del motore |
| 2 | **Zero canonico (c)** | value-exact | −2,74% in più sopra f; f+g0+c+d+e = **0,8528** (−4,66%) | 0 | 0 bit diversi su CPU e CUDA sm_86; mancano i tensor-core sm_120 e la decisione sulle logprob |
| 3 | **Hash nello stato iniziale (e)** | robustezza | +0,007% (+1.352 B/record) | qualche moltiplicazione per simbolo (non misurato su GPU) | chiude il buco dei flip di segno silenziosi |
| 4 | **Tabella lunghezze codificata (d)** | byte-exact | −0,137% (−25,8 KB) | un warp-scan per gruppo | misurato su CPU |
| 5 | **WNR2 con ancora fissa (solo encoder)** | byte-exact | ~−0,24…−0,29% (stima entropica) | 0, il decoder legge già il byte mediana | **da verificare**: mai codificato |
| 6 | Tabella compattata a 259 righe | solo decoder | 0 B | 0,65–0,87× (3070) | misurato |
| 7 | Slot dei tier in byte | ingegneria | +3–6 punti di capacità | 0 | calcolato su V4; massimo record V4.1 da misurare |
| 8 | Sign bypass (layout 2) | velocità | 0 B | 0,59–0,93× best vs best (3070) | **incompatibile con (c)** (il piano dei segni fisso ridà indietro il guadagno), salvo un piano compattato |
| — | Tail seeding al posto dell'hash | byte-exact | −0,221% | — | sconsigliato per ora: integrità non misurata (sezione 3) |

Profilo byte-exact consigliato **f+g0+d+e = 0,8802** (−1,92%), capacità contro WNR2 +2,2%. Profilo value-exact **f+g0+c+d+e = 0,8528**, capacità contro WNR2 +5,5%.

## 1. Ceil-form: la scoperta principale (byte-exact)

**Cosa si vede nei dati** (misurato su 160 expert, 176.947.200 blocchi da 32):
- La magnitudine massima del blocco è il codice 6 nel 44,8% dei blocchi e il 7 nel 55,2%; non è mai sotto 6.
- Il **30,65%** dei blocchi ha max 7 e nessuna magnitudine 1 o 3. Usa quindi solo i codici 0, 2, 4, 5, 6, 7, cioè i valori 0, 1, 2, 3, 4, 6: sono codici raddoppiati.
- Per caso questo succederebbe tra lo 0,02% e lo 0,8% dei blocchi: 1,9·10⁻⁴ con le quote globali, fino a 0,65–0,81% con le quote dei soli blocchi max-7.

**Perché succede** (meccanismo simulato e letto nel codice; team 4, ceil-origin):
1. I pesi stanno sulla griglia della regola *ceil*, s = 2^ceil(log2(amax/6)): `fast_round_scale` in `inference/kernel.py` di DeepSeek, e il default `ROUND_UP` di `triton_kernels`.
2. All'export la scala viene ricavata di nuovo dai valori dequantizzati; ceil e floor OCP danno gli stessi byte. Un blocco ceil con massimo 3·s ha amax/6 = s/2, una potenza di due esatta. Il secondo passaggio sceglie quindi s/2 e raddoppia tutti i codici.
3. In simulazione esce esattamente questa forma: 22,3% di blocchi raddoppiati su dati gaussiani (pari alla quota dei blocchi ceil con max 3,0), max sempre 6 o 7, e valori identici.

**Da dove viene** (misurato): il checkpoint ufficiale `deepseek-ai/DeepSeek-V4.1-Flash-1017 @1bae0e1` ha già i blocchi raddoppiati.
- I 4 expert tenuti da REAP che sono stati confrontati sono byte-identici al nostro campione.
- Gli 8 expert potati da REAP hanno il 27,8–31,6% di blocchi raddoppiati, l'expert MTP il 19,7%.
- Il repack REAP copia byte per byte: lo dicono il model card e `ops/s05_surgery.py` della pipeline.

**È generale** (misurato su 9 checkpoint, ~72 M blocchi, sempre 0 blocchi con max < 6 e 0 errori di round trip):

| Checkpoint | Blocchi raddoppiati | Guadagno sul record (held-out) |
|---|---|---|
| DeepSeek V4.1-1017 upstream | 29,4% | −2,14% |
| DeepSeek V4-Flash | 26,0% | −1,15% |
| DeepSeek V4-Pro | 23,9% | −0,99% |
| gpt-oss-20b / 120b | 21,9% / 21,4% | −1,69% / −1,75% |
| AMD Quark gpt-oss-120b | 21,5% | −1,81% |
| PTQ: amd/DeepSeek-R1-0528-MXFP4, RedHatAI/GLM-5.3-MXFP4 | 0,005% / 0,003% (caso) | ~0 |

Il ceil-form è quindi una feature riutilizzabile per **ogni checkpoint MXFP4 nato da un training FP4 QAT** e per i suoi derivati: DeepSeek V4.x, gpt-oss, e i re-export che mantengono la griglia. Sui checkpoint PTQ non costa niente e non rende niente.

**La trasformazione** (byte-exact):
- **Encoder:** un blocco "pari" (max 7, nessun 1 o 3) diventa magnitudini dimezzate (0, 2, 4, 5, 6, 7 → 0, 1, 2, 3, 4, 5, segno invariato) e scala + 1.
- **Decoder:** ogni blocco con max decodificato 5 viene raddoppiato, e la sua scala diminuita di 1.
- **Condizione:** è una biiezione finché nessun blocco salvato ha max < 6. L'encoder deve verificarlo **per matrice** e avere un fallback con flag. Un MXFP4 scritto con un solo passaggio ceil avrebbe ~22% di blocchi con max 5 (simulazione), che senza il gate verrebbero decodificati male: per esempio `quantize_to_mxfp4` di transformers, che usa `ROUND_UP` di default. Tra i 9 checkpoint misurati non ce n'è nessuno di questo tipo.
- **Modello:** va riaddestrato sui dati trasformati, con la mediana calcolata sulle scale trasformate.

**Quanto rende:**
- **Entropia held-out** su 160 expert, misurata da me e dal team 2 con codice indipendente (3,7084 contro 3,7085 bit/w):
  - pesi 3,7824 → 3,7084 bit/w (−1,96%);
  - scale 1,005 → 0,917 bit/scala;
  - record −339,5 KB = **−1,81% del raw**, ratio ~0,882;
  - capacità vs WEX1 ~+13,4% invece di +11,1% (capacità = 1/ratio).
- **Record rANS reali** del prototipo GPU (8 expert, modelli in-sample): da 16.963.218 a 16.573.694 B/record, cioè da 0,90227 a 0,88155 del raw: −2,30% (pesi −2,21%, scale −12,4%).
- **Record reali held-out** (team 4, ceil-ref: 40 expert, 736 record codificati e decodificati, 0 fallback, 0 errori):
  - f da solo: 16.578.765 B = 0,88182 del raw, **−1,761%** (IC −1,948…−1,632; mediana per expert −1,656%).
  - Il delta reale coincide con la stima a 5 B/record: la codifica reale non perde niente rispetto all'entropia. Il −1,81% citato sopra è sui 160 expert; i 40 di test valgono −1,761%.
  - Ancora fissa 120 (g0) sopra f: −6,0 KB. La regola "coppia adiacente più densa" (g) sul ceil-form è neutra (−0,9 KB, IC ±16 KB), perché salta una volta su una matrice quasi in parità.
  - f+g0: 0,88152 (−1,791%).
- **Contesti extra sopra il ceil-form** (misurato): non pagano più.
  - pm2: −0,0005%;
  - bit di block max + metà blocco: +0,06%, cioè peggio;
  - posizione nel blocco: −0,02% con una tabella da 185 KB.

  Il ceil-form assorbe le feature (a), (b) e pm2: erano solo modi indiretti di riconoscere i blocchi raddoppiati.
- **Con lo zero canonico sopra** (value-exact): 3,5920 bit/w (−5,03%), record **−4,55% del raw**, ratio ~0,854, capacità ~+17%.

**GPU** (misurato sulla RTX 3070, team 4 ceil-gpu):
- **Correttezza:** con l'inverso nel kernel di decodifica, l'uscita è byte-exact contro i byte safetensors **originali** di pesi e scale. Vale per tutti i 138.240 stream, in ogni configurazione di lancio e nella pipeline scale → pesi con la scala corretta in place. Controllo negativo: con l'inverso spento sbagliano 138.240 stream su 138.240.
- **Costo:** dipende da come si scrive la scala corretta.
  - Una store di un byte per ogni blocco trasformato costa +1…+20%.
  - La **patch differita** costa da −2,4% a +3,3% rispetto agli stessi stream senza inverso, ~0 nelle configurazioni migliori. Ogni lane tiene una maschera a 64 bit dei blocchi trasformati (due shift/or per blocco) e riscrive le 4 word da 16 B di scale una volta sola, a fine stream.
- **Tempi:** con meno byte da leggere il ceil-form è pari o più veloce di A259.
  - 1 record: 566/557 µs contro 589 µs (0,96/0,94×).
  - 8 record: 4.120 contro 4.162 µs (0,99×).
  - 1 matrice: +1,4%.
  - Con sign bypass: −1,1…+3,8% rispetto a D.
- **Risorse:** registri +0…8, shared memory invariata, 0 local memory.

**Il team 4 ha provato a smontarlo** (ceil-refuter, misurato). Non ci è riuscito, ma ha corretto tre cose.

- **Copertura.**
  - 236 record controllati: i 160 del campione, 70 expert routed in più che coprono tutti i 40 layer, e 6 expert MTP.
  - Su 261 M blocchi: 0 con max sotto 6, 0 con max salvato 5, 0 tutti-zero. Round trip: 0 errori su 708 matrici. Scale tra 112 e 126 (mai 0, 254 o 255).
  - Il coder reale ha decodificato esattamente 17 record su 17, con e senza hash.
  - Fallback attesi sui 7680 record: 0 (limite superiore al 95%: ~100).
- **Condizione esatta della biiezione.** La condizione è "nessun blocco salvato con max == 5", non "max < 6". Saltando la trasformazione quando la scala è 255, T diventa totale: il test esaustivo su blocchi da 4 nibble dà 0 collisioni.
- **Il guadagno si divide in due parti.**
  - Circa 0,27 punti del −1,8% vengono dal fissare l'ancora srel, non dal ceil-form. Sulla forma di oggi la mediana salta 120 → 121 in 43 matrici su 690; in forma ceil solo in 2.
  - Ad ancora fissa da entrambi i lati, il ceil-form vale −1,59…−1,60% del raw.
- **Alternative con un side symbol.**
  - Il miglior modello senza ceil-form prende il 97% dello stesso guadagno: side symbol a 3 classi + metà blocco + ancora fissa + scala condizionata alla classe. Un solo bit "blocco pari" ne prende l'86%.
  - Costano però un simbolo in più per blocco e tabelle più grandi. Il ceil-form tiene la tabella da 259/567 righe e nessuno stream in più, e sopra quel modello aggiunge ancora −0,055%.
  - È quindi il modo **più economico** di prendere questo guadagno, non l'unico.
- **Expert MTP.**
  - Con il modello degli expert routed si espandono (ratio 1,110 nella forma originale, 1,079 in quella ceil), perché le loro scale w1/w3 stanno a 113–114.
  - Servono modelli propri e un'ancora coerente (regola "coppia(originale) + 1"); così tutti e 3 i layer MTP guadagnano −1,25…−1,53%.
- **Integrità.** Senza hash, il ceil-form allarga il buco dei flip di segno su w2: da 0,29% a 0,58% di flip accettati. Non c'è amplificazione: con la regola x == L ogni flip accettato resta un solo segno invertito.
- **GPU.** La scala finale si conosce solo dopo aver decodificato il blocco di pesi, e va scritta in un buffer di output separato. Correggere in place il buffer `const __restrict__` / `__ldg` delle scale è undefined behaviour e rende i retry non idempotenti.

## 2. Zero canonico (WNR2z / feature c, value-exact)

- **Misurato:** +0 e −0 sono esattamente equiprobabili: 5,8255% e 5,8263% di tutti i pesi. In ogni matrice la quota è 0,499–0,501 degli zeri. Il segno dello zero è quindi una moneta e costa 0,1165 bit/w di puro rumore.
- **Misurato, record reali** (implementazione di riferimento del team 3, 40 expert held-out): lo zero canonico da solo fa −3,045% del record WNR2 (IC 95% −3,050; −3,041), cioè 0,87204 del raw. Sopra il ceil-form la stima è −5,03% dei bit dei pesi e −4,55% del record raw (ratio ~0,854).
- **Sicurezza: letto e misurato dal team 2**, con due passate indipendenti, una delle quali cercava di confutare.
  - I kernel CPU di colibri decodificano già il codice 8 come +0.0f (o come int8 0). Per loro il cambio non tocca nemmeno gli input decodificati:
    - `deepseek_v4.c:20123-20129`;
    - `quant.h:1284`, `:1370`, `:1392`.
  - I kernel CUDA FP32 di default (`backend_cuda_dsv4.cu:99`) e gli shader Vulkan decodificano −0.0f. Però tutti gli accumulatori partono da +0 con arrotondamento al più vicino. In IEEE, sommare uno zero di qualunque segno lascia l'accumulatore identico bit per bit.
  - **Misurato** confrontando i record originali con quelli canonicalizzati:
    - kernel di produzione copiati verbatim su CPU: 0 bit diversi su 3,1 M output per kernel (tutti i 160 expert), più 350 k output per kernel con 9 set di attivazioni avversari;
    - kernel CUDA reali di colibri sulla RTX 3070: 0 bit diversi su 14,0 M output per kernel (56 M per i kernel batched a 4 token), con riesecuzioni deterministiche;
    - i controlli positivo e negativo scattano entrambi, quindi l'harness vede le differenze quando ci sono.
  - **Non testato** (serve la 5090):
    - i percorsi tensor-core sm_120: DeepGEMM sm120 FP4 e cuBLASLt MXFP8, dove il codice 8 diventa E4M3 0x80;
    - Vulkan su altri driver.

    L'unico effetto teorico lì è il segno di un output esattamente zero. Nei 40 expert controllati non esiste nessun blocco né riga tutto-zero, e top-k, argmax ed Engram di colibri trattano già −0 come +0.
- **Cosa si rompe davvero** (team 2), tutto fuori dall'aritmetica:
  - il gate di deploy V4.1 (`sample_records.py` + `test_decode_batch`) confronta byte per byte gli slab decodificati con WEX1;
  - i test di decodifica esatta;
  - sha256 e marker `.verified` di W3;
  - l'accoppiamento dei mirror.

  Inoltre un container WNR2z **non può ricreare i byte originali**: cancellare WEX1 o safetensors dopo averlo creato è irreversibile.
- **Proposta, da decidere.** La regola attuale chiede un confronto delle logprob prima della produzione. Il team 2 propone di sostituirlo con due test: un bit-test originale vs canonico sulla 5090 per ogni kernel che il motore può usare, e un memcmp dei logit a ogni passo end-to-end. Se entrambi danno 0 byte diversi, WNR2z è bit-identico per costruzione. I tool di verifica vanno adattati per confrontare con lo slab canonicalizzato.

## 3. Integrità: hash nello stato iniziale rANS

- **Buco reale scoperto dall'audit.** Oggi la corruzione si rileva solo con gli invarianti strutturali del rANS: lunghezza esatta e stato finale == L. Questo controllo accetta in silenzio lo 0,26–0,63% dei bit-flip singoli casuali. Ogni caso accettato è un peso con il segno invertito: f[s] = f[s^8] e i due inizi CDF differiscono di esattamente 8192 = P/2, quindi lo stato non cambia.
  - Misurato su 720 stream reali (ans-prior-art) e su prove ripetute (audit).
  - Le mie 40 prove erano troppo poche per vederlo.
- **Rimedio misurato.** Si inizializza l'encoder a x0 = L + h23, dove h23 è un hash a 23 bit di identità (modello, layer, expert, indice stream) e simboli decodificati. Il decoder verifica che stato finale == L + h23(decodificato).
  - Costo: +0,067–0,070 B/stream, cioè +1.364,6 B/record = +0,0073% (record reali del team 3).
  - Il buco del segno si chiude: 104/104 casi rilevati; 0 accettati su 9000.
  - Copre anche la sostituzione di un record intero valido, se l'identità entra nell'hash.
  - Hash: MurmurHash3_x86_32, verificato sul valore SMHasher 0xB0F57EE3. Usa solo moltiplicazioni e rotate, quindi niente tabelle in shared memory. Le GPU NVIDIA non hanno istruzioni CRC, e un CRC32C da tabella costerebbe il 3–6% del kernel (stima).
- **Correzione di un mio test.** Il risultato "stream valido decodificato all'indice sbagliato: regola di oggi 0/40" era impostato male, perché riusava i contesti dello stream originale. Con i contesti reali dell'indice sbagliato la regola di oggi rileva 24.000/24.000. Lo spostamento dentro un record è quindi già coperto; l'hash serve per il segno e per l'identità del record.
- **Precedenti** (letto): JPEG XL fissa lo stato iniziale "used as CRC" (`libjxl ans_params.h`, ANS_SIGNATURE 0x13); Duda §3.7 propone di mettere informazione nello stato iniziale; Bloom 2015 descrive il "minimal flush". Nessuna fonte trovata mette nello stato x0 un hash dei dati decodificati.
- **Conflitto da decidere.** Gli stessi 23 bit possono portare invece gli ultimi k = 5 codici in chiaro (tail seeding): −0,24% del record, con un digest a 32 bit del record nel campo riservato dell'header (byte 28..31).

  | Opzione | Contenuto dei 23 bit | Byte | Rilevazione |
  |---|---|---|---|
  | (a) | hash per stream | +0,007% | 2^-23 per stream, con localizzazione |
  | (b) | tail seeding + digest del record | −0,24% | 2^-32 per record, senza localizzazione |
  | (c) | split, es. 12 bit di hash + 4 codici | ~−0,12% | intermedia |

  A costo zero si può anche riordinare l'alfabeto CDF come 2·mag+segno: l'accettazione silenziosa scende di 10× (0,058%; audit).
- **Raccomandazione: hash per stream (a) come default v1**, ed è il default della spec 0.2.1. La spec 0.2 proponeva TAIL+DIGEST, ma il critic del team 4 l'ha contestata per quattro motivi:
  - il digest non è implementato da nessuna parte;
  - il tasso di accettazione con il solo TAIL non è mai stato misurato;
  - con TAIL i bit bassi dello stato finale contengono i simboli di coda, quindi una corruzione lì passa il controllo;
  - con il ceil-form, una magnitudine di coda cambiata può far raddoppiare o dimezzare per sbaglio un blocco intero di 32 pesi (dedotto, non misurato).

  Il tutto per 41,5 KB/record, che valgono +0,16% (U:) / +0,25% (E:) di tok/s, sotto la risoluzione di un A/B. L'hash invece è misurato: +1.352 B/record e 104 flip su 104 rilevati.

## 4. Ingegneria a costo quasi zero

- **Tabella delle lunghezze codificata** (misurato).
  - Le lunghezze degli stream sono molto concentrate: std 6,7–9,9 B, range 939–1060 B per stream da 2048 pesi.
  - La tabella u16 da 38.880 B/record scende a 13,1 KB nei record reali del team 3: −25.754 B = −0,137% del raw, IC stretto. Usa residui bit-packed in gruppi da 32.
  - La variante della spec WNR3, con base u32 ogni 32 stream, ha offset calcolati da ogni warp con un warp-scan: niente kernel in più e niente prefix-sum host. Pesa 15,8 KB (−0,123%).
  - Con l'entropy coding ideale sarebbe −0,148%.
- **Fix dell'ancora srel** (misurato, team 3 expert-stats).
  - Le scale stanno quasi tutte su due valori adiacenti, 120 e 121.
  - In 27 matrici su 480 (5,6%) la mediana salta a 121 e tutte le etichette srel scorrono di uno. Quelle matrici costano +0,148 bit/w. Questo spiega l'anomalia del layer 15 (una sola w3 "flippata" in L15_E025), gli altri layer cattivi (24, 30, 32, 34 e 37–39, dove ce ne sono 20 su 27) e tutto il guadagno dei cluster K = 2–8.
  - Ancorare al valore basso della coppia adiacente più densa dà −0,257% held-out, con un solo modello e nessuna informazione aggiuntiva. Dopo il fix la dispersione per expert scende da 0,026 a 0,0105 bit/w.
  - Sopra il ceil-form la regola "coppia più densa" è neutra. Basta un'ancora costante 120 (g0): −6,0 KB, misurato su record reali.
  - **Occasione immediata da verificare.** WNR2 salva già il byte mediana per record (reference, righe 200 e 225). Un encoder che scrive 120 al posto della mediana, con un modello riaddestrato, potrebbe quindi prendere ~0,24–0,29% del raw **senza cambiare il decoder**. Bisogna prima controllare che niente ricalcoli la mediana. Non è mai stato codificato.
  - Attenzione: se il byte di ancora non coincide con quello di addestramento del modello, si perde ~10% del raw. Il builder deve controllarlo.
- **Slot dei tier RAM/VRAM dimensionati in byte**, non sul record massimo (letto e calcolato).
  - Sui numeri V4 si passa da +8,4% a +14,65% di record per budget fisso: 3–6 punti di capacità senza toccare il codec.
  - Su V4.1 la dimensione massima del record non è ancora misurata.
- **Tabella compattata a 259 righe** (solo decoder, stessi byte). Sono raggiungibili solo le righe con pm ≤ mx più la riga "none".
  - La shared memory scende da 18.160 B a 8.304 B.
  - Sul 3070 il tempo è 0,865× per 1 record e 0,785× per 8 record a 32 thread, 0,65× e 0,68× nelle configurazioni migliori (misurato).
- **Stato iniziale piccolo** x0 = 2^8 + h8, in alternativa: −0,184% (misurato da ans-prior-art). È compatibile con stream separati, non con stati interleaved.

## 5. Velocità di decodifica (prototipo RTX 3070)

Il team 2 ha costruito un prototipo CuPy+NVRTC (sm_86) che decodifica byte-exact stream reali WNR2-like di 8 expert, in ogni variante e configurazione: 138.240 stream. Non include il ring cp.async del motore. Tutti i numeri sono **misurati sul 3070** e non si trasferiscono 1:1 alla 5090.

| Variante | Byte | Tempo vs A |
|---|---|---|
| A (come produzione) | — | 302 µs/matrice, 881 µs/record, 5,80 ms per 8 record (22,7 GiB/s), ~300 cicli/simbolo |
| A259 (stessi byte) | = | 0,99× matrice; 0,865× record; 0,785× 8 record (best 0,65×/0,68×) |
| D: sign bypass (CDF 8 simboli + segni in chiaro) | −0,0001% | 0,926× matrice; 0,586× record; 0,641× 8 record |
| DB2: D + 2 catene di blocchi | +0,36% | 0,47× record; 0,53× 8 record (41,5 GiB/s) |
| B1/B2: 2 stati su un solo byte stream | +0,36% | ~1,0× (il byte window condiviso serializza) |
| B2s / DB2s: 2 stati, sub-stream separati | +0,565% | 0,726× / 0,633× latenza su singola matrice |
| C: tabella da 33 KB (block-max + half) | −0,34% | 1,9–2,3× più lento a 32 thread; nel caso migliore ≈ A |

- Il segno, dati magnitudine e contesto, vale 0,999999 bit. Il sign bypass è quindi gratis in byte e riduce la tabella da 18 KB a 4 KB. Però il piano dei segni in chiaro non è protetto da nessun invariante rANS, quindi serve l'hash. Inoltre non si combina con lo zero canonico: un piano di segni fisso paga anche i segni degli zeri e annulla il guadagno di (c). Per averli entrambi servirebbe un piano dei segni compattato che salta gli zeri (value-exact, non misurato).
- Il ceil-form non cambia questi rapporti: sopra D costa da −1,1% a +3,8%, sopra DB2 da −2,4% a +3,0% (team 4).
- Il decoder è limitato dall'occupancy. Emulare gli 8 KB del ring porta il tempo a 1,85× (singola matrice), e il numero di registri da solo sposta il tempo di 1,42×. Ogni tabella più grande costa velocità.
- Sul percorso PCIe la decodifica vale ~0 tok/s (0,2 ms/record contro un floor PCIe di ~0,32 ms per 16,9 MB). La velocità conta solo per un tier VRAM compresso, che il modello del team 3 dà come nullo o negativo su U:.

## 6. Quanto vale in tok/s

Modello del team 3 (`<RESEARCH-PC>\team3\capacity-value\`, 400 estrazioni Monte Carlo su 18 parametri). È una **stima**: presuppone G2 (MoE V4.1 su GPU) e W2 (decode WNR2 batched), che non sono ancora costruiti.

- WNR2 vs WEX1 sullo stesso drive U: +6,4% (90%: +3,0…+11,9; P(≥ +3%) = 0,95). Su E: +11,9%. Viene quasi tutto da meno byte NVMe per miss, non dalla capacità: le letture disco calano solo dello 0–2,3%.
- Elasticità: per ogni 1% di byte in meno, +0,65% tok/s su U: e +1,09% su E:.
- Le modifiche byte-exact WNR3 (a+b+d+e) danno +0,3% su U: e +0,5% su E:. Sono sotto il ±0,85% che un A/B ABBA×2 riesce a risolvere: non si vedranno mai in tok/s.
- Lo zero canonico è l'unica modifica di formato che vale tok/s veri: WNR3z vs WNR2 dà +2,4% su U: (1,3–3,8) e +3,9% su E: (3,0–4,6).
- Contano di più due cose che non riguardano il formato:
  - W2 da solo: +3,6% su U:;
  - il drive: WNR2 su E: contro WEX1 su U: fa −52%.
- Spazio: WNR3z (~125,4 GB) non entra nei 122,6 GB liberi su U: riportati dal reference alle 17:11. Le lettere dei drive di questo PC non sono quelle della macchina di produzione.

**Rifatto con i record reali del team 4** (capacity-update; stesso modello, dati reali come delta accoppiati su WNR2 di produzione):

| Formato (ratio) | U: contro WNR2 | E: contro WNR2 | U: contro WEX1 | E: contro WEX1 | Container V4.1 |
|---|---|---|---|---|---|
| WNR2 (0,8997) | — | — | +6,4% | +11,9% | 129,91 GB |
| WNR3 f+g+d+e (0,8807) | +1,41% (0,74…2,27); P(≥3%) = 0 | +2,31% (1,75…2,75) | +7,9% | +14,6% | 127,17 GB |
| WNR3 con tail seeding (0,8785) | +1,58% | +2,59% | +8,1% | +14,9% | 126,85 GB |
| WNR3z f+g+c+d+e (0,8534) | **+3,49%** (1,91…5,61); P(≥3%) = 0,67 | **+5,93%** (4,46…6,82); P = 1,0 | +10,1% | +18,6% | 123,21 GB |

- Rispetto ai 122,6 GB liberi su U: (reference, 17:11) mancano 4,57 GB per WNR3 e 0,61 GB per WNR3z. Con 5 GiB di margine diventano 9,94 e 5,98 GB.
- **Solo lo zero canonico** porta il formato vicino alla soglia +3% contro WNR2 su U:. Contro WEX1 tutti i WNR passano la soglia, se G2 e W2 vengono costruiti.
- Sono stime di modello. La misura che decide, una volta fatto G2, è la quota di tempo in cui il disco è occupato con WEX1: con ≥ 0,6 il guadagno mediano di WNR2 è ≥ +6%.

## 7. Scartato o superato

| Idea | Risultato | Perché |
|---|---|---|
| (a) bit "block max == 7" + metà blocco | −0,20% oggi (record reali), **+0,06% (peggio) sopra il ceil-form** | sul ceil-form il max ha 3 valori; il side bit costa più di quanto rende (misurato) |
| (b) K modelli / per-layer / cluster | −0,26…−0,28% oggi; −0,02…−0,04% sopra il ceil-form | erano quasi tutti il flip dell'ancora srel e i blocchi raddoppiati |
| pm2 (magnitudine di due posizioni prima) | −0,478% oggi con bucket a 6 valori (audit); **−0,0005% sopra il ceil-form** | proxy dei blocchi raddoppiati (misurato) |
| Posizione nel blocco | −0,02% sopra il ceil-form con una tabella da 185 KB | non entra in shared memory |
| Contesti sulle scale (sinistra/sopra/riga) | −0,5% dei bit di scala = 0,004% del record | le scale pesano lo 0,78% del record |
| Cross-expert, delta, XOR, dedup | informazione mutua 1,8·10⁻⁵ bit/w; 0 righe duplicate su 1,56 M | nessuno spazio di guadagno |
| Albero di contesti appreso (stile JPEG XL MA-tree), classi riga/colonna | 0,002–0,005 bit/w sopra il ceil-form | non vale una context map |
| Modelli adattivi, context-mixing, neurali nel decoder | ≤ 0,001 bit/w | costo di decodifica proibitivo |
| tANS/FSE, Huffman con LUT, coppie di simboli, stato a 16/64 bit | peggio | tabelle troppo grandi o ratio peggiore |
| Compressori generici (zstd, xz, OpenZL, Blosc2, float codecs) | tutti peggio | — |
| Tool pubblicati per pesi LLM (ptwm, EntroPack, crow-nest #148, ZipNN, ...) | nessun risultato 4-bit/MXFP4 batte 3,781 bit/w | nessuno sfrutta l'artefatto ceil→floor |
| Due stati interleaved sullo stesso byte stream | 0 velocità, +0,36% byte | misurato |
| Caricamento parziale per neurone | non lossless con un gate SiLU | — |

## 8. Altri tensori del checkpoint

Fonte: team 3, other-tensors (misurato su campioni reali).

- **Expert MTP (DSpark)**: stessa geometria degli expert principali (384 record × 18,8 MB = 7,22 GB) e si comprimono altrettanto, ma solo con un modello proprio: 3,768 bit/w, ratio 0,902. Con il WNM2 di produzione diventano più grandi (ratio 1,040), e 28 record su 36 finirebbero salvati raw. Servirebbe un secondo modello, e conviene solo se il motore usa davvero DSpark, cosa non verificata.
- **Pesi FP8 densi** (attention, shared expert, engram wkv; 7,40 GB): codificabili a 6,48–6,80 bit/byte. Restano però in memoria (6,65 GiB caricati una volta), quindi si risparmierebbe solo disco e tempo di caricamento.
- **Tabelle Engram** (202,76 GB, il 55% del checkpoint): idealmente ~40 GiB in meno. Ogni token legge solo 48 righe (12,7 KB), con indirizzi noti in anticipo dagli id dei token, quindi la compressione non riduce le letture: si guadagna spazio disco, non tok/s.

## 9. Correzioni ai numeri precedenti

- **Flush:** costa ~3,50 B/stream, non 4 B (audit, rANS esatto su 2,76 M stream). Le dimensioni assolute stimate erano alte di ~9,8 KB/record. Il record WNR2 reale held-out è 16.909.926 B = 0,89943 del raw; la produzione misura 0,8997. Le differenze tra varianti non cambiano.
- **Quota di −0:** è il 5,83% dei pesi V4.1. Il "12,8%" del brief è la quota di tutti gli zeri ±0 del pilota V4. WNR2z su V4.1 vale 0,1165 bit/w, non 0,128.
- **Integrità:** vedi sezione 3. La regola di oggi non rileva lo 0,26–0,63% dei flip singoli, e il test sullo "stream all'indice sbagliato" era impostato male.
- **Bucket pm2:** il mio bucket {none, 0–3, 4–5, 6–7} buttava via quasi tutta l'informazione. Con un bucket a 6 valori il guadagno sulla forma di oggi è −0,478%, non −0,226% (audit). Sopra il ceil-form non conta più.
- **"Block max sempre 6 o 7":** è vero per i byte salvati (forma OCP floor), ma sotto c'è una quantizzazione con regola ceil (sezione 1).
- **Precisione del confronto con la produzione:** ±0,004 bit/w. La std per expert è 0,026 bit/w, quindi l'errore standard su 160 expert è 0,0021.

## 10. Prossimi passi

**Decisioni che spettano al proprietario** (niente deploy né modifiche a the application config senza il suo OK):
1. Adottare il ceil-form (f) + l'ancora fissa (g0) come base di WNR3.
2. Zero canonico: accettare il bit-test sulla 5090 più il memcmp dei logit a ogni passo al posto del confronto delle logprob, oppure tenere la regola attuale.
3. Hash per stream (e) oppure TAIL+DIGEST per i 23 bit dello stato iniziale. La raccomandazione è l'hash.

**Lavoro che resta, in ordine:**
1. **Sul PC di produzione (RTX 5090).**
   - `bench2`/`ceilbench` del team 4 su A259, CF5, D e DB2.
   - Porting del ceil-form nel kernel del motore (ring cp.async, 567 righe), con la scala finale scritta in un output separato.
   - Bit-test dello zero canonico su ogni kernel che il motore può usare, inclusi DeepGEMM sm120 e cuBLASLt MXFP8.
2. **WNR2 con ancora fissa (g0).** Codificare 40 record con la mediana fissata a 120 e un modello riaddestrato, poi verificare nei sorgenti del motore che niente ricalcoli `median[m]`. Se regge, sono −0,24…−0,29% senza toccare il decoder.
3. **Conformità.**
   - Un decoder scritto in modo indipendente (C, o il prototipo GPU) che decodifichi i record del team 4.
   - Il confronto byte per byte dell'encoder Python WNR2 con quello C del motore.
   - I record codificati nel layout esatto della spec 0.2.1 (test T9, T13).
4. **Integrità.** Audit di corruzione (casi A–J del team 2) su record ceil-form con l'hash, e, se si vuole TAIL, implementare e misurare il DIGEST prima di sceglierlo.
5. ~~Correggere la spec v2 con gli errata del critic~~: fatto nella DRAFT 0.2.1, verificata da tre revisori.
6. **Misure mancanti.**
   - Il record più grande dei 7680, per dimensionare gli slot dei tier.
   - Un modello addestrato su tutti i 7680 record e non su 80.
   - Gli expert MTP con modello proprio, se il motore usa DSpark.
7. **Spazio disco.** Nessun formato V4.1 entra in U: con 122,6 GB liberi; serve una decisione su dove mettere il container.

**Gli script e i dati sono riproducibili:**
- `<RESEARCH-PC>\team2\` (audit, gpu-proto, zero-safety);
- `<RESEARCH-PC>\team3\` (ref-impl, wnr3-spec, capacity-value);
- `<RESEARCH-PC>\team4\` (ceil-ref, ceil-gpu, ceil-refuter, ceil-origin, capacity-update, spec-update);
- gli script di misura in `<TMP>\wnr2\` (mxceil.py, mxceil2.py e gli altri), con i risultati in `<RESEARCH-PC>\v41-results\`.

## Fonti principali

Verificate dal team 1: 100 dei 103 URL aperti; nessuna citazione inventata.

- OCP Microscaling Formats MX v1.0 (§5.3.3, §6.1): https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf. Rouhani et al., arXiv 2310.10537 (regola floor).
- DeepSeek-V4 `inference/kernel.py` (regola 2^ceil(log2(amax/6)) in `fp4_act_quant`): https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/kernel.py. Copia nel repo V4.1: https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4.
- rANS:
  - Giesen, `ryg_rans/rans_byte.h`;
  - Giesen, "Interleaved entropy coders", arXiv 1402.3392;
  - Duda, arXiv 1311.2540 §3.7;
  - Bloom, "ANS minimal flush" (cbloomrants, 2015);
  - libjxl `ans_params.h` / `dec_ans.h`.
- Decoder GPU: DietGPU (facebookresearch/dietgpu, `GpuANSDecode.cuh`); Recoil, arXiv 2306.12141; dtANS, arXiv 2603.01915.
- Contesti statici e context map: JPEG XL, arXiv 2506.05987; Brotli, RFC 7932.
- Stato dell'arte sui pesi:
  - ptwm (github.com/khwstolle/ptwm);
  - EntroPack, arXiv 2609.34185;
  - crow-nest #148;
  - Doubleword "weight entropy";
  - Heilper & Singer, arXiv 2508.19263;
  - ZipNN;
  - OpenZL, arXiv 2605.09928.
- Tier e offload: zsmalloc (docs.kernel.org), PagedAttention (arXiv 2309.06180), FluxMoE (arXiv 2604.02715), SSD-LLaMA (arXiv 2609.18110), vLLM RFC #63666.
- Hash: SMHasher (MurmurHash3_x86_32, valore di verifica 0xB0F57EE3).
- Codice colibri letto (albero di sviluppo locale): `deepseek_v4.c`, `deepseek_v41.c`, `quant.h`, `backend_cuda_dsv4.cu`, shader Vulkan.
- Reference interno: `D:\Desktop\WNR2-REFERENCE.md`.
