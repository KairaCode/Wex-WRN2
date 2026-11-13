# Valutazione: proposta "WNR4" e whitepaper "Entropy Engine" (2026-11-08)

> **Dated report (2026-11-08).** It evaluates two documents that are outside this repository; present-tense sentences are the state of that day. What runs today: [STATUS.md](../../STATUS.md).

Documenti valutati:
- **A** = `docs/ricerca-avanzata-lossless-wnr4-20261108.md` (KAIRA, non in questo repo) (proposta WNR4);
- **B** = `KAIRA_Entropy_Engine_Architecture.pdf` (radice di KAIRA, non in questo repo) ("L'Entropia nell'Engine di KAIRA").

**Metodo.** Otto passate indipendenti: 4 controlli per gruppo di affermazioni (fonti citate, numeri WNR4, pipeline fusa, PDF) e 4 revisioni incaricate di smentirli.
- **Dati di confronto:** misure reali sulla 5090 (`research/wnr3-5090-20261107/RESULTS.md`), ricerca sui 160 expert V4.1 (`docs/reports/wnr3-research-20261107.it.md`, `docs/reports/wnr3-research-measurements-20261107.md`), baseline M0 e log G2 (`lab/v41-gpu-20261107/` (solo KAIRA)), codice in `src/colibri/c`. Per le fonti, i paper originali online.
- **Vincoli:** nessuna lettura di pesi, nessun lavoro GPU, perché era in corso l'A/B V4.1.
- **Revisione avversaria:** su ~110 verdetti ne ha ribaltati 9. Le correzioni sono già applicate qui sotto.

## Verdetto in breve

- **A (WNR4): da non seguire.** Si regge su numeri sbagliati per MXFP4: entropia, quota di zeri, dimensioni dei record. Le tre colonne nuove (micro-tANS, bitmap degli zeri, checkpoint stile Recoil) farebbero record **più grandi** o al massimo identici. La quarta (decode fuso nel GEMV) agisce su ~1% del tempo per token. I paper citati esistono, ma riguardano BF16 o un ANS di ordine 0, e WNR2 fa già meglio.
- **B (Entropy Engine): utile come archeologia del codice, non come piano.** Le parti di codice esistono quasi tutte. Hardware e baseline però sono sbagliati, le stime di guadagno non sono mai state misurate, e i tre interventi proposti sono o lossy (top-K dinamico), o non sul percorso caldo e con bug di concorrenza (eviction lockless), o descrivono codice che non esiste (tier CFS1 in `backend_cuda_dsv4.cu`).

## Fatti misurati che smentiscono le premesse

| Affermazione | Nel documento | Misurato su V4.1 REAP | Fonte |
|---|---|---|---|
| Entropia dei pesi MXFP4 | 2,10–2,35 bit/peso | ordine 0: 3,895; WNR2: 3,781; ceil-form + zero canonico: 3,592 | wnr3-research-measurements-20261107.md |
| "Gap" comprimibile | 1,6–2,0 bit (35–45%) | ~0,43 bit/peso (10%) oggi; al massimo ~0,63 (14,7%) value-exact | idem |
| Costo raw di MXFP4 | 4,5 bit/valore | 4,25 (4 bit + 8/32 di scala) | formato |
| Zeri esatti | 25–35% | 11,65% (5,83% +0, 5,83% −0) | wnr3-research-measurements-20261107.md |
| Livelli di magnitudine | "4 classi" | 8 livelli tutti popolati (i primi 4 coprono il 64%) | idem |
| Record WNR2 / WNR3 / WNR4 | 18,82 / 18,41 / 16,10 MB | WNR2 16,91 MB (il raw è 18,80 MB; 18,82 della proposta è circa il raw); WNR3 CFG 16,58 MB; 18,41 MB era un solo record di coda | RESULTS.md, t1_record_sizes |
| Collo di bottiglia | banda GDDR7 | attesa NVMe 270 ms/token (51%), MoE GPU 98 ms, attenzione CPU 62 ms; la lettura GDDR7 di tutto il lavoro GPU costa ~6 ms (~1%) | log G2 `g2-A2-wex1-gd0.stderr.log`, 399 token |
| RAM della macchina | 64 GB | 32 GB (DDR5-6200) | CIM, 2026-11-08 |
| Dischi | 2× Crucial T705 Gen5 su C: e U: | C: Crucial T700 Gen5 (10,66 GB/s), U: SK hynix Gen4 (6,58 GB/s) | m0.md |
| Baseline tok/s | 5,5–6,1 | V4.1 REAP 1,11 su CPU (M0); 1,80–2,10 con G2 su GPU (WEX1 su U:) | m0.md, log G2 |
| SM della 5090 | 192 | 170 | specifiche |

## Documento A, colonna per colonna

1. **Decode fuso nel GEMV (registri → mma).**
   - Oggi non esiste un "doppio round trip" per token: ogni record viene decodificato una volta, quando entra nel mirror VRAM (0,38 ms/record a batch 16).
   - Un GEMV che decodifica rANS a ogni uso costerebbe ~73–149 µs per expert contro 10,5 µs raw, cioè +11–23 ms/token a ~155 hit VRAM per token.
   - Anche il paper ISCA'26 citato ripiega sul decode non fuso a batch piccolo.
   - L'unico vantaggio reale è un tier VRAM compresso con ~11% di expert in più (~17% con WNR3z). Si ottiene anche senza fusione: decodifica in uno scratch con l'API batch che esiste già, costo ~21 µs per expert in più.
   - **No**, salvo un esperimento futuro sul tier VRAM compresso (vedi sotto).
2. **Micro-tANS a 4 stati / 4 simboli.**
   - Non è lossless: le magnitudini sono 8.
   - Anche una tabella unica L=16 senza contesti perde ~0,16 bit/peso rispetto a WNR2, circa +700 KB per record.
   - La ricerca aveva già misurato tANS/FSE come peggiori di rANS con contesti. **No.**
3. **Bitmap degli zeri da 16 bit ogni 16 pesi.**
   - Costa 1 bit/peso contro H(p0) = 0,52, quindi +0,48 bit/peso: record +19–22%, stimato −12…−14% di tok/s su U:.
   - Lo zero canonico va usato dentro il rANS, come nel profilo WNR3z. **No.**
4. **Checkpoint stile Recoil su uno stream unico.**
   - Con il dimensionamento corretto di Recoil (stati a 16 bit nei punti di rinormalizzazione), costano 0,18–0,44% del raw ogni 2048 pesi. Non "<0,05%".
   - Inoltre ridurrebbero il parallelismo, che è proprio ciò che serve al decoder GPU.
   - Oggi tutta la voce metadati + flush vale 0,57% del raw, e la tabella delle lunghezze codificata (d) ne prende già la parte facile. **No.**
5. **Tabella del §5.**
   - "750 / 960 GB/s" sono velocità di decodifica del microbench espresse in byte BF16 equivalenti (2 B/peso): un'unità non dichiarata, non confrontabile con la banda GDDR7.
   - ">2,40 TB/s sopra la banda fisica" e "0 µs" non hanno basi.
6. **Tabella di "convergenza"** (§3, KAIRA anticipa ISCA'26 / ZipServ / Unweight): i collegamenti sono inventati.
   - Il ceil-form è lavoro originale KAIRA.
   - La regola g non ha niente a che fare con Recoil, ma i suoi numeri sono corretti.
   - ZipServ e Unweight codificano gli esponenti BF16, che in E2M1 non esistono.

**Fonti citate** (esistono tutte, con dettagli sbagliati):
- **ISCA'26**, arXiv 2606.15789: gli autori sono Tan, Chen, Alonso, Wong e He, non "Shen". Usa rANS di ordine 0: su V4.1 darebbe 3,895 bit/peso, peggio di WNR2.
- **ZipServ/ZipGEMM**, ASPLOS'26, arXiv 2603.17435: lavora solo su BF16. Il 2,21× è un picco su L40S, con media 1,36×. Sulla 5090 c'è solo un test di compatibilità.
- **Unweight** (Cloudflare): uscito ad aprile 2026, non a gennaio. Solo BF16 e solo Hopper. A batch 1 è ~41% più lento.
- **Recoil**, arXiv 2306.12141: reale, ma per KAIRA non dà niente al parallelismo che serve.

## Documento B, interventi proposti

- **4.1 Top-K dinamico guidato dall'entropia del router: lossy.**
  - Cambia i token, quindi viola la regola dell'A/B con token identici.
  - L'opzione `V4_SKIP_COLD` esistente è già il limite superiore misurato di questa idea.
  - Il codice proposto non è un drop-in e le funzioni "probe" hanno firme diverse. **No.**
- **4.2 Eviction "termodinamica" lockless in `expert_store_wex.c`: no.**
  - Quello store non è sul percorso caldo del serve.
  - Anche credendo ai 480 ns per hit, ~240 lookup per token fanno ~0,1 ms su ~500 ms.
  - Il codice ha race serie: uno slot può essere riusato mentre qualcuno copia `cached_view`, `references` non viene mai decrementato, l'eviction corre contro i lookup. Per un motore lossless, leggere in silenzio l'expert sbagliato è il guasto peggiore.
- **4.3 Tier CFS1 "a 2,924 bit/peso" in VRAM: falso.**
  - 2,924 viene da `fse_coli.h` ed è misurato su INT4 di colibri, non su MXFP4. Su MXFP4 un ordine 0 sta a ~3,9 bit/peso.
  - `dsv4_rans_decompress_warp_kernel` e la decodifica dentro `dsv4_cuda_moe` non esistono.
  - Il tier DietGPU reale (`COLI_ANS`) vive solo in `colibri.c` / `backend_cuda.cu`, per il formato int4. È spento di default e solo per Linux.
  - "+42% RAM / +37% VRAM" diventa in realtà +2,6% con slot fissi, fino a ~+11% con slot impacchettati in byte.
- **Tabella del §5** (480→22 ns, −62% PCIe, 7,2–8,4 tok/s): nessun numero è misurato. Il −62% verrebbe solo da expert scartati, cioè lossy.

## Cosa vale davvero, in ordine di valore per KAIRA

1. **Byte su NVMe e layout dei dischi.** L'attesa NVMe è circa metà del tempo per token. Contano:
   - W2 (decode WNR2 batched);
   - l'A/B WEX1 vs WNR2 sullo stesso disco veloce;
   - lo striping su due dischi (esiste già: `mirror_cut_256` in expert_store_wex.c, `COLI_MODEL_MIRROR`, `COLI_WEX_EXTRA_MIRRORS` + `COLI_WEX_WEIGHTS`, obbligatorio con un mirror del solo container);
   - F1 (prefetch con soglia di confidenza).
2. **WNR3 byte-exact: ceil-form + regola g**, già portato nel kernel KAIRA in lab. −1,76% del raw, decode alla stessa velocità, decoder di produzione invariato per la sola regola g. Sopra si aggiungono:
   - (d) tabella delle lunghezze codificata, −0,137%;
   - (e) hash per stream, +0,007%, chiude il buco dei flip di segno silenziosi;
   - accorciare il flush rANS (small_k0), −0,19% del raw: idea nuova emersa dalla verifica, dati in `team2/ans-prior-art/flushx.json`.
3. **Zero canonico (WNR3z):** l'unica leva di formato che il modello stima sopra la soglia del +3% di tok/s.
   - È sicuro sul percorso di decode di default.
   - Il percorso DeepGEMM masked (`dg_moe<256>`, non di default) non è deterministico nemmeno a parità di input, e produce −0 nelle uscite bf16. Serve un controllo dedicato prima di usarlo con WNR3z.
4. **Prefetch predittivo dalle transizioni di routing** (B #26/#71, lossless). Il predittore V4 arrivava a R@8 84–85%; va riaddestrato per i 192 expert e i 40 layer di V4.1.
5. **Eviction valutata sul tasso di hit, non sul costo del lock** (B #61/#63). Si può provare offline con `V4_REPLAY_TRACE`. Su V4.1 l'hit rate è 34,6–37,1%, e ogni punto in più vale ~2,4 miss in meno per token.
6. **GEMV FP4 per FC1 di V4.1** (non misurato). `mv_fp4_grouped` usa il percorso veloce solo con I ≤ 4096 (`backend_cuda_dsv4.cu:234`). FC1 di V4.1 (I = 5120, due terzi dei byte di ogni expert) passa dal loop generico per byte. Va profilato dopo G2.
7. **Tier VRAM compresso** (solo come esperimento, dopo G2/W2): la parte fredda del pool mirror tenuta come record WNR2 con slot impacchettati in byte, decodificata in uno scratch con l'API batch. Niente kernel fuso.

## Correzioni a `research/wnr3-5090-20261107/RESULTS.md` (applicate il 2026-11-08)

Il documento A ha copiato questi errori dal report:
- **DB2:** costa +0,36% di byte di peso (134.946.449 contro 134.462.859), non "Baseline".
- **CF5:** il 30,6% è la quota di blocchi riscritti, non di codici di scala eliminati. Il risparmio sui byte di scala è −12,04%.
- **Totale 74.195.200 output:** non è ricostruibile; le righe delle fasi sommano a 83.841.024.
- **Percorso masked `dg_moe<256>`:** è escluso dal riepilogo e non è deterministico. Contiene −0 nelle uscite bf16, mentre il report dice "100% +0.0f".
- **"tcgen05 su sm_120a":** probabilmente errato. sm_12x usa mma.sync con block_scale.

## Errori nella nota di sintesi del 2026-11-08 (KAIRA docs, non in questo repo)

- RAM 64 GB invece di 32.
- `prefill_tail_rows` e `v41_dspark_rollback_row_t` non esistono nel codice.
- "752 µs per blocco MoE" è in realtà un microbench di decodifica.
- "Banda PCIe dimezzata" è falso: WNR2 risparmia ~10%.
- La regola 5 (bit-test obbligatorio) è presentata come decisa, ma la decisione spetta ancora all'utente.

I risultati completi per affermazione, con file:riga e URL, non sono in questo repo.
