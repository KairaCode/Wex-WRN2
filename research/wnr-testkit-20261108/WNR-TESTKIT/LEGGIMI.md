# Kit di test WNR2 / WNR3 — istruzioni rapide

Questo kit serve a lanciare sull'altro PC (previsto: Windows con RTX 5090) i test che ho fermato qui. Qui **non è stato eseguito nessun test completo**: ho fatto solo controlli di compilazione e prove CPU di pochi secondi. I dettagli in inglese sono in `README.md` e nei README di ogni cartella.

## In 5 passi (PowerShell, dalla cartella del kit)

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r setup\requirements-gpu.txt
.\venv\Scripts\python.exe -I setup\check_env.py
.\venv\Scripts\python.exe -I setup\fetch_data.py --extra
powershell -ExecutionPolicy Bypass -File .\setup\run_all.ps1 -Quick
```

1. **Python:** serve la 3.12 o più recente. Per la sola CPU basta la 3.10; senza GPU NVIDIA installa `setup\requirements-cpu.txt`.
2. **Driver:** la 5090 vuole il driver R570 o più recente, cioè CUDA ≥ 12.8 in `nvidia-smi`. Non serve installare il CUDA toolkit.
3. **Dati:** 3,0 GB per il campione, più 1,4 GB con `--extra`.
   - Di default si scaricano dalla revisione Hugging Face pinnata.
   - Se sull'altro PC hai già il checkpoint: `--local-shards <cartella con i safetensors>`.
   - Ogni file viene verificato con sha256.
4. **Prima passata:** `-Quick`, stimata in 30–60 minuti (non misurata). Poi la versione completa, senza `-Quick`, richiede alcune ore: il solo audit prende 1–2 ore.
5. **Rimandarmi i risultati:** `powershell -ExecutionPolicy Bypass -File .\setup\pack_results.ps1` crea `results_<PC>_<data>.zip`.

## Cosa gira

| Test | Cosa verifica | Dove |
|---|---|---|
| `heldout_configs` | La tabella dei record reali (WNR2, ceil-form, g0, d, e, t, c), decodificati in modo byte-exact o value-exact, uguali ai valori di questo PC | CPU |
| `wnr2_g0` | WNR2 con ancora fissa (120 o coppia più densa): il decoder WNR2 non modificato decodifica byte-exact, con le dimensioni misurate | CPU |
| `chunk_sweep` | Stream da 1024/2048/4096/8192 pesi, con hash o tail | CPU |
| `train_curve_mtp` | Curva di apprendimento (10–190 expert di training) e opzioni per gli expert MTP. **Richiede `--extra`** | CPU |
| `audit` | Audit di corruzione T3: regola di oggi vs HASH vs TAIL; ogni flip non rilevato e il danno che fa | CPU |
| `nvrtc_check` | Tutti i kernel compilano per sm_86 e sm_120 (non lancia niente) | GPU, solo compilazione |
| `zero_bittest` | Zero canonico: 0 bit diversi nei kernel colibri copiati, con controlli positivo e negativo | GPU |
| `gpu_bench` | Benchmark del decoder: A259, ceil-form CF5/CF6, sign bypass D, DB2; byte-exact contro i byte originali e tempi accoppiati | GPU |
| `gpu_v1_profile` | **Sperimentale**: profilo CEIL+LENC+HASH nel kernel. Gira solo con `-All` | GPU |

Ogni test scrive `results\<test>\summary.json` e stampa `RESULT <test> PASS|FAIL|DONE`. `run_all` continua anche se un test fallisce, e alla fine raccoglie tutto in `results\ALL_RESULTS.json`. Usa `-NoGpu` per saltare la GPU e `-Only`/`-Skip` per scegliere i test.

## Workflow di ricerca (opzionali, con Claude Code sull'altro PC)

`workflows\` contiene le versioni portabili dei team 5, 6 e 7 che ho fermato. Riprendono il lavoro parziale salvato in `partial\`:
- team 5: encoder conforme alla spec più decoder C indipendente, e profilo v1 su GPU;
- team 6: decoder CPU AVX2 e valore sul percorso CPU-only, piano dei segni con zero canonico;
- team 7: NVFP4, altri modelli MXFP4, formato Engram e piano disco.

Lanciane uno alla volta, mai insieme a `run_all`. Il comando esatto è in `workflows\README_WORKFLOWS.md`:

```
Workflow({scriptPath: "<kit>\\workflows\\wnr-kit-team5-conformance.js", args: {root: "<kit>", data: "<kit>\\data"}})
```

## Cose da sapere

- **GPU mai provata:** nessun percorso GPU del kit è stato eseguito qui, per scelta. Eventuali errori compariranno al primo `-Quick`.
- **`train_curve_mtp`:** senza gli expert extra fallisce. Scaricali con `--extra`, oppure salta il test con `-Skip train_curve`.
- **Formato dei record:** l'audit e i test CPU usano il formato record di ceil-ref, non il layout esatto della spec 0.2.1. Quello lo produce solo il workflow del team 5.
- **Percorsi KAIRA:** quelli che richiedono il build di KAIRA (DeepGEMM sm120, cuBLASLt MXFP8, Vulkan, memcmp dei logit end-to-end) sono elencati in `tests\gpu\CHECKLIST-kaira-gpu-paths.md`. Si fanno a mano.
- **Documenti:** `docs\` contiene report, spec 0.2.1, misure e WNR2-REFERENCE. `docs\README.md` mappa i percorsi di questo PC citati nei documenti sulle cartelle del kit.
- **Regole:** il codice del formato KAIRA non va mai upstream su colibri. Niente deploy né modifiche a kaira.json senza un OK esplicito.
