# Test kit: GLM-5.3-Flash as a WEX4 container with WNR3 expert records

This kit lets you, on your own machine, turn the published checkpoint
[`amd/GLM-5.3-Flash-Quark-MXFP4`](https://huggingface.co/amd/GLM-5.3-Flash-Quark-MXFP4) (revision
`b5688f25`, MIT licence) into a WEX4 container whose routed experts are WNR3 records, check that
container in every way this repository knows, and send back one report. One script, `run.sh`, does
it in stages; `report.py` gathers the results into a text you paste into a GitHub issue.

You build the container yourself, from the checkpoint: no prebuilt container is published. The
build is the main path of the kit and the longest step (see [Time](#time)).

An Italian summary is at the end ([Riassunto in italiano](#riassunto-in-italiano)).

## What is being tested

1. **The converter on a 185 GB checkpoint.** `wex4/wex4_convert.py train` makes a WNM2 coding model
   from a seeded sample of 80 routed experts, read straight from the safetensors files; `pack
   --experts nr3 --names qwen38` then writes every MXFP4 routed expert (42 layers x 288 experts =
   12,096 records of 13,369,344 bytes) as a WNR3 record and every other tensor as it is
   (WBP1-compressed where that is smaller). The FP8 experts of layer 45 (the MTP layer) stay
   ordinary tensors.
2. **Both verifiers.** `wex4/wex4.py verify` (every hash, every record decoded) and
   `c/tools/wex4_verify` (every hash, the root to pin) must agree on the root.
3. **The round trip.** `unpack` rebuilds all 62 safetensors files; each must have the SHA-256 of
   your download, byte for byte.
4. **The decoders.** Every expert record (or a seeded sample) is decoded by the CPU reference and
   by the fast CPU decoder and compared with the checkpoint's bytes; then, where the hardware is
   there, by the CUDA helper on an NVIDIA GPU and by the Vulkan prototype.
5. **Speed.** Milliseconds per record for each decoder.
6. **The engine.** With a colibri build that has the GLM-5.3 WEX4 hook (colibri series 6, "GLM-5.3
   from a WEX4 container"), the same prompt is run greedily from the shards and from the
   container; the token ids must be the same. It reports tok/s and how many WNR3 records the CUDA
   helper and the CPU decoded.

## What you need

| | Needed | Notes |
|---|---|---|
| Disk | about **185 GB** for the checkpoint and about **160 GB** for the container, on one or two drives | the build checks for 0.9 x the checkpoint free before it starts; the round trip writes nothing by default (`--unpack-files` writes the 185 GB again) |
| RAM | 8 GB for stages 1-7; 32 GB for the engine stage | pack holds about 2 x threads x 60 MB of records in flight |
| CPU | x86-64 with AVX2, or 64-bit ARM | other CPUs work, with the scalar kernel of the fast decoder |
| NVIDIA GPU | optional | for the CUDA helper: `nvidia-smi`, and `nvcc` (CUDA 12 or later) unless you give a built helper |
| Vulkan GPU | optional | for the Vulkan prototype: `vulkaninfo` and `glslangValidator` |
| Network | about 185 GB of download | resumable |

After the round trip has passed you may give the checkpoint's space back with the `clean` stage
(`--delete-checkpoint`, see [Stage by stage](#stage-by-stage)). The engine stage reads the
non-expert weights from the checkpoint, so run it first; `unpack` gives the checkpoint back from the
container, byte for byte, if you need it later.

## Prerequisites

- **Windows:** [MSYS2](https://www.msys2.org/), in the **UCRT64** shell:
  `pacman -S git mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-python mingw-w64-ucrt-x86_64-curl`.
  For the Vulkan part add `mingw-w64-ucrt-x86_64-vulkan-headers mingw-w64-ucrt-x86_64-vulkan-loader
  mingw-w64-ucrt-x86_64-glslang` (and the GPU's driver). For the CUDA part, build the helper from an
  "x64 Native Tools" prompt of Visual Studio with CUDA's `nvcc`, or start the MSYS2 shell from such
  a prompt so that `nvcc` and `cl` are on `PATH`.
- **Linux:** `gcc`, `python3`, `git`, `curl` (Debian/Ubuntu: `apt install build-essential python3
  git curl`); `export PYTHON=python3`. Vulkan: `apt install libvulkan-dev glslang-tools vulkan-tools`.
  CUDA: the CUDA toolkit for `nvcc`, and colibri's `c/` (`--colibri`) for the Linux form of the GPU test.
- **macOS:** CPU only. Homebrew `gcc` and `python3`; `export PYTHON=python3 CC=gcc-15` (the version
  Homebrew installed).
- Optional on every OS: the Hugging Face CLI (`pip install -U huggingface_hub`, command `hf`) for
  the download; without it the kit uses `curl`, file by file.

## Steps

From a checkout of this repository (`git clone https://github.com/KairaCode/Wex-WRN2`):

```sh
cd Wex-WRN2
# 1. a dry run on a small synthetic checkpoint with GLM-5.3-Flash's expert shapes: a few minutes, no download
bash testkits/glm53-wex4/run.sh --synthetic --work ../glm53-dry all
# 2. the real thing, with the checkpoint downloaded by the kit into WORK/checkpoint ...
bash testkits/glm53-wex4/run.sh --work /big/disk/glm53 all
#    ... or with a download you already have
bash testkits/glm53-wex4/run.sh --work /big/disk/glm53 --checkpoint /models/GLM-5.3-Flash-Quark-MXFP4 all
# 3. the report
cat glm53-results/report.md
```

Stage `all` is `env fetch build verify roundtrip decode speed engine report`. Any stage runs alone
(`run.sh ... verify`), and any number of them in the order given. Each writes
`RESULTS/<stage>.log` and `RESULTS/<stage>.status` (`ok`, `failed` or `skipped`, with the reason);
a failed stage stops the run with `FAILED: stage ...` and the name of its log. `RESULTS` is
`./glm53-results` unless `--results` says otherwise. `bash run.sh --help` lists every option.

**Resuming.** Run the same command again after an interruption. `fetch` resumes the download (both
`hf download` and `curl -C -` continue partial files) and checks every file's size against
[`glm53-files.tsv`](glm53-files.tsv), the file list of the pinned revision. `build` keeps a
finished container (it writes `glm53-wnr3.wex4.done` with the root when pack ends) and a trained
model; an interrupted pack leaves `glm53-wnr3.wex4.partial`, which the next build removes and
starts again, since pack cannot continue inside a file. The other stages start over.

**A container built elsewhere.** `--container FILE` skips the build and runs the other stages on
that file. This is a shortcut for a second machine; the checks of stages 4-8 still need the
checkpoint, because they compare with it.

## Stage by stage

The expected output below is from the run on the reference machine ([Measured](#measured)); your
numbers differ, your verdicts must not.

| Stage | What it does | Ends with |
|---|---|---|
| `env` | records OS, CPU and flags, RAM, GPUs, tool versions and free disk (`env.json`); builds `wex4_tools` (fast and reference decoder), `wex_nr3_tools` and `wex4_verify` into `WORK/tools` | `[env] ok: tools built in ...` |
| `fetch` | downloads the checkpoint at revision `b5688f25` (or takes `--checkpoint`) and checks the size of all 72 files against the manifest; `FETCH_SHA256=1` also checks the SHA-256 of the 63 LFS files | `checkpoint: 72 files in the manifest, 185107699217 bytes here, 0 SHA-256 checked, 0 problems` |
| `build` | `train` (80 experts, seed 20261116), then `pack --experts nr3 --names qwen38 --threads N`; `sizes.json` | `12096 expert records; 288 experts kept as ordinary tensors (float or FP8, not MXFP4)`, `root <64 hex digits>` |
| `verify` | `wex4.py verify --dll wex4_tools` (fast decoder), then `wex4_verify FILE N`; both must give pack's root | `[verify] ok: root ...: both verifiers` |
| `roundtrip` | `unpack --sha-only`: every file rebuilt into its SHA-256, nothing written; then the checkpoint's files hashed and compared | `round trip: 62 files, 62 with the SHA-256 of the checkpoint, 0 different` |
| `decode` | every expert record decoded by the reference and the fast decoder and compared with the checkpoint's bytes (`--sample N`: a seeded sample); with an NVIDIA GPU, `test_wex_nr3_gpu` on a bin of 8 real records (20 mutations each); with a Vulkan device, the prototype's `run.sh` on the container (`MUTATIONS=0 WNR3_VK_EVERY=64`) | `[decode] ok: CPU reference and fast decoder[, CUDA helper][, Vulkan prototype]` |
| `speed` | the two CPU decoders on 3 seeded samples of 32 records, the decode alone timed; the CUDA test timed whole; the Vulkan prototype's bench mode | `speed-cpu-*.json`, `speed-*.txt` |
| `engine` | with `--colibri DIR` (colibri's `c/` with `glm53` built from the series-6 branch): `glm53 --model CHECKPOINT --ids IDS --greedy N` from the shards, then with `COLI_WEX_PATH`, `COLI_WEX_ROOT` and `COLI_WEX_VERIFY=1`, then also with `COLI_WEX_CUDA_LIB` if a helper is there | `engine A/B: 8 and 8 greedy tokens, identical; tok/s ...` |
| `report` | `report.md` and `report.json` | the report, printed |
| `clean` | not in `all`: with `--delete-checkpoint` and a passed round trip, deletes the checkpoint's `.safetensors` files | `... safetensors files deleted ...` |

**The engine stage.** Its prompt is the token ids of "The capital of France is" in the
checkpoint's tokenizer (the `DEFAULT_IDS` line of `run.sh`; `--ids` gives others). Build colibri's
glm53 from the series-6 branch (`make glm53`, or `make VK=1 glm53` / the CUDA build of
colibri's docs) and pass its `c/` folder as `--colibri`. The engine needs about 13 GB of RAM for the
dense weights at its default `GLM53_BITS=4`, plus the expert cache it sizes from the free memory.
The last line of each container run, `[glm53] WEX4 records: N read, M WNR3 decoded (CUDA helper
H, CPU C)`, goes into the report: a record the helper does not take (its status 1) is decoded on
the CPU and counted there.

## Time

@TIME_TABLE@

## Measured

@MEASURED@

## Privacy of the report

`report.md` and `report.json` hold the machine's description (OS, CPU model and flags, RAM, GPU
names, tool versions, free disk) and the results. `report.py` removes your login name, host name
and home folder from every string and writes `<path>` for anything that looks like an absolute
path, and refuses to finish if one of those is still in the text. Read the report before you post
it all the same. Paste it into an issue of this repository (the "Benchmark results" form).

## Files

| File | What it is |
|---|---|
| `run.sh` | the stages |
| `kit.py` | its helpers: env, the checkpoint check, sizes, the decode check, the GPU bin, the round-trip comparison, the engine A/B |
| `report.py` | the report |
| `glm53-files.tsv` | every file of the checkpoint at revision `b5688f25`: name, bytes, and the SHA-256 Hugging Face publishes for the LFS files |

## Riassunto in italiano

Il kit trasforma sul tuo computer il checkpoint `amd/GLM-5.3-Flash-Quark-MXFP4` (revisione
`b5688f25`) in un container WEX4 con gli esperti in record WNR3, lo verifica e produce un rapporto da
incollare in una issue. Il container lo costruisci tu dal checkpoint: non ne viene pubblicato uno
già fatto. Servono circa 185 GB per il checkpoint e circa 160 GB per il container; la RAM conta
solo per il motore (32 GB). Una GPU NVIDIA (helper CUDA) e una GPU Vulkan sono facoltative.

Comandi: `bash testkits/glm53-wex4/run.sh --synthetic --work ../glm53-dry all` per la prova a
secco (pochi minuti), poi `bash testkits/glm53-wex4/run.sh --work DISCO/glm53 all` (oppure con
`--checkpoint CARTELLA` se il checkpoint l'hai già). Ogni fase si può rilanciare da sola; dopo
un'interruzione basta rilanciare lo stesso comando: il download riprende, un container finito resta,
un pack interrotto riparte da capo. Dopo il round trip riuscito, `clean --delete-checkpoint`
libera lo spazio del checkpoint (`unpack` lo ricostruisce identico). Il rapporto è
`glm53-results/report.md`: non contiene nomi utente, nomi del computer né percorsi.
