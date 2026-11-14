# Documentation index

Every page under `docs/`, what kind of page it is and which question it answers. The pages at the top of `docs/` are kept true; the pages in `docs/reports/` record one date and are not rewritten. Words you do not know: [GLOSSARY.md](GLOSSARY.md).

## Names in 30 seconds

Two things have names here, and they are numbered separately.

- A **container** is the layout of the file: WEX1, WEX2, WEX3 or WEX4. It holds one record per routed expert, each on a 4096-byte boundary so that the engine fetches it with one unbuffered read.
- A **record codec** is how one record is coded inside the container: raw (codec 0), WNR2 (codec 4), WNR3 (codec 5) and a few others.

| Container | What it is | State |
|---|---|---|
| WEX1 | raw aligned records behind a header page and an entry table | no production file left; still the input of the WNR3 builder and of the WEX4 upgrade |
| WEX2 | WEX1 with the scales packed in tiles | written and tested, never built for a whole model |
| WEX3 | the same layout with coded records: WNR1 (retired), WNR2 (superseded), WNR3 | read path kept; no production file left |
| WEX4 | a SHA-256-hashed header page and record index around unchanged raw or WNR3 records | in production |

So "WNR3" alone names a record codec, and "a WNR3 container" means a WEX3 or WEX4 file whose records are WNR3. The complete table of containers, codecs, magics and flags is in the root [README](../README.md), section "Format family". The repository name spells the older codec "WRN2"; every constant and page says WNR2.

Summary of 2026-11-13; the page of record for what runs is [STATUS.md](../STATUS.md):

| Model | Container | Record codec | Drives |
|---|---|---|---|
| DeepSeek V4.1 REAP-192 | WEX4, root pinned | WNR3 with canonical zero ("WNR3-CZ", codec 5) | U: with a full mirror on C: |
| Qwen3.8-Flash-Next, MXFP4 experts | WEX4, root verified and not pinned | raw (codec 0) | U: with a full mirror on C: |

## I want to...

**...report a vulnerability.** [SECURITY.md](../SECURITY.md): privately, through GitHub's vulnerability reporting; which readers are supported and what the formats do not protect against.

**...understand the design.** [ARCHITECTURE.md](ARCHITECTURE.md): the components, the write and read paths, the trust boundaries, what is not in this repository, and which files of `c/` build here.

**...integrate the formats into an engine.** Open [COLIBRI-INTEGRATION.md](COLIBRI-INTEGRATION.md): what to copy and build (sections 1 and 2), how to make and verify a WEX4 container (3), how to wire an engine (4), every environment variable with its default (5), mirrors, integrity and troubleshooting (6 to 8). The quoted engine code and the patches are listed in [integration/README.md](../integration/README.md); the level-by-level guide of 2026-11-09 (WEX1, WEX2, WNR2) is the dated report [colibri-integration-20261109.md](reports/colibri-integration-20261109.md).

**...implement a reader or a writer.** One page per layer:

- WEX4 container: [WEX4-SPEC.md](WEX4-SPEC.md), sections 3 to 6 (layout, header page, index, validation) and 17 (conformance, with the vectors in [wex4/vectors/](../wex4/vectors/README.md)). Reference code: `c/wex4.h` and the independent Python reader in [wex4/](../wex4/README.md).
- WNR3 record: [WNR3-SPEC.md](WNR3-SPEC.md), sections 2 to 4, after the errata at its top (where they differ, the errata and the code win).
- WEX1 / WEX2 / WEX3 header and entry table, WNR2 record, WNM2 model, helper API: [WNR2-REFERENCE.md](WNR2-REFERENCE.md), sections 5 to 10. Reference code: `c/wex_format.h`, `c/wex_neural_rans.h`.

**...check a result.** Start from the number in [BENCHMARKS.md](BENCHMARKS.md): a row or paragraph names its dated report or its run where the repository has one; the rows that have none are listed in its section "Where the numbers come from". The report is in the table below; the scripts and raw output of the run are in [research/README.md](../research/README.md), which gives the verdict and the entry file of every run. How numbers are stated here (pairs, conditions, what was not run): [CONTRIBUTING.md](../CONTRIBUTING.md), section 5.

**...know what runs today.** [STATUS.md](../STATUS.md): the two production containers with their roots, drives, switches and builds, what is in the engine but switched off, and what was not verified.

**...read the history.** [CHANGELOG.md](../CHANGELOG.md) for the format changes by date, then the dated reports below, newest first. The record of the WNR2 period (2026-10-20 to 2026-11-10: decisions, disk tables, measurements, timeline) is [reports/wnr2-history-20261110.md](reports/wnr2-history-20261110.md).

## Every page

"Current" means the page is edited when the facts change. "Historical" means it records its date: read its banner, and take present-tense sentences as the state of that day. Language: EN English, IT Italian.

### Living pages (`docs/`)

| File | Lang. | Kind | State | Question it answers |
|---|---|---|---|---|
| [README.md](README.md) | EN | index | current | Which page do I need? |
| [GLOSSARY.md](GLOSSARY.md) | EN | reference | current | What does this word mean, and which file defines it? |
| [ARCHITECTURE.md](ARCHITECTURE.md) | EN | overview | current | How do the pieces fit together, where does a record go from the checkpoint to the GPU, and what is not in this repository? |
| [BENCHMARKS.md](BENCHMARKS.md) | EN | measurements | current: every table, newest rows of 2026-11-13 | What was measured, under which conditions, how many runs stand behind it, and where is the raw output? |
| [COLIBRI-INTEGRATION.md](COLIBRI-INTEGRATION.md) | EN | guide | current: WEX4 with RAW or WNR3 records | What do I copy and build, how do I make, verify and pin a container, which variables does the engine read, and what does an error line mean? |
| [WEX4-SPEC.md](WEX4-SPEC.md) | EN | spec | current: format version 4.0, in production | What are the bytes of a WEX4 file, how is it validated and hashed, and how is a WEX1 or WNR3 container upgraded in place? |
| [WNR3-SPEC.md](WNR3-SPEC.md) | EN | spec | current: FINAL v1, in production. Sections 0 to 8 with the errata at the top are the format; sections 9 to 11 are the implementation plan, historical | What are the bytes of a WNR3 record, and how are ceil-form, rule g, canonical zero and the stream hash encoded and checked? |
| [WNR2-REFERENCE.md](WNR2-REFERENCE.md) | EN | reference | current as the description of the WEX1 / WEX2 / WEX3 layout, the WNR2 record, the WNM2 model, the encoder and the helper API (sections 4 to 10); WNR2 itself is superseded and no WNR2 container exists. Sections 1 to 3 and 11 to 15 point to the history report | What is in a WEX1 / WEX2 / WEX3 container, a WNR2 record and a WNM2 model, and what does the CUDA helper offer? |
| [NEXT-FORMAT-SPEC.md](NEXT-FORMAT-SPEC.md) | EN | stub | current | Nothing. It is the old path of WEX4-SPEC.md, kept because a comment in the mirrored `c/wex4.h` names it |

### Dated reports (`docs/reports/`), newest first

| File | Lang. | Kind | State | Question it answers |
|---|---|---|---|---|
| [colibri-integration-20261109.md](reports/colibri-integration-20261109.md) | EN, Italian summary at the top | dated report (the integration guide until 2026-11-13) | historical: WEX1, WEX2 and WNR2 level by level, engine quotes of 2026-11-09, sections 12 to 14 a deployment log of 2026-11-09/10 | How were the WEX1-WNR2 hooks ported into colibri, built, generated, verified and deployed? |
| [wnr2-history-20261110.md](reports/wnr2-history-20261110.md) | EN | dated report (taken out of WNR2-REFERENCE.md on 2026-11-13) | historical: the WNR2 period, 2026-10-20 to 2026-11-10 | What did the owner decide, where did every container and model sit, what was measured with WNR2 and when, and which sources disagreed? |
| [QWEN38-MXFP4-20261111.md](reports/QWEN38-MXFP4-20261111.md) | EN | dated report | historical (banner: its status lines say WEX1, the containers are WEX4). Its measurements are the record for Qwen3.8 | Do the MXFP4 experts of Qwen3.8 keep the quality of the FP8 ones, how large and how fast are their containers, and what did the engine-side fixes gain? |
| [wex4-summary-20261111.it.md](reports/wex4-summary-20261111.it.md) | IT | dated report | historical, superseded by WEX4-SPEC.md | What is WEX4 in a few pages: what it adds over safetensors and WEX3, what was measured, what the deploy needed? |
| [WNR3-PRODUCTION-20261110.md](reports/WNR3-PRODUCTION-20261110.md) | EN | dated report | historical for the configuration (banner). The rebuild steps still apply | How was V4.1 switched to WNR3, what did the switch measure, what can be rolled back, and how is the container rebuilt from the safetensors? |
| [WNR3-VERIFY.md](reports/WNR3-VERIFY.md) | EN, Italian title and summary | dated report (2026-11-10) | historical; the final verification results of WNR3 | Is WNR3 exact, smaller and safe on 103 real V4.1 records: sizes, corruption audit, GPU decode time, the canonical-zero proof? |
| [wnr2-reads-vs-ram-cache-20261109.md](reports/wnr2-reads-vs-ram-cache-20261109.md) | EN | dated report (a note) | historical | Why did WNR2 make about 11% more reads than WEX1 in the W3 A/B? |
| [wnr4-entropy-engine-evaluation-20261108.it.md](reports/wnr4-entropy-engine-evaluation-20261108.it.md) | IT | dated report | historical | Do a "WNR4" proposal and an "Entropy Engine" paper hold against the measurements? Both documents are outside this repository |
| [v41-reap-baseline-m0-20261107.md](reports/v41-reap-baseline-m0-20261107.md) | EN | dated report | historical; its numbers are the CPU-only baseline | How fast is each drive, and how fast is V4.1 on the CPU from safetensors and from WEX1? |
| [v41-reap-wnr2-build-20261107.md](reports/v41-reap-wnr2-build-20261107.md) | EN | dated report | historical (the container it describes was deleted) | How was the V4.1 WNR2 container built, how long did it take and how was it verified? |
| [wnr3-research-20261107.it.md](reports/wnr3-research-20261107.it.md) | IT | dated report | historical, superseded by WNR3-SPEC.md, which took only part of it | What could a successor of WNR2 change, and what is each change worth on 160 real V4.1 experts? |
| [wnr3-research-measurements-20261107.md](reports/wnr3-research-measurements-20261107.md) | EN | dated report | historical | What are the raw numbers and scripts behind the WNR3 research report? |
| [wex-prod-20261105.it.md](reports/wex-prod-20261105.it.md) | IT | dated report | historical; its Qwen3.8 FP8 proposal was not built | What did WEX need on 5 November to be fit for production, and where would Qwen3.8 gain? |
| [wnr2-wex3-when-compression-pays-20261105.it.md](reports/wnr2-wex3-when-compression-pays-20261105.it.md) | IT | dated report | historical (DeepSeek V4) | When is a compressed record faster than a raw one, and what does the literature say for this case? |
| [wnr2-cache-decode-cpu-wddm-20261105.it.md](reports/wnr2-cache-decode-cpu-wddm-20261105.it.md) | IT | dated report | historical (DeepSeek V4) | Which counters decide whether WNR2 pays, and do CPU decode, the cache policy or WDDM (the Windows GPU driver model) change that? |
| [wex-architecture-20261018.it.md](reports/wex-architecture-20261018.it.md) | IT | dated report | historical, with known errors listed in [wnr2-history-20261110.md](reports/wnr2-history-20261110.md) section 14; no longer the format reference | How were WEX1, WEX2 and WEX3 designed in October, on DeepSeek V4? |
| [wex2-scale-tiles-20261018.it.md](reports/wex2-scale-tiles-20261018.it.md) | IT | dated report | historical; the WEX2 format it describes is still what the code reads | How are the scales packed in WEX2, and what did that measure on two experts? |

## Outside `docs/`

- What runs, measurements, sync state: [STATUS.md](../STATUS.md). Format changes by date: [CHANGELOG.md](../CHANGELOG.md). Rules for changing this repository: [CONTRIBUTING.md](../CONTRIBUTING.md).
- Research runs, one row each: [research/README.md](../research/README.md). The studies and reviews behind WEX4 sit next to its code: [wex4/README.md](../wex4/README.md).
- The draft that preceded the WNR3 specification is `research/wnr3-20261107/team4/spec-update/WNR3-DRAFT-v2.1.md`. It describes a model blob, a 64-byte record header and coded stream lengths that were **not** adopted (WNR3-SPEC.md, section 0): do not implement from it.
