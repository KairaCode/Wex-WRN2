# models/

Trained WNM2 probability models. A WNM2 model is the table of symbol frequencies that the WNR2 and WNR3 record codecs code against: one model per container, 59,408 B, identified by the FNV-1a hash of its bytes (the model id). A container carries its own copy of the model, so these files are not needed to read a container. They are kept to rebuild one, and as test input.

**Both files here belong to WNR2 containers that no longer exist. The model of the container in production is not in this repository** (see below).

| File | Model id (FNV-1a) | Size | sha256 | Trained on | Codec | Container that used it |
|---|---|---|---|---|---|---|
| `v41-reap192.wnm2` | `0xab4261bd` | 59,408 B | `f83fee82734219358d0f1290245298d44e2808d7abb48c9c0a742cd6dd855e3e` | DeepSeek V4.1 Flash REAP-192 MXFP4, all 7,680 routed experts | WNR2 (entry codec 4), chunks 2048:512 | the V4.1 WNR2 container of 2026-11-07: 129,905,360,896 B, 7,680 / 7,680 records. Every copy was deleted after the switch to WNR3 on 2026-11-10 |
| `v4-flash-0731.wnm2` | `0x49435c70` | 59,408 B | `2ddd4220cd6e17951a875c691c37b160efac1a04f6c468530576cc67c69d2d49` | DeepSeek V4 Flash 0731, the full build of 2026-10-20 (11,008 experts; the builder trains on every expert present) | WNR2 (entry codec 4), chunks 4096:2048 | the V4 WNR2 container: 128,368,873,472 B. Deleted (disk-state table of the WNR2 reference) |

Sources: ids and "trained on" from [docs/WNR2-REFERENCE.md](../docs/WNR2-REFERENCE.md) (the model-blob table and the build log) and the build table in section 6.5 of [docs/reports/colibri-integration-20261109.md](../docs/reports/colibri-integration-20261109.md). Size, sha256 and id were recomputed from the files on 2026-11-12.

Check a file:

```sh
sha256sum models/*.wnm2
python -I -c "import sys; sys.path.insert(0,'archive/nr2'); import nr2lib; print(hex(nr2lib.fnv1a(open(sys.argv[1],'rb').read())))" models/v41-reap192.wnm2
```

## What production uses (as recorded on 2026-11-12; current state: [STATUS.md](../STATUS.md))

- **DeepSeek V4.1 REAP-192** runs from WNR3-CZ records in a WEX4 container. Its model has id `0x27e8a1e0` ([docs/reports/WNR3-PRODUCTION-20261110.md](../docs/reports/WNR3-PRODUCTION-20261110.md); [full-build result](../research/wnr3-20261110/full-build/RESULT-20261110.md)). **That file is not in the repository**: none of the 15 tracked `.wnm2` / `.wnm3` files has that id (FNV-1a of each, computed 2026-11-12). The model is inside the production container. `c/tools/build_nr3.py ... --train-lab80 --cz` trains it again from a WEX1 container of the same checkpoint (the production report has the rebuild procedure); the full build of 2026-11-10 produced the same id as the lab's model. `--save-model FILE` writes the model out.
- **Qwen3.8-Flash-Next MXFP4** runs from raw records (codec 0) in WEX4 containers. Raw records need no model.

## Models elsewhere in the repository

Test fixtures and lab models under `research/`, not for production: `research/wnr3-5090-20261107/ceil-kaira/model_CFG_rule_g.wnm2` (id `0x299dc586`, ceil-form with rule g and no canonical zero, 80 experts; the fixture of gate G3 in [docs/WNR3-SPEC.md](../docs/WNR3-SPEC.md)) and `model_CF_median.wnm2` next to it (id `0xc2a18e9c`). The other 11 tracked model files are lab models of the 2026-11-07/08 exploration, with three distinct ids, under `research/wnr3-5090-20261107/g0/` and the test-kit archive.
