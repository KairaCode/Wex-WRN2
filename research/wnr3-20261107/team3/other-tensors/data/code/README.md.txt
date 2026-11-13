---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - deepseek
  - deepseek-v4.1-flash
  - moe
  - reap
  - hope
  - expert-pruning
  - mxfp4
  - fp8
  - engram
---

# DeepSeek-V4.1-Flash-1017-REAP192-MXFP4

[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) with 192 of its 384 routed experts kept in each of the 40 decoder layers. Every kept expert is copied byte for byte from the base checkpoint, so this repository holds no requantized weights. It is the bit-exact master that the [GGUF](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-GGUF) and [Thor](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-Thor) builds were made from.

It is meant for people who want to study the pruned model at full source precision or derive their own builds. It needs about 340 GiB of disk with the Engram tables. Teacher-forced against the unpruned model on 185,565 held-out tokens, its top-1 agreement is **0.798**.

> **Release status.** This checkpoint **failed its release gate as pre-registered.** It did not beat a random keep-set of the same size on kl32 by a significant margin. The owner amended the rule on 2026-10-24, *after* seeing the result, to leave out the multilingual domain. The checkpoint passes under that amendment and was published under it. See [Release gates](#release-gates).

## Model overview

| | |
|---|---|
| Base model | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) at revision `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| Architecture | `DeepseekV41ForCausalLM` (`model_type: deepseek_v41`), with text, vision tower and aligner |
| Decoder layers | 40, hidden size 5120 |
| Routed experts | **192 of 384 per layer** (the same count in every layer), 6 active per token, plus 1 shared expert |
| Routed-expert parameters | 271.8 B (40 layers × 192 experts, counted from the safetensors headers) |
| Speculative-decoding layers | 3 DSpark/MTP layers, 128 experts each, **not pruned** |
| Vision | 32-layer ViT (hidden 1024) and aligner, unchanged |
| Engram | n-gram embedding tables on layers 1 and 14. They are unchanged and **not included** (see [Engram tables](#engram-tables-required-not-included)) |
| Context length | `max_position_embeddings` 1,048,576 (YaRN ×16 over 65,536), as in the base config. Not evaluated past 8K tokens (see Limitations) |
| Routed experts, precision | MXFP4 (E2M1 values, one E8M0 scale per 32), bit-exact copies of the base |
| Everything else, precision | As in the base: FP8 E4M3 with E8M0 block scales for attention, compressor, indexer and shared expert; BF16 router, embedding and head; F32 norms and hyper-connections |
| Router | `gate.weight`, `gate.bias` and `gate.bias_vl` sliced to the kept experts |
| Size on disk | 46 safetensors shards, 162,750,843,576 bytes (**151.6 GiB**). The two base Engram shards add 203,073,077,576 bytes (189.1 GiB) |
| Keep-set | `reap_keepset.json`, sha256 `1ace9b33134a2ca74174952593682e94edbf3c3348b4ac505ff59597a05d49ec` |
| Licence | MIT, inherited from the base model (`LICENSE` is copied unchanged) |

## Which version should I use

| repository | what it is | size | runs on |
|---|---|---:|---|
| **REAP192-MXFP4** (this repo) | bit-exact master: MXFP4 experts plus the base FP8/BF16 dense path | 151.6 GiB + Engram | no serving runtime has been run on it (see Quickstart). It does **not** fit a 128 GB device |
| [REAP192-Thor](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-Thor) | `thor-v1` layout: Q2_K/Q3_K experts (2.83 bpw), FP8 dense path, optional NVFP4 dense overlay | 106.9 GiB (+3.4 GiB overlay) | the [pure-CUDA server](https://github.com/patrickbdevaney/deepseek-v4.1-flash-cuda-server) on one 128 GB Jetson AGX Thor. **Recommended for Thor.** |
| [REAP192-GGUF](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-GGUF) | GGUF level L3: I-quant experts (2.46 bpw), Q8_0/Q6_K dense path, plus mmproj and DSpark drafter | 85.6 GiB | the [llama.cpp branch `deepseek-v4.1-reap-thor`](https://github.com/patrickbdevaney/llama.cpp/tree/deepseek-v4.1-reap-thor) |

No NVFP4 repack of this master was built. It would only have differed by a heal gain, and that gain was rejected (see [How it was made](#how-it-was-made)).

## Quickstart

**Download (both steps are required):**

```bash
hf download patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4 --local-dir DeepSeek-V4.1-Flash-1017-REAP192-MXFP4
cd DeepSeek-V4.1-Flash-1017-REAP192-MXFP4
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 model-00047-of-00048.safetensors model-00048-of-00048.safetensors \
    --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir .
```

`model.safetensors.index.json` already maps the Engram tensors to those two file names. A loader that reads the index fails until both files are present.

**Inference: untested.** The pipeline has not run any serving runtime (Transformers, vLLM, SGLang or the reference code) on this checkpoint. The only forward passes made on it were by the build pipeline's own teacher-forced replay harness (`ops/s10_eval.py`), which produced the numbers below. The repository includes DeepSeek's reference `inference/` scripts and `encoding/` chat encoder from the base repository, with `n_routed_experts` set to 192 in both config files. The reference `inference/run.sh` expects a checkpoint converted with `inference/convert.py` and 8-way model parallelism. The `library_name` tag reflects the base model's format, not a tested Transformers load.

## Evaluation

**Method.** This is a paired, teacher-forced comparison against the unpruned base model, run in the base model's own MXFP4/FP8 numerics. It covers **185,565 scored tokens from 26 held-out documents**: the *confirmation* split, which was never used to choose or tune anything. Intervals are 95% document-cluster bootstrap intervals.
- **kl32** is the mean KL divergence on the teacher's top-32 support.
- **top-1** is the share of tokens where this model's argmax equals the teacher's.
- **dNLL** is this model's next-token NLL minus the teacher's.
- **confident-flip rate** is the share of tokens where the teacher was confident and the argmax changed.

Source: `s10/final/results.json` of the build pipeline (gates file sha256 `f95b2672…`).

| arm | what it is | kl32 (95% CI) | top-1 (95% CI) | dNLL (95% CI) | confident-flip rate |
|---|---|---|---|---|---|
| `control` | all 384 experts, same harness (checks the harness, not the model) | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| **`mxfp4_A`** | **this checkpoint** | **0.4405 [0.2639, 0.6465]** | **0.7980 [0.7431, 0.8481]** | **0.2718 [0.1219, 0.4585]** | **0.0524** |
| `random0` | a random 192-expert keep-set (the gate's anchor) | 0.5419 [0.4351, 0.6554] | 0.7507 [0.7088, 0.7925] | 0.2070 [0.0726, 0.3210] | 0.0587 |
| `thorA` | the Thor build (K-quant experts) | 0.5194 [0.3261, 0.7423] | 0.7756 [0.7179, 0.8284] | 0.3091 [0.1426, 0.5119] | 0.0619 |
| `b128` | 128 experts, bit-exact MXFP4 (rejected arm) | 0.8026 [0.5627, 1.0801] | 0.7092 [0.6443, 0.7701] | 0.5469 [0.3271, 0.8010] | 0.0995 |

Paired against `random0`, this checkpoint wins on top-1 (+4.7 points, McNemar z = 48.4). On kl32 the all-domain difference is −0.1014 [−0.2196, +0.0400], which is not significant. On dNLL it is +0.0648 [−0.0529, +0.2021], also not significant: on NLL the random keep-set is not worse.

**By domain (`mxfp4_A`).** Each domain has only 1 to 5 held-out documents (math is a single document, so its interval is degenerate). Read these as indicative.

| domain | tokens | kl32 | top-1 | dNLL |
|---|---:|---|---|---|
| agentic | 40,955 | 0.1889 [0.1645, 0.2120] | 0.8589 [0.8514, 0.8670] | 0.0809 [0.0511, 0.1053] |
| code | 24,573 | 0.0462 [0.0310, 0.0709] | 0.9739 [0.9653, 0.9795] | 0.0336 [0.0220, 0.0508] |
| finance | 24,573 | 0.3666 [0.1989, 0.5876] | 0.7775 [0.7241, 0.8147] | 0.2240 [0.1065, 0.3626] |
| general | 32,764 | 0.4440 [0.3468, 0.5679] | 0.7817 [0.7353, 0.8206] | 0.1493 [−0.0422, 0.3407] |
| math | 8,191 | 0.1248 | 0.9390 | 0.0722 |
| **multilingual** | 24,573 | **1.5337 [1.1905, 1.8014]** | **0.5247 [0.4667, 0.5842]** | **1.2209 [0.8392, 1.5565]** |
| science | 24,573 | 0.3517 [0.2395, 0.5378] | 0.7724 [0.7342, 0.8130] | 0.2461 [0.1567, 0.3929] |
| vision | 5,363 | 0.3661 [0.3480, 0.3904] | 0.8745 [0.8649, 0.8842] | −0.1394 [−0.2389, −0.0489] |

**By position in the document (`mxfp4_A`).** The first 64 positions are under-sampled.

| position | tokens | kl32 | top-1 | dNLL |
|---|---:|---|---|---|
| [0, 8) | 180 | 1.4050 [1.0492, 1.7911] | 0.4333 [0.3722, 0.4934] | 0.4165 [−0.0821, 0.9147] |
| [8, 64) | 1,232 | 0.7652 [0.5039, 1.0748] | 0.6607 [0.5885, 0.7240] | 0.4124 [0.1856, 0.7043] |
| [64, 256) | 4,307 | 0.5781 [0.3934, 0.8118] | 0.7407 [0.6751, 0.7959] | 0.3434 [0.1742, 0.5495] |
| [256, 1k) | 17,145 | 0.4713 [0.3119, 0.6586] | 0.7713 [0.7188, 0.8206] | 0.3060 [0.1557, 0.4918] |
| [1k, 2k) | 22,935 | 0.4624 [0.2859, 0.6795] | 0.7866 [0.7284, 0.8396] | 0.3001 [0.1385, 0.5008] |
| [2k, 4k) | 46,798 | 0.3928 [0.2199, 0.6097] | 0.8126 [0.7570, 0.8624] | 0.2333 [0.0828, 0.4262] |
| [4k, 8k) | 92,968 | 0.4409 [0.2472, 0.6627] | 0.8035 [0.7457, 0.8558] | 0.2724 [0.1170, 0.4591] |

**Vision captions.** Caption perplexity over 5,256 tokens is 7.349, against 8.012 for the teacher (ratio 0.917). The random keep-set scores a ratio of 0.398 on the same tokens. A lower ratio is therefore *not* evidence of better captions. The metric serves only as a collapse guard (the gate allows up to 1.3).

**Not measured.** There are no generative benchmarks (MMLU, GSM8K, HumanEval and the like), no long-context quality past 8K tokens (the confirmation split has no longer row), no tool-calling accuracy and no human preference evaluation.

### Release gates

The thresholds were registered on 2026-10-22, before any confirmation result existed (gates file sha256 `169c0bba…`). The confirmation run on 2026-10-24 failed it. Two amendments followed on the same day, both after the result. Each is recorded as a chained hash in the pipeline's `prereg_lock.json`.

| date | change | made | effect on this checkpoint |
|---|---|---|---|
| 2026-10-21 | the long-context clause became a diagnostic, because the confirmation split has no row past 8K | before any result | none |
| 2026-10-22 | multilingual top-1 floor lowered from 0.65 to a 0.40 collapse guard | before any result | multilingual top-1 0.5247 passes only under this override |
| 2026-10-24 | **as registered: FAIL.** The checkpoint must beat `random0` on kl32; d_kl32 −0.1014 [−0.2196, +0.0400] does not exclude zero | — | — |
| 2026-10-24 | amendment 1: the beat-random test excludes multilingual (sha `204259f4…`) | **after the result** | d_kl32 −0.2144 [−0.2633, −0.1676], **PASS** |
| 2026-10-24 | amendment 2: the Thor build's 0.78 top-1 floor is waived (sha `f95b2672…`) | **after the result** | none here; it applies to the Thor build |

Multilingual alone causes the failure. There this checkpoint's kl32 is 1.534 against 0.894 for the random keep-set. In every other domain its kl32 is roughly half the random keep-set's. The as-registered results are kept beside the amended ones in the pipeline (`results.as_registered.json`, `gates.as_registered.json`).

## Hardware & memory

The pipeline ran on one NVIDIA Jetson AGX Thor (128 GB unified memory) and evaluated this checkpoint there, layer by layer. At 151.6 GiB the checkpoint does not fit in 128 GB, and it has never been served as a whole model. No speed is reported for it.

## How it was made

1. **Calibration.** 552 rows and 4,975,198 tokens, split across 9 domain buckets: general 12%, multilingual 10%, agentic 12%, code 12%, vision 14%, math 10%, science 10%, finance 10% and long-context 10%. The long-context bucket holds 10 rows of 32,768 to 65,536 tokens (corpus spec `c1803e70af7db162`). Sources are public Hugging Face datasets, filtered by licence in `ops/corpus_spec.py`: non-commercial sources are excluded, and the rest are permissive or attribution licences (English Wikipedia, CC BY-SA, is one source). No Claude-generated text was used. No corpus text is redistributed. The chat renderer kept role and content only, so tool-call turns were not part of calibration (see Limitations).
2. **Teacher pass.** The unpruned model ran layer by layer on the Thor with Engram on. It recorded per-expert REAP saliency, HOPE pairwise-interaction banks and a top-32 teacher cache.
3. **Selection.** Fourteen candidate keep-sets were replayed through the full model on a separate *screen* split (177,810 tokens) and ranked by held-out kl32 against the teacher. The candidates were HOPE at several protection levels, REAP-sum, routing mass, frequency, three random draws, multilingual-protected variants and two 128-expert variants. The winner, `hope_sum_pf0`, had the lowest screen kl32 (0.2977; top-1 0.8666). It ranks experts by HOPE with the REAP-sum diagonal and adds no per-domain protection (protect fraction 0). Every candidate, random ones included, kept a fixed guard set: Super-Expert candidates and the position-sensitive experts flagged by `ops/drift_guard.py`.
4. **Surgery.** The kept experts were renumbered 0–191 in ascending original id, and their packed bytes and scales were copied unchanged. The router rows and biases were sliced to the same ids. Every other tensor was copied as is. Each kept tensor was then re-read and compared against the source bytes (`ops/s05_surgery.py`).
5. **Heal: rejected.** The candidate was a per-layer output-scale gain on the routed sum. On the screen split it lowered NLL (−0.056 [−0.123, −0.022]) but left kl32 unchanged (+0.0010 [−0.0040, +0.0056]) and made top-1 worse (McNemar z −3.73). So it was not adopted, and every derived build is unhealed too.

Method references: REAP ([arXiv:2510.13999](https://arxiv.org/abs/2510.13999)) and HOPE ([arXiv:2609.18916](https://arxiv.org/abs/2609.18916)). Pipeline code: [`patrickbdevaney/deepseek-v4.1-flash-reap-thor`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor), with the stages under `ops/` and the evaluation summary in `RESULTS.md`.

## Engram tables (required, not included)

The two Engram tables and their projection tensors are not pruned. They are not re-hosted here. Download them from the base repository at the pinned revision (the same information is in `engram_pointer.json`):

| file | bytes | sha256 |
|---|---:|---|
| `model-00047-of-00048.safetensors` (layer 1) | 101,535,150,936 | `76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5` |
| `model-00048-of-00048.safetensors` (layer 14) | 101,537,926,640 | `9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd` |

## Limitations

- **Multilingual quality is much lower than in the unpruned model** (top-1 0.52, kl32 1.53). At 192 of 384 experts, every selection the pipeline tried lost multilingual text. Keep-sets that protected more multilingual experts recovered it only by losing general, finance and science text. For non-English use, prefer the unpruned model.
- **The release gate was amended after the result** (see above).
- **Long context is untested** past 8K tokens on the confirmation split.
- **Tool calling.** The calibration chat renderer dropped `tool_calls`, so no DSML tool-call tokens were routed during selection. On the Thor build, tool-call formatting has been seen to degrade: the release server smoke test returned a malformed argument string. Tool use is not evaluated.
- **No serving runtime has been run on this exact checkpoint** (see Quickstart).
- The per-domain numbers rest on 1 to 5 documents per domain.

## Changelog / corrections

- 2026-10-24: published under the amended gate (s12 card, generated from `s10/final/results.json`).
- 2026-10-26: s12 card regenerated. It added the Thor-build waiver paragraph.
- 2026-11-06: card rewritten and verified against repo files and build artifacts; a 2026-10-26 automated rewrite contained errors. Corrections to earlier cards: the size is 151.6 GiB, not 146.4 GiB. The shipped keep-set used no per-domain protection (protect fraction 0), not the 0.06 knee. The long-context calibration rows are 32K–64K tokens, not 32K–128K. No NVFP4 build exists. The drafter was not refit; the three MTP layers ship unchanged. The cited gates hash `f95b2672…` is the file *after* both amendments; the hash as registered is `169c0bba…`.

## License & attribution

MIT, the same as the base model. The weights are derived from [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) by [DeepSeek-AI](https://huggingface.co/deepseek-ai). The `LICENSE`, `inference/` and `encoding/` files are DeepSeek's, copied unchanged except for the expert count in the configs.

## Citation

```bibtex
@misc{devaney2026dsv41reap192,
  title        = {DeepSeek-V4.1-Flash-1017-REAP192: bit-exact 192-of-384 expert pruning of DeepSeek-V4.1-Flash-1017},
  author       = {Patrick Devaney},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4}}
}
```
