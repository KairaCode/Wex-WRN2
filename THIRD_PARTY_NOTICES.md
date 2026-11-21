# Third-party notices

The repository is licensed under the Apache License 2.0 ([LICENSE](LICENSE)); [NOTICE](NOTICE) holds its attribution notices, including the one colibri requires. This page lists the material of other authors that is not covered by those two files, with its licence.

## colibri (Apache License 2.0)

https://github.com/JustVugg/colibri, Copyright 2026 Vincenzo Fornaro. The files derived from colibri and the patches against it are listed in [NOTICE](NOTICE), each derived file names its source in its header. The licence text is [LICENSE](LICENSE).

## ds4 (MIT), as carried by colibri

colibri's own THIRD_PARTY_NOTICES.md states that portions of its `c/backend_cuda_dsv4.cu`, including the DeepSeek-V4 GPU router selection algorithm, are adapted from `ds4_cuda.cu` in the ds4 project, https://github.com/antirez/ds4. These files of this repository quote parts of `backend_cuda_dsv4.cu`: `integration/backend/backend_cuda_dsv4.wex-excerpt.cu`, `integration/backend/backend-upload-race-fix.patch`, `integration/history/wex-gpu-direct.patch`, `research/wnr3-20261107/team2/zero-safety/scripts/zs_gpu_kernels.cu`, `research/wnr-testkit-20261108/WNR-TESTKIT/tests/gpu/src_zero/zs_gpu_kernels.cu` and `research/wnr3-20261107/team2/zero-refuter/gpu_probe.py`. To the extent that they include portions adapted from ds4, this notice applies to them:

```text
MIT License

Copyright (c) 2026 The ds4.c authors
Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## DeepSeek inference code, model card and checkpoint headers (MIT)

`research/wnr3-20261107/team3/other-tensors/data/` holds files copied from the Hugging Face repository `patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4` (revision named in `data/*/manifest.json`), a pruned copy of `deepseek-ai/DeepSeek-V4.1-Flash-1017`:

- `code/inference__model.py.txt` and `code/inference__engram.py.txt`: byte copies of `inference/model.py` and `inference/engram.py`;
- `code/README.md.txt`: the repository's model card;
- `headers/*.header.json`: the JSON headers of the 48 safetensors shards (tensor names, dtypes, shapes and offsets; no tensor data).

That repository is licensed under the MIT License, Copyright (c) 2023 DeepSeek; the full text is in [code/LICENSE](research/wnr3-20261107/team3/other-tensors/data/code/LICENSE) next to the copies.

## OCP Microscaling Formats (MX) Specification v1.0 (OWFa 1.0)

`research/wnr3-20261107/team2/zero-safety/out/ocp-mx.txt` is a text extract of the Open Compute Project's "OCP Microscaling Formats (MX) Specification", Version 1.0 (https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf), written by `scripts/pdftext.py` of the same run folder to check the format's zero and rounding rules. The specification states that its use is governed by the Open Web Foundation Modified Final Specification Agreement ("OWFa 1.0"), whose terms are at http://www.opencompute.org/participate/legal; the extract keeps the specification's own licence and disclaimer section.

## Data computed from model weights

No model weights are in this repository. These files are statistics computed from the routed-expert weights of public checkpoints. The DeepSeek checkpoints of the first two rows are released under the MIT License (Copyright (c) 2023 DeepSeek); the rows of single runs name the licence of each of their checkpoints as its Hugging Face page states it (the pages of the rows other than `real-experts-20261115` were read on 2026-11-21):

| Files | Computed from |
|---|---|
| `models/v4-flash-0731.wnm2` | `deepseek-ai/DeepSeek-V4-Flash-0731`: symbol frequencies of all routed experts (a WNM2 model, 59,408 B) |
| `models/v41-reap192.wnm2`; the `.wnm2`, `.wnm3` and `.u16` lab models under `research/` | `patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4` (experts copied byte for byte from `deepseek-ai/DeepSeek-V4.1-Flash-1017`): symbol frequencies |
| result files under `research/` and `wex4/research/` | the checkpoints of the other rows and other public checkpoints named in the files: counts, entropies, sizes and hashes of records |
| the result files of `research/q38-wex-q8-20261105/` | the local copy `Qwen3.8-Flash-Next-FP8` (the files do not name its Hugging Face repository; the repository of that name, `Qwen/Qwen3.8-Flash-Next-FP8`, is tagged "other" with licence name "qwen-community-1.0" on Hugging Face, licence at https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/blob/main/LICENSE): read speeds of routed experts, byte entropies, exponent histograms and bits per value of fixed-exponent forms |
| the result files of `research/wnr3-20261107/team4/ceil-origin/` | `openai/gpt-oss-20b`, `openai/gpt-oss-120b` and `amd/gpt-oss-120b-w-mxfp4-a-fp8` (tagged Apache-2.0 on Hugging Face), `deepseek-ai/DeepSeek-V4-Flash`, `deepseek-ai/DeepSeek-V4.1-Flash`, `deepseek-ai/DeepSeek-V4.1-Flash-1017` and `deepseek-ai/DeepSeek-V4-Pro` (MIT License), `amd/DeepSeek-R1-0528-MXFP4` (tagged MIT on Hugging Face), `RedHatAI/GLM-5.3-MXFP4` (tagged "other" with licence name "glm-5.3" on Hugging Face, licence at https://huggingface.co/zai-org/GLM-5.3/blob/main/LICENSE) and `amd/MiniMax-M3-MXFP4` (tagged "other" with licence name "minimax-community" on Hugging Face, licence at https://huggingface.co/amd/MiniMax-M3-MXFP4/blob/main/LICENSE; named in the table of `fetch_other.py`, no result file of the folder comes from it): bits per weight under the original and the ceil-form scales, and a byte comparison of the OpenAI and AMD gpt-oss-120b experts |
| the result files of `research/qwen38-mxfp4-20261111/` | `amd/Qwen3.8-Flash-Next-Quark-MXFP4`, the source of the routed experts as [docs/reports/QWEN38-MXFP4-20261111.md](docs/reports/QWEN38-MXFP4-20261111.md) states (tagged "other" on Hugging Face, licence file "Qwen Community License 1.0" at https://huggingface.co/amd/Qwen3.8-Flash-Next-Quark-MXFP4/blob/main/LICENSE), and the local copy `Qwen3.8-Flash-Next-FP8` of the `q38-wex-q8-20261105` row for the rest of the model: generated tokens and text, log-likelihood gaps, tok/s and memory figures |
| the logs and result files of `research/real-experts-20261115/` (no tensor data, no model, no record) | `deepseek-ai/DeepSeek-V4.1-Flash` (MIT License), `amd/Qwen3-235B-A22B-Instruct-2507-MXFP4` (tagged Apache-2.0 on Hugging Face) and `amd/MiniMax-M2.1-MXFP4` (tagged "modified-mit" on Hugging Face, licence at https://github.com/MiniMax-AI/MiniMax-M2.1/blob/main/LICENSE): sizes, SHA-256 of fetched tensors and of records, compression ratios and decode times |
| the README and the log of `research/converter-qwen35-20261116/` | `amd/Qwen3.5-35B-A3B-MXFP4` (tagged Apache-2.0 on Hugging Face, licence at https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/LICENSE): sizes, record counts and SHA-256 of the source files and of the container |
| the logs and result files of `research/nvfp4-wnr3-20261117/` | `deepseek-ai/DeepSeek-V4.1-Flash` (MIT License), `nvidia/DeepSeek-V4.1-Flash-1017-NVFP4`, `amd/DeepSeek-V4.1-Flash-Quark-MXFP4`, `nvidia/GLM-5.3-Flash-NVFP4` and `amd/GLM-5.3-Flash-Quark-MXFP4` (tagged MIT on Hugging Face), `nvidia/Qwen3.8-Flash-Next-NVFP4` (tagged "other" with licence name "nvidia-open-model-license" on Hugging Face), `amd/Qwen3.8-Flash-Next-Quark-MXFP4` (tagged "other" on Hugging Face, licence file "Qwen Community License 1.0") and `Bizuayeu/GLM-5.3-Flash-Jammed-XSFP4-NZ5` (tagged "other" with licence name "nvidia-open-model-agreement" on Hugging Face, licence at https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/): byte ranges and SHA-256 of fetched tensors, coded sizes, entropies and decode speeds |

`fixtures/v41-tiny/` is synthetic: it was written by colibri's `tools/make_dsv41_tiny.py` (seeded random weights at toy shapes, DeepSeek V4.1 tensor layout), so it holds no weights of a released model.
