# Note 2 from the main session (2026-11-09): add a safetensors arm

The user asked how WEX1/WNR2 compare with plain safetensors. The only direct measurement is the CPU M0 baseline (lab\v41-gpu-20261107\m0.md section 6):
- safetensors on U: 0.976 / 1.008 tok/s;
- WEX1 on U: 1.110 / 1.137 tok/s, i.e. +13-14%.

There is NO measurement on the GPU tiers.

When validating the short ABBA, also run a safetensors arm with the same ws-ab build on the GPU tiers: no COLI_EXPERT_STORE=wex and no COLI_WEX_PATH, experts read from U:\models\DeepSeek-V4.1-Flash-1017-REAP192-MXFP4.
- Keep everything else identical: --memory-gb, prompts, tokens, prefetch knobs.
- Order: S, A, B, B, A, S (S = safetensors, A = WEX1, B = WNR2), or an equivalent balanced order.
- Report tok/s, TTFT, NVMe wait and GB read per arm, and check that the token ids are identical across all three.
