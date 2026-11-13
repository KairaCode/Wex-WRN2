# Mirror A/B, 2026-11-10: WNR2 on two drives vs WEX1 on one

Kit step 3 (`short-abba/short_abba.sh`), V4.1 REAP-192, RAM_GB 16, ctx 32768, 7 prompts x 64 tokens plus a 16-token warm-up, one A1 B1 pair, idle PC after a reboot.

- A: WEX1 `v4_flash_experts.wex` on U: only.
- B: WNR2 `v4_flash_experts.nr2.wex` on U: plus a full mirror on C:, engine read split 38/62 (`COLI_WEX_EXTRA_MIRRORS`, `COLI_WEX_WEIGHTS=38,62`).

Result: B decodes **+49%** faster (2.99 vs 2.02 tok/s mean, +37..+63% per prompt), TTFT -22%, disk wait per token 130 vs 305 ms, per-read latency 6.7 vs 15.0 ms. Token ids and logits are bit-identical. One pair only, so there is no interval; B changes both the format and the drive split, so this does not isolate the mirror effect.

```text
== runs done 15:44:09, 9 min
A1: WEX v1 U:\...\v4_flash_experts.wex, READY 18.1 s, run 298 s, RAM_GB 16 (budget 16.0 GiB, 9 slots, OS available 18), VRAM experts 971 (GPU memory used before start 899 MiB), logits a14761eb37afa76c
   read path: edisk 1298 s / 86752 reads = 14.97 ms per read, 4.64 outstanding; token median 479 ms, stall limit 2000 ms
   OS read bytes per drive: C: 0% (1 GiB), U: 100% (1527 GiB)
   w0          16 tok  2.272 tok/s TTFT    4.37 s wait  252.7 ms/tok wall   11.4 s
   code1       64 tok  2.237 tok/s TTFT   10.63 s wait  252.2 ms/tok wall   39.3 s
   code2       64 tok  2.323 tok/s TTFT   11.17 s wait  240.7 ms/tok wall   38.7 s
   chat_it1    64 tok  1.648 tok/s TTFT   10.11 s wait  392.4 ms/tok wall   49.0 s
   chat_it2    64 tok  1.829 tok/s TTFT   10.71 s wait  356.8 ms/tok wall   45.7 s
   reason1     64 tok  1.948 tok/s TTFT   12.26 s wait  311.3 ms/tok wall   45.1 s
   reason2     64 tok  2.039 tok/s TTFT   11.26 s wait  291.1 ms/tok wall   42.7 s
   short1       8 tok  2.097 tok/s TTFT    4.42 s wait  291.3 ms/tok wall    8.3 s
B1: WEX v3 U:\...\v4_flash_experts.nr2.wex, READY 17.99 s, run 214 s, RAM_GB 16 (budget 16.0 GiB, 9 slots, OS available 18.9), VRAM experts 970 (GPU memory used before start 888 MiB), logits a14761eb37afa76c
   read path: edisk 628 s / 93179 reads = 6.74 ms per read, 3.20 outstanding; token median 321 ms, stall limit 2000 ms
   mirrors ['C:\...\v4_flash_experts.nr2.wex'], engine split: 2 drives: 38% 62%
   OS read bytes per drive: C: 61% (897 GiB), U: 39% (579 GiB)
   w0          16 tok  3.065 tok/s TTFT    3.86 s wait  118.5 ms/tok wall    9.1 s
   code1       64 tok  3.060 tok/s TTFT    8.48 s wait  118.0 ms/tok wall   29.4 s
   code2       64 tok  3.205 tok/s TTFT    8.71 s wait  110.4 ms/tok wall   28.7 s
   chat_it1    64 tok  2.663 tok/s TTFT    7.57 s wait  158.7 ms/tok wall   31.6 s
   chat_it2    64 tok  2.980 tok/s TTFT    7.89 s wait  140.0 ms/tok wall   29.4 s
   reason1     64 tok  2.938 tok/s TTFT    9.62 s wait  130.1 ms/tok wall   31.4 s
   reason2     64 tok  2.977 tok/s TTFT    9.17 s wait  127.4 ms/tok wall   30.7 s
   short1       8 tok  3.078 tok/s TTFT    3.38 s wait  128.4 ms/tok wall    6.0 s
arm A (1 runs, 7 run x prompt): mean 2.017 tok/s, TTFT 10.08 s, wait 305.1 ms/tok (means only: the prompt mix dominates their spread); read path 14.97 ms per read, 4.64 outstanding, edisk 1298 s, 86752 reads
arm B (1 runs, 7 run x prompt): mean 2.986 tok/s, TTFT 7.83 s, wait 130.4 ms/tok (means only: the prompt mix dominates their spread); read path 6.74 ms per read, 3.20 outstanding, edisk 628 s, 93179 reads
paired B vs A over pairs [1] x ['code1', 'code2', 'chat_it1', 'chat_it2', 'reason1', 'reason2', 'short1'] (prompts fixed, df=0):
   tok/s +48.98 +- nan % (n=7); per prompt {'code1': [36.79], 'code2': [37.97], 'chat_it1': [61.59], 'chat_it2': [62.93], 'reason1': [50.82], 'reason2': [46.0], 'short1': [46.78]}
   cross-check, pair means [48.98]: +- nan % (df=0; not the verdict's interval)
   TTFT  -22.46 +- nan %
   wait  -174.7 +- nan ms/tok
   validity: per-prompt means code1 +36.79, code2 +37.97, chat_it1 +61.59, chat_it2 +62.93, reason1 +50.82, reason2 +46.00, short1 +46.78; SD between prompts 10.33 points. The interval is pair noise for this prompt set only; another set can move the mean by about SD/sqrt(n_prompts) = 3.91 points.
   same-arm spread between runs, %: run means {} (guard > 2%); per prompt {} (warning > 3.5%)
identity (token ids (logits argmax)) over ['w0', 'code1', 'code2', 'chat_it1', 'chat_it2', 'reason1', 'reason2', 'short1']: identical in every run; logits bit-identical: True
verdict: INCONCLUSIVE: no interval (prompts ['code1', 'code2', 'chat_it1', 'chat_it2', 'reason1', 'reason2', 'short1'], df=0)
dumps: deleted 2 logits files (all in tokens.json with the same sha256)
short_abba.sh rc=0
```
