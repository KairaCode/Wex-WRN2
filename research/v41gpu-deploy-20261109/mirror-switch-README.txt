Mirror switch, 2026-11-09 06:17 (approved by the user, 2026-11-09).
Backup folder: <KAIRA>\lab\app-before-mirror-20261109-061724 (kaira_wex_cuda.dll + kaira.json)

Rollback (KAIRA closed): copy kaira_wex_cuda.dll back; in the current app\kaira.json restore only the v41gpu env by hand (COLI_WEX_PATH to the .wex, drop EXTRA_MIRRORS/WEIGHTS/V4_WEX_BATCH). Do not copy the backup kaira.json over later changes.

Changes:
- helper app\colibri-optimized\kaira_wex_cuda.dll:
  a79772004426c38f... -> cceb4fb573b3b3e7b71843bf85a2b87cf9efdd94155bfc5deec717cce949e782
  (the W1 lane build in lab\v41-gpu-20261107\w1\build, the one W3 measured with WNR2)
- kaira.json profile v41gpu:
  COLI_WEX_PATH  U:\kaira_wex\v41-reap\v4_flash_experts.wex -> U:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex
  + COLI_WEX_EXTRA_MIRRORS=C:\kaira_wex\v41-reap
  + COLI_WEX_WEIGHTS=38,62
  + V4_WEX_BATCH=1
