# Rollback of the 2026-11-09 V4.1 GPU install (master cfb7ea0 + 6be34da) into app\.
# Restores the 4 engine files from this folder and removes the "v41gpu" profile from app\kaira.json.
# Usage (KAIRA closed):  powershell -ExecutionPolicy Bypass -File rollback.ps1 [-RestoreKairaJson]
#   -RestoreKairaJson : if kaira.json was rewritten since the install (the exact v41gpu block is no longer
#                       found), restore the backed-up kaira.json; the current one is kept here first.
param(
  [string]$AppDir = '<KAIRA>\app',
  [switch]$RestoreKairaJson
)
$ErrorActionPreference = 'Stop'
$Here = $PSScriptRoot

$running = Get-Process -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -match '^(KAIRA|deepseek_v4|deepseek_v41|qwen38|qwen36|colibri|serve_bench)$' }
if ($running) { throw "Close KAIRA first; still running: $(($running.Name | Sort-Object -Unique) -join ', ')" }

foreach ($f in 'deepseek_v4.exe', 'coli_cuda_dsv4_dg.dll', 'coli_cuda_dsv4.dll', 'mcp_server.py') {
  $src = Join-Path $Here "colibri-optimized\$f"
  $dst = Join-Path $AppDir "colibri-optimized\$f"
  Copy-Item $src $dst -Force
  if ((Get-FileHash $dst).Hash -ne (Get-FileHash $src).Hash) { throw "hash mismatch after restore: $f" }
  Write-Host ("restored {0,-24} {1}" -f $f, (Get-FileHash $dst).Hash.ToLower())
}

$kj = Join-Path $AppDir 'kaira.json'
$block = [IO.File]::ReadAllText((Join-Path $Here 'v41gpu-block.txt'))
$text = [IO.File]::ReadAllText($kj)
$at = $text.IndexOf($block)
if ($at -ge 0) {
  $text = $text.Remove($at, $block.Length)
  [IO.File]::WriteAllText($kj, $text, (New-Object Text.UTF8Encoding $false))
  Write-Host 'kaira.json: v41gpu profile removed (rest of the file untouched)'
} elseif ($RestoreKairaJson) {
  $keep = Join-Path $Here ("kaira.json.before-rollback-" + (Get-Date -Format 'yyyyMMdd-HHmmss'))
  Copy-Item $kj $keep
  Copy-Item (Join-Path $Here 'kaira.json') $kj -Force
  Write-Host "kaira.json: restored from backup; previous file kept as $keep"
} else {
  Write-Warning 'kaira.json changed since the install: v41gpu block not found verbatim. Remove the "v41gpu" entry from "models" by hand, or rerun with -RestoreKairaJson.'
}
if ([IO.File]::ReadAllText($kj) -match '"active_model"\s*:\s*"v41gpu"') {
  Write-Warning 'active_model is still "v41gpu": set it back (e.g. "v4") before starting KAIRA.'
}
Write-Host 'rollback done'
