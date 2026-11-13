# team6 / cpu-simd build (kit version: no paths of the original PC).
# Needs gcc (MSYS2 ucrt64 / mingw-w64) on PATH, or set $env:WNR_GCC_BIN to its bin folder. Nothing is installed.
# Usage (from this folder):  powershell -ExecutionPolicy Bypass -File build.ps1 [-March native]
param([string]$March = 'native')
if ($env:WNR_GCC_BIN) { $env:PATH = "$env:WNR_GCC_BIN;$env:PATH" }
$src = $PSScriptRoot
$out = Split-Path $PSScriptRoot -Parent
# original PC (i7-7700K) used -march=haswell -mtune=skylake; on the KAIRA CPU use native (AVX-512 if present)
$common = @('-O3', "-march=$March", '-fopenmp', '-static', '-Wno-misleading-indentation', '-Wno-unused-variable', '-Wno-unknown-pragmas')
foreach ($nb in 5, 6, 7) {
    & gcc @common "-DNBSH=$nb" -o "$out\wnrdec_nb$nb.exe" "$src\wnrdec.c"
    if ($LASTEXITCODE -ne 0) { Write-Output "build nb$nb failed"; exit 1 }
}
Copy-Item "$out\wnrdec_nb7.exe" "$out\wnrdec.exe" -Force
Write-Output "built"
