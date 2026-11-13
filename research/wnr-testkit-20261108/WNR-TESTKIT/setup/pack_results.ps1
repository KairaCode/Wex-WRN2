<#
.SYNOPSIS
  Zips the kit's results\ folder for sending back (results_<COMPUTER>_<yyyyMMdd_HHmm>.zip in the kit root).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\setup\pack_results.ps1
  .\setup\pack_results.ps1 -Results D:\wnr-results -MaxFileMB 50 -Dest D:\send

.NOTES
  Files larger than -MaxFileMB (default 200) are left out and listed in SKIPPED_LARGE_FILES.txt inside the zip.
  Data files (*.bin expert tensors, *.part downloads) are never included. Regenerable cache folders are left out by
  default and listed (folder, file count, bytes) in the same file: any folder named enc, v1enc, t4enc, _cache, records,
  parts, feat or downloads (e.g. results\gpu_bench\enc ~800 MB, results\_cache\cpu_counts). -IncludeCaches keeps them.
  The computer name in the zip name can be replaced with -Tag.
#>
[CmdletBinding()]
param(
    [string]$Results = '',
    [string]$Dest = '',
    [int]$MaxFileMB = 200,
    [string]$Tag = '',
    [switch]$IncludeCaches
)
$CacheDirs = @('enc', 'v1enc', 't4enc', '_cache', 'records', 'parts', 'feat', 'downloads')
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
function Resolve-Full([string]$p) { return $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($p) }
$Res = if ($Results) { Resolve-Full $Results } else { Join-Path $Root 'results' }
if (-not (Test-Path $Res)) { Write-Host "no results folder at $Res"; exit 2 }
$DestDir = if ($Dest) { Resolve-Full $Dest } else { $Root }
New-Item -ItemType Directory -Force -Path $DestDir | Out-Null
if (-not $Tag) { $Tag = $env:COMPUTERNAME; if (-not $Tag) { $Tag = 'pc' } }
$stamp = Get-Date -Format 'yyyyMMdd_HHmm'
$zip = Join-Path $DestDir "results_${Tag}_$stamp.zip"

$stage = Join-Path ([System.IO.Path]::GetTempPath()) ("wnr_pack_" + [guid]::NewGuid().ToString('N'))
$stageRes = Join-Path $stage 'results'
New-Item -ItemType Directory -Force -Path $stageRes | Out-Null
$skipped = @()
$cacheSkip = @{}
$n = 0; $bytes = 0
$resFull = (Get-Item $Res).FullName.TrimEnd('\', '/')
Get-ChildItem -Path $Res -Recurse -File | ForEach-Object {
    $rel = $_.FullName.Substring($resFull.Length).TrimStart('\', '/')
    if (-not $IncludeCaches) {
        $segs = @($rel -split '[\\/]')
        $hitDir = $null
        for ($i = 0; $i -lt $segs.Count - 1; $i++) {
            if ($CacheDirs -contains $segs[$i].ToLower()) { $hitDir = ($segs[0..$i] -join '\'); break }
        }
        if ($hitDir) {
            if (-not $cacheSkip.ContainsKey($hitDir)) { $cacheSkip[$hitDir] = @(0, [int64]0) }
            $cacheSkip[$hitDir][0] += 1; $cacheSkip[$hitDir][1] += $_.Length
            return
        }
    }
    if ($_.Extension -eq '.bin' -or $_.Extension -eq '.part' -or $_.Length -gt ([int64]$MaxFileMB * 1MB)) {
        $skipped += "$rel`t$($_.Length)"
        return
    }
    $dst = Join-Path $stageRes $rel
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
    Copy-Item -LiteralPath $_.FullName -Destination $dst
    $n++; $bytes += $_.Length
}
foreach ($k in ($cacheSkip.Keys | Sort-Object)) {
    $skipped += "$k\`t$($cacheSkip[$k][1])`t(regenerable cache folder, $($cacheSkip[$k][0]) files; -IncludeCaches keeps it)"
}
if ($skipped.Count -gt 0) {
    $skipped | Out-File -Encoding utf8 -FilePath (Join-Path $stageRes 'SKIPPED_LARGE_FILES.txt')
}
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip -CompressionLevel Optimal
Remove-Item -Recurse -Force $stage
$zs = (Get-Item $zip).Length
Write-Host "packed $n files ($([math]::Round($bytes / 1MB, 1)) MB) -> $zip ($([math]::Round($zs / 1MB, 1)) MB)"
if ($skipped.Count -gt 0) { Write-Host "left out $($skipped.Count) large/data files or cache folders (listed in SKIPPED_LARGE_FILES.txt)" }
if (-not (Test-Path (Join-Path $Res 'ALL_RESULTS.json'))) {
    Write-Host 'note: results\ALL_RESULTS.json is missing; run: python -I setup\collect_results.py'
}
