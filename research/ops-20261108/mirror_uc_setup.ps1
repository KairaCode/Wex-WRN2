# Set up the WNR2 U:+C: mirror (approved by the user, 2026-11-09).
# 1. Re-verify the WEX1 copy on U: against its recorded sha256, then delete the duplicate WEX1 on D: (144 GB).
# 2. Move C:\models\DeepSeek-V4.1-Flash-1017-engram (engram shard 47, 101.5 GB) to D:, verifying sha256,
#    and leave a junction at the old path so COLI_MODEL_DIRS and kaira.json keep working.
# 3. Copy the WNR2 container E: -> C:\kaira_wex\v41-reap\ and verify it against the known sha256.
# Runs under heavy.lock (atomic create, 60 s poll, released in finally). Log: moves.log.
$ErrorActionPreference = 'Stop'
$lab  = '<KAIRA>\lab\v41-gpu-20261107'
$lock = Join-Path $lab 'heavy.lock'
$log  = Join-Path $lab 'moves.log'
function Log($m) { $l = '{0} {1}' -f (Get-Date -Format 's'), $m; Add-Content -LiteralPath $log -Value $l -Encoding utf8; Write-Output $l }
function Sha($p) { (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash }
function FreeGB($d) { [math]::Round((Get-PSDrive $d).Free / 1e9, 1) }

$wexU = 'U:\kaira_wex\v41-reap\v4_flash_experts.wex'
$wexD = 'D:\kaira_wex\v41-reap\v4_flash_experts.wex'
$engC = 'C:\models\DeepSeek-V4.1-Flash-1017-engram'
$engD = 'D:\models\DeepSeek-V4.1-Flash-1017-engram'
$nr2E = 'E:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex'
$nr2C = 'C:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex'
$nr2Sha = '3F703FA4D993ED46D2873676C71BE07CF11C11C7BAEF3E0BCE6428C30921EB9D'

$deadline = (Get-Date).AddMinutes(240)
while ($true) {
    try { $fs = [IO.File]::Open($lock, 'CreateNew', 'Write'); $w = New-Object IO.StreamWriter($fs); $w.WriteLine("main session: U:+C: mirror setup (delete WEX1 on D:, engram 47 C:->D:, WNR2 E:->C:) $(Get-Date -Format s)"); $w.Close(); break }
    catch [System.IO.IOException] { if ((Get-Date) -gt $deadline) { throw "lock: gave up after 240 min (held by: $(Get-Content $lock -ErrorAction SilentlyContinue))" }; Start-Sleep -Seconds 60 }
}
Log "lock taken (mirror setup)"
try {
    Log ("free before: C: {0} GB, D: {1} GB" -f (FreeGB C), (FreeGB D))

    # 1. WEX1: U: must match its recorded sha256 before the D: copy goes.
    if (Test-Path -LiteralPath $wexD) {
        $rec = (Get-Content -LiteralPath ($wexU + '.verified') -Raw)
        $m = [regex]::Match($rec, '[0-9A-Fa-f]{64}')
        if (-not $m.Success) { throw "no sha256 in $wexU.verified" }
        $hU = Sha $wexU
        if ($hU -ne $m.Value.ToUpper()) { throw "WEX1 on U: does not match its recorded sha256 ($hU vs $($m.Value)); D: copy kept" }
        Log "WEX1 on U: re-verified sha256 $hU; deleting duplicate $wexD"
        Remove-Item -LiteralPath $wexD -Force
        if (Test-Path -LiteralPath ($wexD + '.verified')) { Remove-Item -LiteralPath ($wexD + '.verified') -Force }
        Log ("deleted $wexD; D: free {0} GB" -f (FreeGB D))
    } else { Log "skip: $wexD already gone" }

    # 2. engram shard folder C: -> D: with junction
    $item = Get-Item -LiteralPath $engC -Force
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { Log "skip: $engC is already a junction" }
    else {
        if (Test-Path -LiteralPath $engD) { throw "target exists: $engD" }
        Log "move $engC -> $engD start"
        $a = @{}; Get-ChildItem -LiteralPath $engC -Recurse -File -Force | ForEach-Object { $a[$_.FullName.Substring($engC.Length)] = Sha $_.FullName }
        robocopy $engC $engD /E /COPY:DAT /DCOPY:T /J /R:1 /W:1 /NP /NFL /NDL /NJH /NJS | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "robocopy engram failed ($LASTEXITCODE)" }
        $b = @{}; Get-ChildItem -LiteralPath $engD -Recurse -File -Force | ForEach-Object { $b[$_.FullName.Substring($engD.Length)] = Sha $_.FullName }
        if ($a.Count -ne $b.Count) { throw "engram file count differs ($($a.Count) vs $($b.Count))" }
        foreach ($k in $a.Keys) { if ($a[$k] -ne $b[$k]) { throw "engram sha256 differs: $k" } }
        Log "engram verified ($($a.Count) files sha256 identical)"
        Remove-Item -LiteralPath $engC -Recurse -Force
        New-Item -ItemType Junction -Path $engC -Target $engD | Out-Null
        if (-not (Test-Path -LiteralPath (Join-Path $engC 'model-00047-of-00048.safetensors'))) { throw "junction not usable: $engC" }
        Log ("junction $engC -> $engD; C: free {0} GB" -f (FreeGB C))
    }

    # 3. WNR2 E: -> C:
    if (Test-Path -LiteralPath $nr2C) { throw "target exists: $nr2C" }
    $need = (Get-Item -LiteralPath $nr2E).Length + 5GB
    if ((Get-PSDrive C).Free -lt $need) { throw ("not enough space on C: for WNR2: {0} GB free" -f (FreeGB C)) }
    New-Item -ItemType Directory -Force -Path (Split-Path $nr2C) | Out-Null
    Log "WNR2 copy E: -> C: start"
    robocopy (Split-Path $nr2E) (Split-Path $nr2C) (Split-Path $nr2E -Leaf) /COPY:DAT /J /R:1 /W:1 /NP /NFL /NDL /NJH /NJS | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy WNR2 failed ($LASTEXITCODE)" }
    $h = Sha $nr2C
    if ($h -ne $nr2Sha) { throw "WNR2 on C: sha256 differs: $h" }
    Set-Content -LiteralPath ($nr2C + '.verified') -Value "sha256 $h  copied from $nr2E  $(Get-Date -Format s)" -Encoding utf8
    Log "WNR2 verified on C: sha256 $h"
    Log ("free after: C: {0} GB, D: {1} GB, U: {2} GB" -f (FreeGB C), (FreeGB D), (FreeGB U))
}
finally {
    Remove-Item -LiteralPath $lock -Force -ErrorAction SilentlyContinue
    Log "lock released (mirror setup)"
}
