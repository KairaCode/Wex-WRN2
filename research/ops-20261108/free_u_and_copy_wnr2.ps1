# Free space on U: (approved by the user, 2026-11-08) and copy the WNR2 container E: -> U:.
# Moves lfm2.5, mellum2.1 and U:\kaira_lab\wnr2-samples to H: (sha256-verified), leaves junctions in place,
# then copies E:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex to U:\kaira_wex\v41-reap\ and verifies it.
# Runs under heavy.lock (atomic create, 60 s poll, released in finally). Log: moves.log next to this file.
$ErrorActionPreference = 'Stop'
$lab  = '<KAIRA>\lab\v41-gpu-20261107'
$lock = Join-Path $lab 'heavy.lock'
$log  = Join-Path $lab 'moves.log'
function Log($m) { $l = '{0} {1}' -f (Get-Date -Format 's'), $m; Add-Content -LiteralPath $log -Value $l -Encoding utf8; Write-Output $l }

function Hashes($root) {
    $h = @{}
    Get-ChildItem -LiteralPath $root -Recurse -File -Force | ForEach-Object {
        $rel = $_.FullName.Substring($root.Length).TrimStart('\')
        $h[$rel] = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
    }
    $h
}

function MoveWithJunction($src, $dst) {
    if ((Get-Item -LiteralPath $src -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { Log "skip $src (already a junction)"; return }
    if (Test-Path -LiteralPath $dst) { throw "target exists: $dst" }
    Log "move $src -> $dst start"
    $a = Hashes $src
    robocopy $src $dst /E /COPY:DAT /DCOPY:T /R:1 /W:1 /NP /NFL /NDL /NJH /NJS /MT:8 | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed ($LASTEXITCODE) for $src" }
    $b = Hashes $dst
    if ($a.Count -ne $b.Count) { throw "file count differs: $($a.Count) vs $($b.Count) for $src" }
    foreach ($k in $a.Keys) { if ($a[$k] -ne $b[$k]) { throw "sha256 differs: $k" } }
    Log "verified $($a.Count) files sha256 identical: $dst"
    Remove-Item -LiteralPath $src -Recurse -Force
    New-Item -ItemType Junction -Path $src -Target $dst | Out-Null
    if (-not (Test-Path -LiteralPath $src)) { throw "junction not usable: $src" }
    Log "junction $src -> $dst"
}

# take the lock
$deadline = (Get-Date).AddMinutes(180)
while ($true) {
    try { $fs = [IO.File]::Open($lock, 'CreateNew', 'Write'); $w = New-Object IO.StreamWriter($fs); $w.WriteLine("main session: free U: (move lfm2.5/mellum2.1/wnr2-samples to H:) + copy WNR2 E:->U: $(Get-Date -Format s)"); $w.Close(); break }
    catch [System.IO.IOException] { if ((Get-Date) -gt $deadline) { throw "lock: gave up after 180 min (held by: $(Get-Content $lock -ErrorAction SilentlyContinue))" }; Start-Sleep -Seconds 60 }
}
Log "lock taken"
try {
    Log ("U: free before {0:N1} GB" -f ((Get-PSDrive U).Free / 1e9))
    New-Item -ItemType Directory -Force -Path 'H:\kaira_lab' | Out-Null
    MoveWithJunction 'U:\models\lfm2.5-24b-a2b'      'H:\models\lfm2.5-24b-a2b'
    MoveWithJunction 'U:\models\mellum2.1-12b-a2.5b' 'H:\models\mellum2.1-12b-a2.5b'
    MoveWithJunction 'U:\kaira_lab\wnr2-samples'     'H:\kaira_lab\wnr2-samples'
    $free = (Get-PSDrive U).Free
    Log ("U: free after moves {0:N1} GB" -f ($free / 1e9))

    $srcW = 'E:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex'
    $dstW = 'U:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex'
    $need = (Get-Item -LiteralPath $srcW).Length + 5GB
    if ($free -lt $need) { throw ("not enough space on U: for WNR2: free {0:N1} GB, need {1:N1} GB" -f ($free/1e9), ($need/1e9)) }
    if (Test-Path -LiteralPath $dstW) { throw "target exists: $dstW" }
    Log "WNR2 copy E: -> U: start"
    $hs = (Get-FileHash -LiteralPath $srcW -Algorithm SHA256).Hash
    robocopy 'E:\kaira_wex\v41-reap' 'U:\kaira_wex\v41-reap' 'v4_flash_experts.nr2.wex' /COPY:DAT /J /R:1 /W:1 /NP /NFL /NDL /NJH /NJS | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy WNR2 failed ($LASTEXITCODE)" }
    $hd = (Get-FileHash -LiteralPath $dstW -Algorithm SHA256).Hash
    if ($hs -ne $hd) { throw "WNR2 sha256 differs: $hs vs $hd" }
    Set-Content -LiteralPath ($dstW + '.verified') -Value "sha256 $hd  copied from $srcW  $(Get-Date -Format s)" -Encoding utf8
    Log "WNR2 verified on U: sha256 $hd"
    Log ("U: free at end {0:N1} GB" -f ((Get-PSDrive U).Free / 1e9))
}
finally {
    Remove-Item -LiteralPath $lock -Force -ErrorAction SilentlyContinue
    Log "lock released"
}
