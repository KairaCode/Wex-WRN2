# Run one phase script under heavy.lock: atomic create (poll 60 s, up to 180 min), label, run, always release.
#   powershell -NoProfile -File locked.ps1 PHASE.cmd LOGFILE
param([string]$Script, [string]$Log)
$lock = '<KAIRA>\lab\v41-gpu-20261107\heavy.lock'
$label = "wnr3-5090 zero bit-test $(Split-Path $Script -Leaf)"
$deadline = (Get-Date).AddMinutes(180)
$held = $false
while (-not $held) {
    try {
        New-Item -ItemType File $lock -ErrorAction Stop | Out-Null
        Set-Content -Path $lock -Value ("$label " + (Get-Date -Format s)) -Encoding utf8
        $held = $true
    } catch {
        if ((Get-Date) -gt $deadline) { "lock: gave up after 180 min (held by: " + (Get-Content $lock -ErrorAction SilentlyContinue) + ")"; exit 3 }
        "lock busy " + (Get-Date -Format s) + ": " + (Get-Content $lock -ErrorAction SilentlyContinue)
        Start-Sleep -Seconds 60
    }
}
"lock: taken " + (Get-Date -Format s)
$rc = 1
try {
    cmd /c "`"$Script`" > `"$Log`" 2>&1"
    $rc = $LASTEXITCODE
    "phase exit $rc"
} finally {
    Remove-Item $lock -ErrorAction SilentlyContinue
    "lock: released " + (Get-Date -Format s)
}
exit $rc
