# W2 REAP runs under heavy.lock: wait (60 s polls, up to $WaitMin minutes) for the lock, run
# run_w2.sh (which applies the RAM rule), always release the lock.
#   powershell -NoProfile -File locked_w2.ps1 DUMPDIR [ARMS] [WaitMin]
param([string]$Dump, [string]$Arms = 'A R B B2 R2', [int]$WaitMin = 90)
$here = '<KAIRA>\lab\v41-gpu-20261107'
$lock = "$here\heavy.lock"
$label = 'lane W (W2 batched WNR2): REAP GPU arms ' + $Arms + ' x w0,code1,chat_it1,reason1 at 128 tok, ~19 min'
$deadline = (Get-Date).AddMinutes($WaitMin)
$held = $false
while (-not $held) {
    try {
        New-Item -ItemType File $lock -ErrorAction Stop | Out-Null
        Set-Content -Path $lock -Value ("$label " + (Get-Date -Format s)) -Encoding utf8
        $held = $true
    } catch {
        if ((Get-Date) -gt $deadline) { "gave up after $WaitMin min, lock: " + (Get-Content $lock -ErrorAction SilentlyContinue); exit 3 }
        "lock busy at {0}: {1}" -f (Get-Date -Format T), (Get-Content $lock -ErrorAction SilentlyContinue)
        Start-Sleep -Seconds 60
    }
}
"lock: taken " + (Get-Date -Format s)
try {
    $env:PATH = 'C:\msys64\usr\bin;C:\msys64\ucrt64\bin;' + $env:PATH
    $env:DUMP = $Dump; $env:ARMS = $Arms
    & C:\msys64\usr\bin\bash.exe "$here\w2\run_w2_v2.sh"
} finally {
    Remove-Item $lock -ErrorAction SilentlyContinue
    "lock: released " + (Get-Date -Format s)
}
