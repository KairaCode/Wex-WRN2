# W3 pilots under heavy.lock: pilot C (ab\run_w3c.sh: WNR2 + RAM cache at equal bytes), then the first
# F1 gating grid (f1\run_f1.sh). Waits (60 s polls, up to $WaitMin minutes) for the lock, then until no
# build/encode/copy runs (nvcc, cl, make, gcc, robocopy, any python but the MCP server); always
# releases the lock (only our own).
#   powershell -NoProfile -File locked_pilots.ps1 DUMPDIR [WaitMin]
param([string]$Dump, [int]$WaitMin = 90)
$here = '<KAIRA>\lab\v41-gpu-20261107'
$lock = "$here\heavy.lock"
$label = 'W3 pilots: WNR2+RAM cache at equal bytes (4 runs) + F1 gating grid (6 runs), 4 prompts at 128 tok, ~50 min'
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
    $quietBy = (Get-Date).AddMinutes(60)
    while ($true) {
        $busy = @(Get-CimInstance Win32_Process | Where-Object {
            $_.Name -match '^(nvcc|cl|make|gcc|cc1|lto1|ld|robocopy|deepseek_v4)\.exe$' -or
            ($_.Name -match '^python' -and $_.CommandLine -notmatch 'TOOL MCP AI\\server\.py') })
        if ($busy.Count -eq 0) { break }
        if ((Get-Date) -gt $quietBy) { "machine not quiet after 60 min: " + (($busy | ForEach-Object { "$($_.Name) $($_.ProcessId)" }) -join ', '); exit 4 }
        "busy at {0}: {1}" -f (Get-Date -Format T), (($busy | ForEach-Object { "$($_.Name) $($_.ProcessId)" }) -join ', ')
        Start-Sleep -Seconds 60
    }
    "quiet: no build/encode/copy/python process at " + (Get-Date -Format T)
    powercfg /getactivescheme
    $env:PATH = 'C:\msys64\usr\bin;C:\msys64\ucrt64\bin;' + $env:PATH
    $env:DUMP = $Dump
    & C:\msys64\usr\bin\bash.exe "$here\ab\run_w3c.sh"
    if ($LASTEXITCODE -eq 0) { & C:\msys64\usr\bin\bash.exe "$here\f1\run_f1.sh" }
} finally {
    if ((Get-Content $lock -ErrorAction SilentlyContinue) -match '^W3 pilots') { Remove-Item $lock -ErrorAction SilentlyContinue }
    "lock: released " + (Get-Date -Format s)
}
