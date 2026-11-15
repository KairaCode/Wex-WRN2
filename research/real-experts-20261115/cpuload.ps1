# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
# Total CPU load of the machine, all processes, about once a second (a sample takes 0.3 to 0.4 s itself, so in
# the run of 2026-11-15 there were 3 samples every 4 s): one line "HH:mm:ss,percent" per sample appended to OUT,
# until the file STOP exists. The value is Win32_PerfFormattedData_PerfOS_Processor of _Total, which Windows
# computes between two samples. Windows only; bench.sh starts it.
#   powershell -NoProfile -ExecutionPolicy Bypass -File cpuload.ps1 OUT STOP
param([string]$Out, [string]$Stop)
while (-not (Test-Path $Stop)) {
    $p = (Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter "Name='_Total'").PercentProcessorTime
    Add-Content -Path $Out -Value ("{0},{1}" -f (Get-Date -Format 'HH:mm:ss'), $p)
    Start-Sleep -Milliseconds 1000
}
