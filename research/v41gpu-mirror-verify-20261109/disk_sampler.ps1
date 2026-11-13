# 1 s sampler of PhysicalDisk raw cumulative read bytes for disk 3 (C:) and disk 5 (U:). Killed by host_launch.py.
param([string]$Csv)
"time,unix_ms,C_read_bytes,U_read_bytes" | Out-File -FilePath $Csv -Encoding ascii
while ($true) {
    $d = @{}
    Get-CimInstance Win32_PerfRawData_PerfDisk_PhysicalDisk -Filter "Name='3 C:' OR Name='5 U:'" | ForEach-Object { $d[$_.Name] = $_.DiskReadBytesPersec }
    $now = Get-Date
    "{0},{1},{2},{3}" -f $now.ToString('HH:mm:ss.fff'), [DateTimeOffset]::Now.ToUnixTimeMilliseconds(), $d['3 C:'], $d['5 U:'] | Out-File -FilePath $Csv -Append -Encoding ascii
    Start-Sleep -Milliseconds 1000
}
