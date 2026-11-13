<#
.SYNOPSIS
  Runs the WNR test kit: environment check, data check (optionally fetch), then every discovered runner,
  continuing on failure, and merges all results\*\summary.json into results\ALL_RESULTS.json.

.EXAMPLE
  # from the kit root, first pass (minutes):
  powershell -ExecutionPolicy Bypass -File .\setup\run_all.ps1 -Quick
  # full run, download data first if missing, also the extra 70+6 experts:
  powershell -ExecutionPolicy Bypass -File .\setup\run_all.ps1 -Fetch -Extra
  # data from local safetensors shards instead of the network:
  powershell -ExecutionPolicy Bypass -File .\setup\run_all.ps1 -LocalShards D:\models\V4.1-REAP192-MXFP4 -Extra
  # only some runners / skip GPU / list what would run:
  .\setup\run_all.ps1 -Only ceil,zero -NoGpu
  .\setup\run_all.ps1 -ListOnly
  # also the EXPERIMENTAL runners (KIT_META 'optional', e.g. gpu_v1_profile):
  .\setup\run_all.ps1 -Quick -All

.NOTES
  Python is taken from -Python, else <kit>\venv\Scripts\python.exe, else <kit>\venv-gpu\Scripts\python.exe (older
  README name), else 'python' on PATH. One venv for everything: setup\requirements-gpu.txt (includes the CPU set).
  -NoGpu skips only runners whose needs contain 'gpu' (nvrtc_check needs only 'nvrtc' and still runs).
  Optional (EXPERIMENTAL) runners run only with -All or when an -Only pattern names them.
  Every runner is called as: python -X utf8 -u -I <runner> --root <kit> [--data D] [--out O\<name>] [--quick]
  Per-runner stdout/stderr go to results\_runlogs\. -TimeoutMin kills a runner (and its children) after N minutes.
#>
[CmdletBinding()]
param(
    [switch]$Quick,
    [string]$Python = '',
    [string]$Data = '',
    [string]$Out = '',
    [string[]]$Only = @(),
    [string[]]$Skip = @(),
    [int]$TimeoutMin = 0,
    [switch]$Fetch,
    [switch]$Extra,
    [string]$LocalShards = '',
    [switch]$NoGpu,
    [switch]$All,
    [switch]$ListOnly
)

$ErrorActionPreference = 'Continue'
function Resolve-Full([string]$p) { return $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($p) }
$Root = Split-Path -Parent $PSScriptRoot
$Setup = Join-Path $Root 'setup'
$onWindows = ($env:OS -eq 'Windows_NT')

if (-not $Python) {
    $cands = @((Join-Path $Root 'venv\Scripts\python.exe'), (Join-Path $Root 'venv/bin/python'),
               (Join-Path $Root 'venv-gpu\Scripts\python.exe'), (Join-Path $Root 'venv-gpu/bin/python'))
    foreach ($c in $cands) { if (Test-Path $c) { $Python = $c; break } }
    if (-not $Python) {
        if (Get-Command python -ErrorAction SilentlyContinue) { $Python = 'python' }
        elseif (Get-Command python3 -ErrorAction SilentlyContinue) { $Python = 'python3' }
        else { Write-Host 'ERROR: no python found; pass -Python <path to python.exe>'; exit 2 }
    }
}
$pyver = & $Python -c "import sys; print(sys.version.split()[0])"
Write-Host "kit root : $Root"
Write-Host "python   : $Python ($pyver)"

$ResDir = if ($Out) { (Resolve-Full $Out) } else { Join-Path $Root 'results' }
$LogDir = Join-Path $ResDir '_runlogs'
$RunsFile = Join-Path $LogDir 'runs.jsonl'

function Quote-Arg([string]$a) {
    if ($a -eq '') { return '""' }
    if ($a -notmatch '[\s"]') { return $a }
    $b = $a -replace '"', '\"'
    if ($b.EndsWith('\')) { $b = $b + '\' }   # "C:\dir\" would escape the closing quote
    return '"' + $b + '"'
}

function Show-NewOutput([string]$path, [long]$pos, [bool]$final) {
    if (-not (Test-Path $path)) { return $pos }
    $fs = $null
    try {
        $fs = [System.IO.File]::Open($path, 'Open', 'Read', 'ReadWrite')
        if ($fs.Length -le $pos) { return $pos }
        [void]$fs.Seek($pos, 'Begin')
        $n = [int]($fs.Length - $pos)
        $buf = New-Object byte[] $n
        $got = $fs.Read($buf, 0, $n)
        $cut = $got
        if (-not $final) {
            $cut = [Array]::LastIndexOf($buf, [byte]10, $got - 1) + 1
            if ($cut -le 0) { return $pos }
        }
        $txt = [System.Text.Encoding]::UTF8.GetString($buf, 0, $cut)
        Write-Host -NoNewline $txt
        return $pos + $cut
    } catch {
        return $pos
    } finally {
        if ($fs) { $fs.Dispose() }
    }
}

function Invoke-Step([string]$Name, [string]$Script, [string[]]$ScriptArgs) {
    $stdout = Join-Path $LogDir "$Name.stdout.txt"
    $stderr = Join-Path $LogDir "$Name.stderr.txt"
    $all = @('-X', 'utf8', '-u', '-I', $Script) + $ScriptArgs
    $argLine = ($all | ForEach-Object { Quote-Arg $_ }) -join ' '
    Write-Host ''
    Write-Host ("=" * 100)
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $Name : $Python $argLine"
    Write-Host ("=" * 100)
    $t0 = Get-Date
    $timedOut = $false
    $exit = $null
    try {
        $p = Start-Process -FilePath $Python -ArgumentList $argLine -NoNewWindow -PassThru `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WorkingDirectory $Root
        $null = $p.Handle   # keep the handle so ExitCode is available after exit
        $pos = 0
        while (-not $p.HasExited) {
            Start-Sleep -Milliseconds 700
            $pos = Show-NewOutput $stdout $pos $false
            if ($TimeoutMin -gt 0 -and ((Get-Date) - $t0).TotalMinutes -gt $TimeoutMin) {
                Write-Host "TIMEOUT after $TimeoutMin min: killing $Name (pid $($p.Id))"
                if ($onWindows) { & taskkill /T /F /PID $p.Id | Out-Null } else { Stop-Process -Id $p.Id -Force }
                $timedOut = $true
                break
            }
        }
        $p.WaitForExit()
        $pos = Show-NewOutput $stdout $pos $true
        $exit = $p.ExitCode
    } catch {
        Write-Host "ERROR starting $Name : $_"
        $exit = -1
    }
    $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    $errTail = ''
    if (Test-Path $stderr) {
        $errLines = Get-Content $stderr -Tail 15 -ErrorAction SilentlyContinue
        if ($errLines) {
            $errTail = ($errLines -join "`n")
            Write-Host '--- stderr (tail) ---'
            Write-Host $errTail
        }
    }
    $resultLine = $null
    if (Test-Path $stdout) {
        $rl = Select-String -Path $stdout -Pattern '^RESULT ' -ErrorAction SilentlyContinue | Select-Object -Last 1
        if ($rl) { $resultLine = $rl.Line.Trim() }
    }
    if (-not $resultLine) { $resultLine = "RESULT $Name FAIL no RESULT line (exit $exit)" }
    $rec = [ordered]@{
        name = $Name; script = $Script; args = ($ScriptArgs -join ' '); exit_code = $exit; seconds = $secs
        timed_out = $timedOut; result_line = $resultLine; stdout_log = $stdout; stderr_log = $stderr
        stderr_tail = $errTail; finished = (Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')
    }
    ($rec | ConvertTo-Json -Compress) | Out-File -Append -Encoding utf8 -FilePath $RunsFile
    $script:Summary += [pscustomobject]@{ Step = $Name; Exit = $exit; Seconds = $secs; Result = $resultLine }
    return $exit
}

$script:Summary = @()
$common = @('--root', $Root)
if ($Data) { $common += @('--data', (Resolve-Full $Data)) }
$qk = @(); if ($Quick) { $qk = @('--quick') }

# ---------------------------------------------------------------- runner list
$listTxt = (& $Python -I (Join-Path $Setup 'kitenv.py') --root $Root --list-runners) -join ''
$runners = @()
try { $runners = ConvertFrom-Json -InputObject $listTxt } catch { Write-Host "ERROR: runner discovery failed: $listTxt" }
$sel = @()
foreach ($r in $runners) {
    $tag = "$($r.name) $($r.path)"
    $hit = $false
    if ($Only.Count -gt 0) {
        foreach ($o in $Only) { if ($tag -like "*$o*") { $hit = $true } }
        if (-not $hit) { continue }
    }
    $skipIt = $false; foreach ($s in $Skip) { if ($tag -like "*$s*") { $skipIt = $true } }
    if ($NoGpu -and (@($r.needs) -contains 'gpu')) { $skipIt = $true }
    if ($skipIt) { Write-Host "skip     : $($r.path)"; continue }
    $isOpt = ($null -ne $r.optional) -and [bool]$r.optional
    if ($isOpt -and -not $All -and -not $hit) { Write-Host "optional : $($r.path) (EXPERIMENTAL; add -All or -Only $($r.name))"; continue }
    $sel += $r
}
Write-Host "runners  : $($sel.Count) selected of $(@($runners).Count) discovered"
foreach ($r in $sel) {
    $mins = if ($Quick) { $r.quick_minutes } else { $r.full_minutes }
    $est = if ($null -ne $mins) { "~$mins min" } else { '' }
    Write-Host ("  {0,-4} {1,-44} needs={2,-12} {3}" -f $r.order, $r.path, (@($r.needs) -join ','), $est)
}
if ($ListOnly) { exit 0 }
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
if (Test-Path $RunsFile) { Move-Item -Force $RunsFile (Join-Path $LogDir 'runs.prev.jsonl') }

# ---------------------------------------------------------------- setup steps
$ceArgs = $common + @('--out', (Join-Path $ResDir 'check_env')) + $qk
[void](Invoke-Step 'check_env' (Join-Path $Setup 'check_env.py') $ceArgs)

$fdExtra = @(); if ($Extra) { $fdExtra = @('--extra') }
if ($Fetch -or $LocalShards) {
    $fa = $common + @('--out', (Join-Path $ResDir 'fetch_data')) + $fdExtra
    if ($LocalShards) { $fa += @('--local-shards', (Resolve-Full $LocalShards)) }
    [void](Invoke-Step 'fetch_data' (Join-Path $Setup 'fetch_data.py') $fa)
}
$va = $common + @('--verify-only', '--out', (Join-Path $ResDir 'verify_data')) + $fdExtra + $qk
$vexit = Invoke-Step 'verify_data' (Join-Path $Setup 'fetch_data.py') $va
if ($vexit -ne 0) {
    Write-Host 'WARNING: data verification failed (missing or corrupt files). Runners that need data will fail.'
    Write-Host '         Fetch with: python -I setup\fetch_data.py [--extra]   or   run_all.ps1 -Fetch'
}

# ---------------------------------------------------------------- runners
foreach ($r in $sel) {
    $ra = $common
    if ($Out) { $ra += @('--out', (Join-Path $ResDir $r.name)) }
    $ra += $qk
    $nm = ($r.path -replace '[\\/:]', '_') -replace '\.py$', ''
    [void](Invoke-Step $nm (Join-Path $Root $r.path) $ra)
}

# ---------------------------------------------------------------- collect
$ca = @('--root', $Root, '--results', $ResDir, '--runs', $RunsFile)
& $Python -X utf8 -I (Join-Path $Setup 'collect_results.py') @ca

Write-Host ''
Write-Host ("=" * 100)
$script:Summary | Format-Table -AutoSize -Wrap | Out-String -Width 220 | Write-Host
$failed = @($script:Summary | Where-Object { $_.Exit -ne 0 })
Write-Host "steps: $($script:Summary.Count), nonzero exit: $($failed.Count). All results: $(Join-Path $ResDir 'ALL_RESULTS.json')"
Write-Host 'Send back with: powershell -ExecutionPolicy Bypass -File .\setup\pack_results.ps1'
if ($failed.Count -gt 0) { exit 1 } else { exit 0 }
