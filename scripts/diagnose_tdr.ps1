# Diagnose the GPU driver resets (TDR, nvlddmkm event 153) during RL training.
#
# Runs a series of tests one after another and records for each one whether a
# TDR happened, how long it ran, and the peak GPU clock / power / temperature:
#   cuda_steady   pure CUDA, large matmuls (full steady load, like FurMark)     no Isaac
#   cuda_burst    pure CUDA, many small kernels + sync (like the MPC walker)    no Isaac
#   isaac_4096 / isaac_8192 / isaac_16384   the real training at that env count,
#                 resumed from logs/run1/model_325.pt into logs/diag/<test>
#
#   powershell -ExecutionPolicy Bypass -File scripts\diagnose_tdr.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\diagnose_tdr.ps1 -Tests cuda_burst,isaac_8192 -Minutes 15
#
# Results: logs\diag\summary.csv (+ one .out log and one clocks .csv per test).
# Close games, Chrome, NVIDIA overlay and Flydigi first, and do not use the GPU while it runs.
param(
  [string[]]$Tests = @('cuda_steady', 'cuda_burst', 'isaac_4096', 'isaac_8192', 'isaac_16384'),
  [double]$Minutes = 10,
  [string]$Checkpoint = 'logs\run1\model_325.pt'
)

$ErrorActionPreference = 'Stop'
# "powershell -File" passes "-Tests a,b" as one string: split it.
$Tests = @($Tests | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$root = Resolve-Path "$PSScriptRoot\.."
$py = 'C:\Users\ADMIN\Isaacsim\IsaacLab\env_isaaclab\Scripts\python.exe'
$diag = Join-Path $root 'logs\diag'
New-Item -ItemType Directory -Force $diag | Out-Null
$env:OMNI_KIT_ACCEPT_EULA = 'YES'
# Seconds per training iteration measured on this laptop (used to size each run).
$secPerIt = @{ 4096 = 1.4; 8192 = 2.0; 16384 = 2.3 }

function Get-TdrCount {
  @(Get-WinEvent -FilterHashtable @{ LogName = 'System'; ProviderName = 'nvlddmkm'; Id = 153 } -ErrorAction SilentlyContinue).Count
}

$summaryPath = Join-Path $diag 'summary.csv'
# Keep rows of earlier runs (e.g. when re-running only some tests).
$summary = @(if (Test-Path $summaryPath) { Import-Csv $summaryPath | Where-Object { $Tests -notcontains $_.test } })
foreach ($test in $Tests) {
  $out = Join-Path $diag "$test.out"
  $clk = Join-Path $diag "$test.clocks.csv"
  if ($test -like 'cuda_*') {
    $mode = $test.Substring(5)
    $argList = @("`"$root\scripts\cuda_burst_test.py`"", '--mode', $mode, '--minutes', $Minutes, '--batch', 16384)
    $startup = 1
  } elseif ($test -match '^isaac_(\d+)$') {
    $n = [int]$Matches[1]
    $runDir = "logs\diag\$test"
    Remove-Item -Recurse -Force (Join-Path $root $runDir) -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force (Join-Path $root $runDir) | Out-Null
    Copy-Item (Join-Path $root $Checkpoint) (Join-Path $root $runDir)
    $it0 = [int]([regex]::Match($Checkpoint, 'model_(\d+)\.pt').Groups[1].Value)
    $iters = $it0 + [int]($Minutes * 60 / $secPerIt[$n])
    $argList = @("`"$root\scripts\train.py`"", '--num-envs', $n, '--iterations', $iters, '--run-dir', $runDir, '--resume')
    $startup = 3  # Isaac Sim start + scene creation
  } else {
    Write-Warning "unknown test $test"; continue
  }

  $tdr0 = Get-TdrCount
  Write-Host ("{0:HH:mm:ss} === {1} ({2} min) ===" -f (Get-Date), $test, $Minutes)
  # GPU sampler in a background job: clock, power, temperature, util, VRAM every 2 s.
  $sampler = Start-Job -ArgumentList $clk -ScriptBlock {
    param($f)
    'time,clock_mhz,power_w,temp_c,util_pct,vram_mib' | Out-File $f -Encoding utf8
    while ($true) {
      $s = (nvidia-smi --query-gpu=clocks.gr,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader,nounits) -replace ' ', ''
      "{0:HH:mm:ss},{1}" -f (Get-Date), $s | Out-File $f -Append -Encoding utf8
      Start-Sleep -Seconds 2
    }
  }
  $t0 = Get-Date
  $p = Start-Process -FilePath $py -ArgumentList $argList -WorkingDirectory $root -NoNewWindow -PassThru `
    -RedirectStandardOutput $out -RedirectStandardError "$out.err"
  $null = $p.Handle  # without this, ExitCode is empty after the process ends
  $limitMs = [int](($Minutes + $startup + 6) * 60 * 1000)
  $hung = -not $p.WaitForExit($limitMs)
  if ($hung) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; Start-Sleep 5 }
  $secs = [int]((Get-Date) - $t0).TotalSeconds
  Stop-Job $sampler; Remove-Job $sampler
  Start-Sleep -Seconds 5
  $tdr = (Get-TdrCount) - $tdr0
  $ErrorActionPreference = 'Continue'  # a parsing problem below must not stop the remaining tests

  $rows = Import-Csv $clk | Where-Object { $_.clock_mhz -match '^\d+$' }
  $stat = { param($col) ($rows | Measure-Object -Property $col -Maximum).Maximum }
  $log = (Get-Content $out -ErrorAction SilentlyContinue) + (Get-Content "$out.err" -ErrorAction SilentlyContinue)
  $m = $log | Select-String 'Learning iteration (\d+)' | Select-Object -Last 1
  $lastIt = if ($m) { $m.Matches[0].Groups[1].Value } else { '' }  # the cuda_* tests have no iterations
  $err = ($log | Select-String 'CUDA error|FAILED|Traceback' | Select-Object -First 1).Line
  $result = if ($tdr -gt 0) { 'TDR' } elseif ($hung) { 'HUNG' } elseif ($p.ExitCode -eq 0) { 'PASS' } else { "EXIT $($p.ExitCode)" }
  $row = [pscustomobject]@{
    test = $test; result = $result; tdr_events = $tdr; seconds = $secs; exit_code = $p.ExitCode
    last_iteration = $lastIt; max_clock_mhz = (& $stat 'clock_mhz'); max_power_w = (& $stat 'power_w')
    max_temp_c = (& $stat 'temp_c'); max_vram_mib = (& $stat 'vram_mib'); first_error = $err
  }
  $summary += $row
  $row | Format-List | Out-String | Write-Host
  $summary | Export-Csv $summaryPath -NoTypeInformation -Encoding utf8
  # Let the driver recover / the GPU cool down before the next test.
  Start-Sleep -Seconds $(if ($tdr -gt 0) { 45 } else { 20 })
  $ErrorActionPreference = 'Stop'
}

Write-Host "`n===== SUMMARY ====="
$summary | Format-Table test, result, tdr_events, seconds, last_iteration, max_clock_mhz, max_power_w, max_temp_c, max_vram_mib -AutoSize | Out-String | Write-Host
Write-Host "Saved: $diag\summary.csv"
