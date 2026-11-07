# Show training progress and GPU temperature/power every few seconds (read-only).
#   powershell -ExecutionPolicy Bypass -File scripts\watch_train.ps1 [-RunDir logs\run1] [-Every 10]
param([string]$RunDir = "$PSScriptRoot\..\logs\run1", [int]$Every = 10)

$keys = 'Learning iteration', 'Steps per second', 'Mean reward', 'Mean episode length', 'Mean action std',
        'Episode_Reward/track_vel', 'Episode_Reward/residual', 'Metrics/vel_error', 'Metrics/episode_s',
        'Episode_Termination/fall', 'Iteration time', 'ETA'
while ($true) {
  $log = Get-ChildItem "$RunDir\train_*.out" -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
  $gpu = (nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,clocks.gr --format=csv,noheader) -split ', '
  Clear-Host
  "{0:HH:mm:ss}   GPU {1} C   {2}   util {3}   VRAM {4}   clock {5}" -f (Get-Date), $gpu[0], $gpu[1], $gpu[2], $gpu[3], $gpu[4]
  if ($log) {
    "log: $($log.Name)"
    $tail = Get-Content $log.FullName -Tail 60
    $start = ($tail | Select-String 'Learning iteration' | Select-Object -Last 1).LineNumber
    if ($start) {
      foreach ($line in $tail[($start - 1)..($tail.Count - 1)]) {
        $t = ($line -replace '\x1b\[[0-9;]*m', '').Trim()
        if ($keys | Where-Object { $t.StartsWith($_) }) { "  $t" }
      }
    }
    $err = $tail | Select-String 'Traceback|CUDA error' | Select-Object -Last 1
    if ($err) { "  !! $($err.Line.Trim())" }
  }
  if (Test-Path "$RunDir\supervisor.log") { "supervisor: " + (Get-Content "$RunDir\supervisor.log" -Tail 1) }
  if (Test-Path "$RunDir\done.txt") { "DONE: " + (Get-Content "$RunDir\done.txt") }
  Start-Sleep -Seconds $Every
}
