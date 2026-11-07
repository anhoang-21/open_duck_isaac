"""Run train.py until it finishes, restarting from the last checkpoint after a crash.

On this laptop a Windows GPU driver reset (TDR) kills the training process
with ``CUDA error: unknown error``. This supervisor waits for the driver to
recover, then resumes from the newest checkpoint (saved every 25 iterations).
It does not use the GPU itself.

  run.bat scripts/train_supervised.py --num-envs 8192 --iterations 3000 --run-dir logs/run1
"""

import argparse
import datetime
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def log(msg: str, path: Path) -> None:
  line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
  print(line, flush=True)
  with path.open("a", encoding="utf-8") as f:
    f.write(line + "\n")


# Performance cores of the Core Ultra 9 275HX (logical CPUs with the higher nominal
# clock in Windows' "Processor Information" counters). The training loop is bound by
# one Python thread launching small CUDA kernels; when Windows moves that thread to
# an E-core, training slows down by about 1/3. Child processes inherit the affinity.
P_CORES = [0, 1, 10, 11, 12, 13, 22, 23]


def pin_to_p_cores() -> None:
  try:
    import psutil

    p = psutil.Process()
    p.cpu_affinity(P_CORES)
    p.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
  except Exception as e:  # not fatal: training still works, only slower
    print(f"[supervisor] could not pin to P-cores: {e}", flush=True)


def main():
  pin_to_p_cores()
  ap = argparse.ArgumentParser()
  ap.add_argument("--num-envs", type=int, default=4096)
  ap.add_argument("--iterations", type=int, default=3000)
  ap.add_argument("--run-dir", default="logs/run")
  ap.add_argument("--max-restarts", type=int, default=30)
  ap.add_argument("--wait", type=float, default=30.0, help="seconds to let the driver recover")
  args = ap.parse_args()
  run_dir = (ROOT / args.run_dir).resolve()
  run_dir.mkdir(parents=True, exist_ok=True)
  sup_log = run_dir / "supervisor.log"
  for attempt in range(args.max_restarts + 1):
    if (run_dir / "done.txt").exists():
      log("done", sup_log)
      return
    cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), "--num-envs", str(args.num_envs),
           "--iterations", str(args.iterations), "--run-dir", args.run_dir]
    if any(run_dir.glob("model_*.pt")):
      cmd.append("--resume")
    n = 0
    while (run_dir / f"train_{n:02d}.out").exists():  # keep logs of earlier launches
      n += 1
    out = (run_dir / f"train_{n:02d}.out").open("w", encoding="utf-8")
    log(f"attempt {attempt}: {' '.join(cmd[1:])}", sup_log)
    tic = time.time()
    code = subprocess.call(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                           env={**__import__("os").environ, "OMNI_KIT_ACCEPT_EULA": "YES"})
    out.close()
    log(f"attempt {attempt} exited with code {code} after {time.time() - tic:.0f} s", sup_log)
    if (run_dir / "done.txt").exists():
      log("done", sup_log)
      return
    time.sleep(args.wait)
  log("too many restarts, giving up", sup_log)


if __name__ == "__main__":
  main()
