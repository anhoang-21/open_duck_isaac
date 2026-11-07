"""Train the MPC-residual walking policy with PPO (rsl_rl) in Isaac Lab.

  run.bat scripts/train.py --num-envs 8192 --iterations 3000 --run-dir logs/run1
  run.bat scripts/train.py --run-dir logs/run1 --iterations 3000 --resume   # continue from the last checkpoint

``--resume`` picks up the newest ``model_<it>.pt`` in the run folder and trains
until iteration ``--iterations``. ``train_supervised.py`` restarts this script
that way after a crash (e.g. a GPU driver reset). ``done.txt`` marks the end.
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
import torch  # noqa: F401  (Windows: load torch / rsl_rl DLLs before Kit)
import tensordict  # noqa: F401
import rsl_rl  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if not any(a.startswith("--/app/vulkan") for a in sys.argv):
  sys.argv.append("--/app/vulkan=false")  # DirectX 12 (fewer driver resets on this machine)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=3000, help="train until this iteration")
parser.add_argument("--run-dir", default="logs/run")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

from isaaclab_rl.rsl_rl import (  # noqa: E402
  RslRlVecEnvWrapper, check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg,
)

from open_duck_isaac.agent_cfg import OpenDuckPPORunnerCfg  # noqa: E402
from open_duck_isaac.env import OpenDuckEnv, OpenDuckEnvCfg  # noqa: E402


def latest_checkpoint(run_dir: Path) -> tuple[Path | None, int]:
  best, it = None, -1
  for p in run_dir.glob("model_*.pt"):
    m = re.fullmatch(r"model_(\d+)\.pt", p.name)
    if m and int(m.group(1)) > it:
      best, it = p, int(m.group(1))
  return best, it


def main():
  run_dir = (ROOT / args.run_dir).resolve()
  run_dir.mkdir(parents=True, exist_ok=True)
  env_cfg = OpenDuckEnvCfg()
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  agent_cfg = OpenDuckPPORunnerCfg()
  agent_cfg.seed = args.seed
  agent_cfg.device = env_cfg.sim.device
  agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())

  env = OpenDuckEnv(env_cfg)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = create_rsl_rl_runner(env, agent_cfg, log_dir=str(run_dir))

  start = 0
  if args.resume:
    ckpt, it = latest_checkpoint(run_dir)
    if ckpt is not None:
      print(f"[train] resuming from {ckpt}")
      runner.load(str(ckpt))
      start = runner.current_learning_iteration
  remaining = args.iterations - start
  print(f"[train] {args.num_envs} envs, iterations {start} -> {args.iterations}, logs in {run_dir}")
  if remaining > 0:
    tic = time.time()
    runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=True)
    print(f"[train] {remaining} iterations in {time.time() - tic:.0f} s")
  (run_dir / "done.txt").write_text(f"finished at iteration {args.iterations}\n")


if __name__ == "__main__":
  main()
  # Skip env.close()/app.close(): on this laptop a Kit shutdown after a 16k-env run
  # coincided with a GPU driver reset (TDR). Checkpoints are already on disk; the
  # OS frees the GPU memory when the process exits.
  os._exit(0)
