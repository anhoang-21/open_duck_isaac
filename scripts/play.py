"""Watch the trained MPC-residual policy (or the MPC alone) walk in the Isaac Sim window.

  run.bat scripts/play.py                                  # newest checkpoint in logs/run1, 16 robots
  run.bat scripts/play.py --num-envs 64 --vx 0.1           # 64 robots, all commanded 0.1 m/s forward
  run.bat scripts/play.py --checkpoint logs/run1/model_1649.pt --no-dr
  run.bat scripts/play.py --mpc-only                       # zero residual: the MPC walker alone
  run.bat scripts/play.py --headless --seconds 20          # no window, just the printed stats

The camera is free (right mouse + W/A/S/D/Q/E to fly, scroll to zoom); --follow makes it
follow robot 0 instead. Close the window (or Ctrl+C) to stop.
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
  sys.argv.append("--/app/vulkan=false")  # DirectX 12, like train.py
# Isaac Lab 3.0 opens the Isaac Sim window only when the Kit visualizer is requested.
if "--headless" not in sys.argv and not any(a.startswith(("--viz", "--visualizer")) for a in sys.argv):
  sys.argv += ["--viz", "kit"]

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=16, help="number of robots")
parser.add_argument("--checkpoint", default="", help="rsl_rl checkpoint (default: newest model_*.pt in --run-dir)")
parser.add_argument("--run-dir", default="logs/run1")
parser.add_argument("--mpc-only", action="store_true", help="zero policy output: the MPC walker alone")
parser.add_argument("--vx", type=float, default=None, help="fixed forward command [m/s] (default: random like training)")
parser.add_argument("--vy", type=float, default=0.0, help="fixed sideways command [m/s], used with --vx")
parser.add_argument("--no-dr", action="store_true", help="no randomization, pushes or sensor noise")
parser.add_argument("--no-delay", action="store_true", help="no servo delay")
parser.add_argument("--follow", action="store_true", help="camera follows robot 0 (default: free camera)")
parser.add_argument("--seconds", type=float, default=0.0, help="stop after this much sim time (0: until closed)")
parser.add_argument("--no-real-time", action="store_true", help="run as fast as possible")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app = AppLauncher(args).app

from isaaclab.envs import ViewerCfg  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
  RslRlVecEnvWrapper, check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg,
)

from open_duck_isaac.agent_cfg import OpenDuckPPORunnerCfg  # noqa: E402
from open_duck_isaac.env import OpenDuckEnv, OpenDuckEnvCfg  # noqa: E402


def newest_checkpoint(run_dir: Path) -> Path:
  best, it = None, -1
  for p in run_dir.glob("model_*.pt"):
    m = re.fullmatch(r"model_(\d+)\.pt", p.name)
    if m and int(m.group(1)) > it:
      best, it = p, int(m.group(1))
  if best is None:
    raise FileNotFoundError(f"no model_*.pt in {run_dir}")
  return best


def main():
  cfg = OpenDuckEnvCfg()
  cfg.scene.num_envs = args.num_envs
  if args.no_dr:
    cfg.events = None
    cfg.gyro_noise = 0.0
    cfg.joint_noise = 0.0
  if args.no_delay:
    cfg.max_delay_substeps = 0
  if args.vx is not None:
    cfg.vx_range = (args.vx, args.vx)
    cfg.vy_range = (args.vy, args.vy)
    cfg.zero_cmd_prob = 0.0
  if args.follow:
    cfg.viewer = ViewerCfg(eye=(0.7, 0.7, 0.35), lookat=(0.0, 0.0, 0.1), origin_type="asset_root",
                           asset_name="robot", env_index=0)
  env = OpenDuckEnv(cfg)

  agent_cfg = OpenDuckPPORunnerCfg()
  agent_cfg.device = env.device
  agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
  venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if args.mpc_only:
    def policy(_obs):
      return torch.zeros(env.num_envs, 10, device=env.device)
    print("[play] MPC only (zero residual)")
  else:
    ckpt = Path(args.checkpoint) if args.checkpoint else newest_checkpoint(ROOT / args.run_dir)
    runner = create_rsl_rl_runner(venv, agent_cfg, log_dir=None)
    runner.load(str(ckpt.resolve()))
    policy = runner.get_inference_policy(device=env.device)
    print(f"[play] policy {ckpt}")
  print(f"[play] {env.num_envs} robots, commands: "
        + (f"vx {args.vx:+.2f} vy {args.vy:+.2f} m/s" if args.vx is not None else "random (as in training)"))

  obs = venv.get_observations()
  dt = env.step_dt
  steps, falls, report_every = 0, 0, int(5.0 / dt)
  v_sum = torch.zeros(2, device=env.device)
  with torch.inference_mode():
    while app.is_running():
      tic = time.perf_counter()
      obs, _, _, _ = venv.step(policy(obs))
      falls += int(env.reset_terminated.sum().item())
      v_sum += env.robot.data.root_lin_vel_b.torch[0, :2]
      steps += 1
      if steps % report_every == 0:
        v = v_sum / report_every
        c = env.cmd[0]
        print(f"[play] t {steps * dt:6.1f} s  robot0 v ({v[0]:+.3f}, {v[1]:+.3f}) m/s for cmd "
              f"({c[0]:+.3f}, {c[1]:+.3f})  falls so far {falls}")
        v_sum.zero_()
      if args.seconds and steps * dt >= args.seconds:
        break
      if not args.no_real_time:
        time.sleep(max(0.0, dt - (time.perf_counter() - tic)))
  print(f"[play] {steps * dt:.1f} s sim, {falls} falls across {env.num_envs} robots")


if __name__ == "__main__":
  main()
  os._exit(0)  # skip the Kit shutdown (it coincided with a GPU driver reset on this laptop)
