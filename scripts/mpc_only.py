"""Run the Isaac Lab env with zero policy output (the MPC walker alone), or with a
trained residual policy, and print survival / velocity / roll statistics.

  run.bat scripts/mpc_only.py --num-envs 64 --vx 0.1 --seconds 20
  run.bat scripts/mpc_only.py --no-dr --no-delay
  run.bat scripts/mpc_only.py --no-dr --vx 0 --push 0.4                          # push sweep
  run.bat scripts/mpc_only.py --no-dr --set "step_gain=(0,0.3)" "mpc_fb_vel=(0,0)"  # walker overrides
  run.bat scripts/mpc_only.py --checkpoint logs/run1/model_1649.pt --vx 0.1
"""

import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
import torch  # noqa: F401  (Windows: load torch / rsl_rl DLLs before Kit)
import tensordict  # noqa: F401
import rsl_rl  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# DirectX 12 instead of Vulkan, like the other Isaac Lab project on this machine (fewer driver resets).
if not any(a.startswith("--/app/vulkan") for a in sys.argv):
  sys.argv.append("--/app/vulkan=false")
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--vx", type=float, default=None, help="fixed vx for all envs (default: random commands)")
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--no-dr", action="store_true", help="no randomization or pushes")
parser.add_argument("--no-delay", action="store_true")
parser.add_argument("--monitor", type=int, default=0, help="print stability stats every N steps")
parser.add_argument("--policy", default="", help="optional exported policy (TorchScript) instead of zero actions")
parser.add_argument("--checkpoint", default="", help="optional rsl_rl checkpoint (model_*.pt) instead of zero actions")
parser.add_argument("--set", nargs="*", action="extend", default=[], metavar="NAME=VALUE",
                    help="override WalkParams fields, e.g. --set step_gain=(0.4,0.3)")
parser.add_argument("--hip-roll", type=float, default=10.0, help="start pose hip roll [deg], 0 = flat soles")
parser.add_argument("--push", type=float, default=0.0,
                    help="push sweep: env i gets a base velocity kick of push * (i+1)/n [m/s] at t = 6, 9, 12, 15 s"
                         " (+x, -x, +y, -y); the report gives the smallest kick that made a robot fall")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

from open_duck_isaac.env import OpenDuckEnv, OpenDuckEnvCfg  # noqa: E402

cfg = OpenDuckEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.episode_length_s = args.seconds + 1.0
if args.no_dr:
  cfg.events = None
  cfg.gyro_noise = 0.0
  cfg.joint_noise = 0.0
if args.no_delay:
  cfg.max_delay_substeps = 0
cfg.start_hip_roll = args.hip_roll
if args.set:
  cfg.walk_overrides = {kv.split("=", 1)[0]: eval(kv.split("=", 1)[1]) for kv in args.set}
if args.vx is not None:
  cfg.vx_range = (args.vx, args.vx)
  cfg.vy_range = (0.0, 0.0)
  cfg.zero_cmd_prob = 0.0
env = OpenDuckEnv(cfg)
obs, _ = env.reset()
env.episode_length_buf[:] = 0
policy = torch.jit.load(args.policy, map_location=env.device) if args.policy else None
venv = None
if args.checkpoint:
  from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper, check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg,
  )

  from open_duck_isaac.agent_cfg import OpenDuckPPORunnerCfg

  agent_cfg = OpenDuckPPORunnerCfg()
  agent_cfg.device = env.device
  agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
  venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = create_rsl_rl_runner(venv, agent_cfg, log_dir=None)
  runner.load(str(Path(args.checkpoint).resolve()))
  policy = runner.get_inference_policy(device=env.device)
  obs = venv.get_observations()
  # The wrapper resets the env, which randomizes episode lengths: start all at 0 again
  # so no env times out (and gets moved back to its origin) during the test.
  env.episode_length_buf[:] = 0
  print(f"[eval] policy from {args.checkpoint}")
n = env.num_envs
start = env.robot.data.root_pos_w.torch[:, :2].clone()
alive = torch.ones(n, dtype=torch.bool, device=env.device)
fall_t = torch.full((n,), float("nan"), device=env.device)
roll_sq = torch.zeros(n, device=env.device)
steps = int(args.seconds / env.step_dt)
kick = args.push * torch.arange(1, n + 1, device=env.device) / n
kicks = {int(round(tp / env.step_dt)): d for tp, d in ((6.0, (1, 0)), (9.0, (-1, 0)), (12.0, (0, 1)), (15.0, (0, -1)))}
tic = time.perf_counter()
with torch.inference_mode():
  for k in range(steps):
    if args.push and k in kicks:
      vel = env.robot.data.root_vel_w.torch.clone()
      vel[:, 0] += kick * kicks[k][0]
      vel[:, 1] += kick * kicks[k][1]
      env.robot.write_root_velocity_to_sim_index(root_velocity=vel)
    if venv is not None:
      obs, _, _, _ = venv.step(policy(obs))
      term = env.reset_terminated.clone()  # falls only, not time-outs
    else:
      act = policy(obs["policy"]) if policy is not None else torch.zeros(n, 10, device=env.device)
      obs, rew, term, trunc, _ = env.step(act)
    newly = alive & term
    fall_t[newly] = k * env.step_dt
    alive &= ~term
    g = env.robot.data.projected_gravity_b.torch
    if args.monitor and k % args.monitor == 0:
      jv = env.robot.data.joint_vel.torch.abs()
      lv = env.robot.data.root_lin_vel_w.torch.norm(dim=1)
      av = env.robot.data.root_ang_vel_w.torch.norm(dim=1)
      fin = torch.isfinite(env.robot.data.root_pos_w.torch).all(1) & torch.isfinite(jv).all(1)
      jmax = torch.nan_to_num(jv, nan=1e9, posinf=1e9).max().item()
      print(f"step {k}: max|qd| {jmax:.1f} rad/s, max|v| {torch.nan_to_num(lv, nan=1e9).max().item():.2f} m/s, "
            f"max|w| {torch.nan_to_num(av, nan=1e9).max().item():.1f} rad/s, non-finite {(~fin).sum().item()}, "
            f"walker bad {env.walker.bad.sum().item()}, falls so far {(~alive).sum().item()}", flush=True)
    roll_sq += torch.where(alive, torch.square(torch.atan2(g[:, 1], -g[:, 2])), torch.zeros_like(roll_sq))
wall = time.perf_counter() - tic
moved = env.robot.data.root_pos_w.torch[:, :2] - start
cmd = env.cmd
print(f"{n} envs, {args.seconds:.0f} s sim in {wall:.1f} s wall ({n * steps / wall:,.0f} env-steps/s)")
print(f"survived {alive.float().mean().item() * 100:.1f}%  (falls at {fall_t[~alive].nanmean().item() if (~alive).any() else float('nan'):.1f} s mean)")
if args.push:
  pushed = ~alive & (fall_t >= 6.0)
  smallest = kick[pushed].min().item() if pushed.any() else float("nan")
  print(f"push sweep 0..{args.push:.2f} m/s: smallest kick that made a robot fall {smallest:.3f} m/s;"
        f" {alive.float().mean().item() * 100:.0f}% survived all 4 kicks")
ok = alive
if ok.any():
  v = moved[ok] / args.seconds
  print(f"survivors: vx {v[:, 0].mean().item():+.3f} m/s for cmd {cmd[ok, 0].mean().item():+.3f}, "
        f"vy {v[:, 1].mean().item():+.3f} for {cmd[ok, 1].mean().item():+.3f}; "
        f"roll rms {torch.rad2deg(torch.sqrt(roll_sq[ok] / steps)).mean().item():.1f} deg")
import os  # noqa: E402

os._exit(0)  # skip the Kit shutdown (it coincided with a GPU driver reset on this laptop)
