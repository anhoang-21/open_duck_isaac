"""Run the batched torch walker (batch of 1) on Open Duck in plain MuJoCo.

Same setup as mjlab_cd2/scripts/mpc_walk.py (crouch10 start pose, position
servos, leg odometry): checks that the GPU port walks like the original.

  python scripts/check_walker_mujoco.py --duration 30 --latency 0.02 --vx 0.1
  python scripts/check_walker_mujoco.py --ideal --vx 0      # ideal actuators, step in place
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
import torch

from open_duck_isaac.constants import FOOT_SITES, JOINT_NAMES, MJCF_PATH, START_POSE
from open_duck_isaac.kinematics import RobotKinematics, quat_to_mat
from open_duck_isaac.walker import BatchedZmpWalker, LegOdometry, WalkParams


IDEAL_MJCF_PATH = MJCF_PATH.with_name("open_duck_mini_v4_ideal.xml")


def build_model(path=MJCF_PATH) -> mujoco.MjModel:
  spec = mujoco.MjSpec.from_file(str(path))
  for g in spec.geoms:
    if g.name in ("left_foot_collision", "right_foot_collision"):
      g.contype, g.conaffinity, g.condim, g.priority = 1, 1, 3, 1
      g.friction = [0.8, 0.005, 0.0001]
  floor = spec.worldbody.add_geom()
  floor.type = mujoco.mjtGeom.mjGEOM_PLANE
  floor.size = [0, 0, 0.01]
  floor.contype = floor.conaffinity = 1
  return spec.compile()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--duration", type=float, default=20.0)
  ap.add_argument("--vx", type=float, default=0.1)
  ap.add_argument("--vy", type=float, default=0.0)
  ap.add_argument("--latency", type=float, default=0.0)
  ap.add_argument("--truth", action="store_true", help="true CoM instead of leg odometry")
  ap.add_argument("--video", default="")
  ap.add_argument("--push", type=float, default=0.0,
                  help="push the base with this force [N] for 0.1 s at t = 6, 9, 12, 15 s (+x, -x, +y, -y)")
  ap.add_argument("--trace", default="", help="save a per-tick trace (.npz) for plotting")
  ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE",
                  help="override WalkParams fields, e.g. --set period=0.5 q_vel=1e-5")
  ap.add_argument("--ideal", action="store_true", help="ideal actuators (open_duck_mini_v4_ideal.xml), no load compensation")
  args = ap.parse_args()
  m = build_model(IDEAL_MJCF_PATH if args.ideal else MJCF_PATH)
  d = mujoco.MjData(m)
  qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in JOINT_NAMES])
  base = m.body("base_link").id
  sites = [m.site(s).id for s in FOOT_SITES]
  p = WalkParams(load=0.0) if args.ideal else WalkParams()
  for kv in args.set:
    name, value = kv.split("=", 1)
    old = getattr(p, name)
    setattr(p, name, type(old)(eval(value)) if not isinstance(old, tuple) else tuple(eval(value)))
  d.qpos[3] = 1.0
  d.qpos[qadr] = START_POSE
  d.ctrl[:] = START_POSE
  mujoco.mj_forward(m, d)
  d.qpos[2] += 0.002 - min(d.site_xpos[s][2] for s in sites)
  for _ in range(int(0.3 / m.opt.timestep)):
    mujoco.mj_step(m, d)

  kin = RobotKinematics(m)
  walker = BatchedZmpWalker(kin, 1, p)
  odo = LegOdometry(kin, 1, p.dt)
  t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)[None]
  ids = torch.tensor([0])
  walker.reset(ids, t(d.qpos[:3]), quat_to_mat(t(d.qpos[3:7])), t(d.qpos[qadr]))
  odo.reset(ids, walker.feet0.clone())

  renderer = None
  frames = []
  if args.video:
    renderer = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = base; cam.distance = 0.9; cam.elevation = -15; cam.azimuth = 135
  dt = m.opt.timestep
  sub = int(round(p.dt / dt))
  x0 = d.subtree_com[base][0]
  pending = []
  rolls, fell = [], None
  trace = {k: [] for k in ("t", "rp", "com", "com_plan", "zmp_plan", "feet", "feet_plan", "contact_plan", "dcm_err")}
  for k in range(int(args.duration / p.dt)):
    rot = quat_to_mat(t(d.qpos[3:7]))
    if args.truth:
      mujoco.mj_subtreeVel(m, d)
      com, vcom = t(d.subtree_com[base]), t(d.subtree_linvel[base])
      feet = t([d.site_xpos[s] for s in sites])
    else:
      com, vcom = odo.update(rot, t(d.qpos[qadr]), walker.contact)
      feet = odo.foot_world.clone()
    ramp = min(1.0, max(0.0, (k * p.dt - 2.0) / 1.0))
    cmd = torch.tensor([[args.vx * ramp, args.vy * ramp]])
    q = walker.step(cmd, com, vcom, feet)[0].numpy()
    pending.append((k * p.dt, q))
    for s in range(sub):
      now = k * p.dt + s * dt
      while len(pending) > 1 and pending[1][0] <= now - args.latency:
        pending.pop(0)
      if pending[0][0] <= now - args.latency:
        d.ctrl[:] = pending[0][1]
      d.xfrc_applied[base, :3] = 0.0
      for tp, direction in ((6.0, (1, 0)), (9.0, (-1, 0)), (12.0, (0, 1)), (15.0, (0, -1))):
        if args.push and tp <= now < tp + 0.1:
          d.xfrc_applied[base, :2] = args.push * np.array(direction)
      mujoco.mj_step(m, d)
      if renderer is not None and (k * sub + s) % int(round(1 / (30 * dt))) == 0:
        renderer.update_scene(d, cam); frames.append(renderer.render())
    r = d.xmat[base].reshape(3, 3)
    roll, pitch = np.arctan2(r[2, 1], r[2, 2]), -np.arcsin(r[2, 0])
    rolls.append((roll, pitch))
    if args.trace:
      mujoco.mj_subtreeVel(m, d)
      trace["t"].append((k + 1) * p.dt)
      trace["rp"].append((roll, pitch))
      trace["com"].append(d.subtree_com[base].copy())
      trace["com_plan"].append(walker.out_com[0].numpy().copy())
      trace["zmp_plan"].append(walker.out_zmp[0].numpy().copy())
      trace["feet"].append(np.array([d.site_xpos[s] for s in sites]))
      trace["feet_plan"].append(walker.out_feet[0].numpy().copy())
      trace["contact_plan"].append(walker.contact[0].numpy().copy())
      trace["dcm_err"].append(walker.dcm_err[0].numpy().copy())
    if d.xpos[base][2] < 0.12 or abs(roll) > 0.7 or abs(pitch) > 0.7:
      fell = k * p.dt
      break
  rp = np.degrees(np.array(rolls))
  print(f"{'FELL at %.2f s' % fell if fell is not None else 'no fall'}; moved {d.subtree_com[base][0] - x0:+.3f} m;"
        f" roll max {np.abs(rp[:, 0]).max():.1f} rms {rp[150:, 0].std():.1f} deg, pitch max {np.abs(rp[:, 1]).max():.1f} deg")
  if args.trace:
    np.savez(args.trace, **{k: np.array(v) for k, v in trace.items()})
  if renderer is not None:
    import mediapy
    try:  # Windows: use the ffmpeg binary bundled with imageio-ffmpeg
      import imageio_ffmpeg
      mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
      pass
    mediapy.write_video(args.video, frames, fps=30)


if __name__ == "__main__":
  main()
