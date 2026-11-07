import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco, torch, numpy as np
from open_duck_isaac.constants import MJCF_PATH, START_POSE
from open_duck_isaac.kinematics import RobotKinematics
from open_duck_isaac.walker import BatchedZmpWalker, LegOdometry
m = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
kin = RobotKinematics(m, "cuda")
for n in [int(x) for x in sys.argv[1:]]:
  w = BatchedZmpWalker(kin, n); odo = LegOdometry(kin, n, 0.02)
  ids = torch.arange(n, device="cuda")
  q = torch.tensor(START_POSE, dtype=torch.float32, device="cuda").repeat(n, 1)
  rot = torch.eye(3, device="cuda").repeat(n, 1, 1)
  w.reset(ids, torch.zeros(n, 3, device="cuda"), rot, q); odo.reset(ids, w.feet0.clone())
  cmd = torch.zeros(n, 2, device="cuda"); cmd[:, 0] = 0.1
  torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
  for k in range(60):
    if k == 10: torch.cuda.synchronize(); tic = time.perf_counter()
    com, vcom = odo.update(rot, w.ik_q, w.contact)
    w.step(cmd, com, vcom, odo.foot_world)
  torch.cuda.synchronize()
  print(f"envs {n}: {1e3 * (time.perf_counter() - tic) / 50:.1f} ms/tick, peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
