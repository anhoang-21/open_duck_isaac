"""Check the torch kinematics against MuJoCo (FK, CoM, Jacobians, gravity torques)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
import torch
from open_duck_isaac.constants import MJCF_PATH, JOINT_NAMES, FOOT_SITES, BASE_BODY
from open_duck_isaac.kinematics import RobotKinematics, quat_to_mat

m = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
d = mujoco.MjData(m)
kin = RobotKinematics(m)
rng = np.random.default_rng(0)
qadr = [m.jnt_qposadr[m.joint(n).id] for n in JOINT_NAMES]
dadr = [m.jnt_dofadr[m.joint(n).id] for n in JOINT_NAMES]
base = m.body(BASE_BODY).id
errs = []
for trial in range(5):
  d.qpos[:3] = rng.normal(size=3) * 0.1
  quat = rng.normal(size=4); quat /= np.linalg.norm(quat)
  d.qpos[3:7] = quat
  d.qpos[qadr] = rng.uniform(-0.4, 0.4, size=10)
  d.qvel[:] = 0
  mujoco.mj_forward(m, d)
  t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)[None]
  ks = kin.fk(t(d.qpos[:3]), quat_to_mat(t(quat)), t(d.qpos[qadr]))
  sites = np.array([d.site_xpos[m.site(s).id] for s in FOOT_SITES])
  srot = np.array([d.site_xmat[m.site(s).id].reshape(3, 3) for s in FOOT_SITES])
  e_site = np.abs(ks.site_pos[0].numpy() - sites).max()
  e_rot = np.abs(ks.site_rot[0].numpy() - srot).max()
  e_com = np.abs(ks.com[0].numpy() - d.subtree_com[base]).max()
  # Jacobian of the left foot site wrt joints vs MuJoCo.
  jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
  mujoco.mj_jacSite(m, d, jp, jr, m.site(FOOT_SITES[0]).id)
  jt = kin.point_jac(ks, ks.site_pos[:, 0], kin.site_desc[0], t(d.qpos[:3]))[0].numpy()
  e_jac = np.abs(jt[:, 6:] - jp[:, dadr]).max()
  jc = np.zeros((3, m.nv)); mujoco.mj_jacSubtreeCom(m, d, jc, base)
  e_jcom = np.abs(kin.com_jac(ks, t(d.qpos[:3]))[0].numpy()[:, 6:] - jc[:, dadr]).max()
  # Gravity torques with no ground force vs qfrc_bias.
  tau = kin.leg_torques(ks, torch.zeros(1, 2, 3), None)[0].numpy()
  e_tau = np.abs(tau - d.qfrc_bias[dadr]).max()
  errs.append((e_site, e_rot, e_com, e_jac, e_jcom, e_tau))
print("max errors site/rot/com/jac/jcom/tau:", np.max(errs, axis=0))
print("mass", kin.total_mass)
