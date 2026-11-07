"""Batched forward kinematics, Jacobians and whole-body IK in torch.

The kinematic tree, masses and inertias are read once from the MuJoCo model
(the same MJCF the Isaac Lab asset is converted from), so the walker plans on
exactly the robot it controls. Everything is batched over environments and
runs on the GPU during training and on the CPU on the robot (batch of one).

Generalized coordinates of the IK: base position (3), a world-frame rotation
increment of the base (3) and the 10 leg joints in ``JOINT_NAMES`` order.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from open_duck_isaac.constants import BASE_BODY, FOOT_SITES, JOINT_NAMES

GRAVITY = 9.81


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
  """(..., 4) wxyz -> (..., 3, 3)."""
  w, x, y, z = q.unbind(-1)
  return torch.stack(
    [
      1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
      2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
      2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ],
    -1,
  ).reshape(*q.shape[:-1], 3, 3)


def axis_angle_to_mat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
  """Rotation about unit ``axis`` (3,) or (..., 3) by ``angle`` (...)."""
  axis = axis.expand(*angle.shape, 3)
  k = skew(axis)
  s = torch.sin(angle)[..., None, None]
  c = torch.cos(angle)[..., None, None]
  eye = torch.eye(3, device=angle.device, dtype=angle.dtype)
  return eye + s * k + (1 - c) * (k @ k)


def rotvec_to_mat(v: torch.Tensor) -> torch.Tensor:
  angle = torch.linalg.norm(v, dim=-1).clamp_min(1e-12)
  return axis_angle_to_mat(v / angle[..., None], angle)


def skew(v: torch.Tensor) -> torch.Tensor:
  x, y, z = v.unbind(-1)
  o = torch.zeros_like(x)
  return torch.stack([o, -z, y, z, o, -x, -y, x, o], -1).reshape(*v.shape[:-1], 3, 3)


def rot_error(r_des: torch.Tensor, r_cur: torch.Tensor) -> torch.Tensor:
  """Small-angle rotation vector taking r_cur to r_des, world frame."""
  e = r_des @ r_cur.transpose(-1, -2)
  return 0.5 * torch.stack(
    [e[..., 2, 1] - e[..., 1, 2], e[..., 0, 2] - e[..., 2, 0], e[..., 1, 0] - e[..., 0, 1]], -1
  )


def remove_yaw(r: torch.Tensor) -> torch.Tensor:
  yaw = torch.atan2(r[..., 1, 0], r[..., 0, 0])
  z = torch.tensor([0.0, 0.0, 1.0], device=r.device, dtype=r.dtype)
  return axis_angle_to_mat(z, -yaw) @ r


@dataclass
class KinState:
  body_pos: torch.Tensor  # (B, nb, 3), bodies 0..nb-1 = base + legs
  body_rot: torch.Tensor  # (B, nb, 3, 3)
  com_body: torch.Tensor  # (B, nb, 3), each body's own CoM
  com: torch.Tensor  # (B, 3), whole robot
  site_pos: torch.Tensor  # (B, 2, 3) foot sites
  site_rot: torch.Tensor  # (B, 2, 3, 3)
  axis: torch.Tensor  # (B, 10, 3) joint axes, world
  anchor: torch.Tensor  # (B, 10, 3) joint positions, world


@dataclass
class IkWeights:
  com: float = 100.0
  foot_pos: float = 100.0
  foot_rot: float = 30.0
  base_rot: tuple[float, float, float] = (0.05, 5.0, 5.0)
  posture: float = 1e-3
  damping: float = 1e-4
  iterations: int = 8


class RobotKinematics:
  def __init__(self, model: mujoco.MjModel, device: str | torch.device = "cpu"):
    self.device = torch.device(device)
    base = model.body(BASE_BODY).id
    # Bodies of the robot in tree order (parents first).
    bodies = [b for b in range(model.nbody) if b == base or self._is_descendant(model, b, base)]
    self.bodies = bodies
    index = {b: i for i, b in enumerate(bodies)}
    self.parent = [index.get(int(model.body_parentid[b]), -1) for b in bodies]
    self.parent[0] = -1
    jnames = list(JOINT_NAMES)
    self.joint_of_body = [-1] * len(bodies)
    axis = np.zeros((len(jnames), 3))
    self.joint_body = [0] * len(jnames)
    for i, b in enumerate(bodies[1:], start=1):
      j = model.body_jntadr[b]
      if model.body_jntnum[b] != 1 or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
        raise ValueError(f"body {model.body(b).name}: expected one hinge joint")
      if np.linalg.norm(model.jnt_pos[j]) > 1e-9:
        raise ValueError("joint anchors must be at the body origin")
      ji = jnames.index(model.joint(j).name)
      self.joint_of_body[i] = ji
      self.joint_body[ji] = i
      axis[ji] = model.jnt_axis[j]
    f = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=self.device)
    self.body_pos0 = f(model.body_pos[bodies])
    rot0 = np.zeros((len(bodies), 9))
    for i, b in enumerate(bodies):
      mujoco.mju_quat2Mat(rot0[i], model.body_quat[b])
    self.body_rot0 = f(rot0.reshape(-1, 3, 3))
    self.joint_axis = f(axis)
    self.mass = f(model.body_mass[bodies])
    self.total_mass = float(self.mass.sum())
    self.ipos = f(model.body_ipos[bodies])
    sites = [model.site(s).id for s in FOOT_SITES]
    self.site_body = [index[int(model.site_bodyid[s])] for s in sites]
    self.site_pos0 = f(model.site_pos[sites])
    srot = np.zeros((2, 9))
    for i, s in enumerate(sites):
      mujoco.mju_quat2Mat(srot[i], model.site_quat[s])
    self.site_rot0 = f(srot.reshape(2, 3, 3))
    jids = [model.joint(n).id for n in jnames]
    self.qmin = f(model.jnt_range[jids, 0])
    self.qmax = f(model.jnt_range[jids, 1])
    # desc[j, b] = 1 if body b moves with joint j.
    nb = len(bodies)
    desc = np.zeros((len(jnames), nb))
    for ji, bj in enumerate(self.joint_body):
      for b in range(nb):
        k = b
        while k != -1:
          if k == bj:
            desc[ji, b] = 1.0
            break
          k = self.parent[k]
    self.desc = f(desc)
    self.site_desc = self.desc[:, self.site_body].T  # (2, 10)
    # Joint ordering per leg (L: 0..4, R: 5..9).
    self.legs = (torch.arange(0, 5, device=self.device), torch.arange(5, 10, device=self.device))

  @staticmethod
  def _is_descendant(model: mujoco.MjModel, b: int, root: int) -> bool:
    while b > 0:
      b = int(model.body_parentid[b])
      if b == root:
        return True
    return False

  # ------------------------------------------------------------------ FK
  def fk(self, base_pos: torch.Tensor, base_rot: torch.Tensor, q: torch.Tensor) -> KinState:
    bsz = q.shape[0]
    nb = len(self.bodies)
    pos = [base_pos]
    rot = [base_rot]
    axis = [None] * 10
    anchor = [None] * 10
    for i in range(1, nb):
      p = self.parent[i]
      r_fixed = rot[p] @ self.body_rot0[i]
      x = pos[p] + (rot[p] @ self.body_pos0[i])
      ji = self.joint_of_body[i]
      a_loc = self.joint_axis[ji]
      r = r_fixed @ axis_angle_to_mat(a_loc, q[:, ji])
      axis[ji] = r_fixed @ a_loc
      anchor[ji] = x
      pos.append(x)
      rot.append(r)
    body_pos = torch.stack(pos, 1)
    body_rot = torch.stack(rot, 1)
    com_body = body_pos + (body_rot @ self.ipos[None, :, :, None]).squeeze(-1)
    com = (self.mass[None, :, None] * com_body).sum(1) / self.total_mass
    sb = self.site_body
    site_pos = body_pos[:, sb] + (body_rot[:, sb] @ self.site_pos0[None, :, :, None]).squeeze(-1)
    site_rot = body_rot[:, sb] @ self.site_rot0[None]
    return KinState(
      body_pos, body_rot, com_body, com, site_pos, site_rot,
      torch.stack(axis, 1).expand(bsz, 10, 3), torch.stack(anchor, 1),
    )

  # ------------------------------------------------------------ Jacobians
  def point_jac(self, ks: KinState, p: torch.Tensor, desc: torch.Tensor, base_pos: torch.Tensor) -> torch.Tensor:
    """Jacobian (B, 3, 16) of world point(s) p (B, 3) that move with the joints in ``desc`` (10,)."""
    bsz = p.shape[0]
    eye = torch.eye(3, device=p.device).expand(bsz, 3, 3)
    j_rot = -skew(p - base_pos)  # d p = dtheta x (p - base)
    j_q = torch.cross(ks.axis, p[:, None, :] - ks.anchor, dim=-1) * desc[None, :, None]
    return torch.cat([eye, j_rot, j_q.transpose(1, 2)], 2)

  def com_jac(self, ks: KinState, base_pos: torch.Tensor) -> torch.Tensor:
    bsz = base_pos.shape[0]
    w = self.mass / self.total_mass  # (nb,)
    eye = torch.eye(3, device=base_pos.device).expand(bsz, 3, 3)
    j_rot = -skew(ks.com - base_pos)
    # sum_b w_b desc[j, b] (c_b - anchor_j) = (sum_b w_b desc c_b) - (sum_b w_b desc) anchor_j
    wd = self.desc * w[None, :]  # (10, nb)
    mc = torch.einsum("jb,nbk->njk", wd, ks.com_body)
    msum = wd.sum(1)[None, :, None]
    j_q = torch.cross(ks.axis, mc - msum * ks.anchor, dim=-1)
    return torch.cat([eye, j_rot, j_q.transpose(1, 2)], 2)

  def rot_jac(self, ks: KinState, desc: torch.Tensor | None) -> torch.Tensor:
    bsz = ks.axis.shape[0]
    z = torch.zeros(bsz, 3, 3, device=ks.axis.device)
    eye = torch.eye(3, device=ks.axis.device).expand(bsz, 3, 3)
    if desc is None:  # base orientation
      return torch.cat([z, eye, torch.zeros(bsz, 3, 10, device=z.device)], 2)
    return torch.cat([z, eye, (ks.axis * desc[None, :, None]).transpose(1, 2)], 2)

  # ------------------------------------------------------------------- IK
  def ik(
    self,
    base_pos: torch.Tensor,
    base_rot: torch.Tensor,
    q: torch.Tensor,
    com_des: torch.Tensor,
    feet_des: torch.Tensor,
    feet_rot_des: torch.Tensor,
    q_home: torch.Tensor,
    w: IkWeights,
    iterations: int | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, KinState]:
    """Damped least squares from the given (warm-start) configuration."""
    dev = q.device
    bsz = q.shape[0]
    wt = torch.cat([
      torch.full((3,), w.com), torch.full((3,), w.foot_pos), torch.full((3,), w.foot_rot),
      torch.full((3,), w.foot_pos), torch.full((3,), w.foot_rot), torch.tensor(w.base_rot),
      torch.full((10,), w.posture),
    ]).to(dev)
    eye_r = torch.eye(3, device=dev).expand(bsz, 3, 3)
    post = torch.cat([torch.zeros(10, 6, device=dev), torch.eye(10, device=dev)], 1).expand(bsz, 10, 16)
    damp = w.damping * torch.eye(16, device=dev)
    for _ in range(iterations or w.iterations):
      ks = self.fk(base_pos, base_rot, q)
      rows = [self.com_jac(ks, base_pos)]
      errs = [com_des - ks.com]
      for i in range(2):
        rows.append(self.point_jac(ks, ks.site_pos[:, i], self.site_desc[i], base_pos))
        errs.append(feet_des[:, i] - ks.site_pos[:, i])
        rows.append(self.rot_jac(ks, self.site_desc[i]))
        errs.append(rot_error(feet_rot_des[:, i], ks.site_rot[:, i]))
      rows.append(self.rot_jac(ks, None))
      errs.append(rot_error(eye_r, base_rot))
      rows.append(post)
      errs.append(q_home - q)
      jac = torch.cat(rows, 1)  # (B, 28, 16)
      err = torch.cat(errs, 1)
      jtw = jac.transpose(1, 2) * wt[None, None, :]
      dx = torch.linalg.solve_ex(jtw @ jac + damp, (jtw @ err[..., None]))[0][..., 0]
      dx = torch.where(torch.isfinite(dx).all(1, keepdim=True), dx, torch.zeros_like(dx))
      base_pos = base_pos + dx[:, :3]
      base_rot = rotvec_to_mat(dx[:, 3:6]) @ base_rot
      q = torch.clamp(q + dx[:, 6:], self.qmin, self.qmax)
    ks = self.fk(base_pos, base_rot, q)
    return base_pos, base_rot, q, ks

  # ------------------------------------------------------- gravity / loads
  def leg_torques(self, ks: KinState, forces: torch.Tensor, moments: torch.Tensor | None) -> torch.Tensor:
    """Joint torques (B, 10) to hold the legs against gravity and ground forces.

    ``tau_j = a_j . ((c_j - p_j) x m_j g z) - a_j . ((p_foot - p_j) x F - M)``
    where c_j, m_j are the CoM and mass of everything below joint j, F, M
    the ground force and moment on that leg's foot (world frame, on the robot).
    """
    w = self.mass  # (nb,)
    wd = self.desc * w[None, :]
    msum = wd.sum(1)[None, :, None]  # (1, 10, 1)
    mc = torch.einsum("jb,nbk->njk", wd, ks.com_body)  # sum m c
    up = torch.tensor([0.0, 0.0, GRAVITY], device=ks.com.device)
    grav = torch.cross(mc - msum * ks.anchor, up.expand_as(mc), dim=-1)  # (sum m (c - p)) x g z
    tau = (ks.axis * grav).sum(-1)
    leg_of_joint = torch.tensor([0] * 5 + [1] * 5, device=tau.device)
    foot = ks.site_pos[:, leg_of_joint]  # (B, 10, 3)
    f = forces[:, leg_of_joint]
    lever = torch.cross(foot - ks.anchor, f, dim=-1)
    if moments is not None:
      lever = lever + moments[:, leg_of_joint]
    return tau - (ks.axis * lever).sum(-1)
