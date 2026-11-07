"""Batched model-based walker (ZMP-MPC + capture point + whole-body IK) in torch.

GPU port of ``mjlab_cd2/src/open_duck_mjlab/mpc/zmp_walker.py`` for training
thousands of robots in parallel, and for running one robot on its own
computer. Per control tick (50 Hz), for every environment:

1. Gait clock and footsteps: double support, right swing, double support,
   left swing; the landing of the step in progress is committed when the step
   starts, later footsteps follow the current velocity command.
2. ZMP-MPC per horizontal axis (triple-integrator CoM, jerk input, ZMP
   tracking over an 80-step horizon). The QP is solved without the ZMP box
   constraints, which makes it a fixed linear map (precomputed once): the
   reference keeps the ZMP well inside the soles, so the box is rarely active.
3. Capture-point stepping: the landing of the swing foot moves by the DCM
   error (measured minus planned), relative to the support feet.
4. Swing trajectories, then whole-body IK (CoM, feet, base orientation,
   posture) to joint angles.
5. Load compensation for position servos: ``q += tau / kp`` with ``tau`` the
   torque the planned motion needs (gravity of the legs and the planned
   ground force and center-of-pressure moment on the stance feet).

The RL policy adds a residual to these joint targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from open_duck_isaac.kinematics import GRAVITY, IkWeights, KinState, RobotKinematics, remove_yaw


@dataclass
class WalkParams:
  dt: float = 0.02
  period: float = 0.4
  double_support: float = 0.1
  swing_height: float = 0.02
  zmp_offset_x: float = 0.005
  zmp_offset_in: float = 0.004
  zmp_box_back: float = 0.022
  zmp_box_front: float = 0.045
  zmp_box_half_width: float = 0.009
  start_shift: float = 0.4
  step_gain: tuple[float, float] = (0.0, 0.3)
  step_limit: tuple[float, float] = (0.04, 0.03)
  min_step_width: float = 0.12
  reanchor_x: bool = True
  load: float = 1.0  # load compensation, 0..1
  cop_moment: bool = True  # include the planned CoP moment in the load term
  servo_kp: float = 13.37
  ik: IkWeights = field(default_factory=IkWeights)
  ik_iterations_warm: int = 3  # IK iterations per tick after the first
  # ZMP MPC
  horizon: int = 80
  q_zmp: float = 1.0
  q_vel: float = 1e-3
  r_jerk: float = 1e-6


def _smoothstep(s: torch.Tensor) -> torch.Tensor:
  s = s.clamp(0.0, 1.0)
  return 3.0 * s * s - 2.0 * s**3


class ZmpPreview:
  """Unconstrained condensed ZMP MPC as a linear map (shared by both axes)."""

  def __init__(self, p: WalkParams, com_height: float, device: torch.device):
    n, dt, h = p.horizon, p.dt, com_height
    a = np.array([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
    b = np.array([dt**3 / 6.0, 0.5 * dt * dt, dt])
    c = np.array([1.0, 0.0, -h / GRAVITY])
    phi = np.zeros((n, 3, 3))
    gam = np.zeros((n, 3, n))
    ak = np.eye(3)
    powers = [np.eye(3)]
    for _ in range(n):
      powers.append(a @ powers[-1])
    for k in range(n):
      ak = a @ ak
      phi[k] = ak
      for j in range(k + 1):
        gam[k, :, j] = powers[k - j] @ b
    z_x = np.einsum("i,kij->kj", c, phi)
    z_u = np.einsum("i,kij->kj", c, gam)
    v_x = phi[:, 1, :]
    v_u = gam[:, 1, :]
    hess = p.q_zmp * z_u.T @ z_u + p.q_vel * v_u.T @ v_u + p.r_jerk * np.eye(n)
    hinv = np.linalg.inv(hess)
    f = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    # u = Kz (zref - Zx x0) + Kv (vref - Vx x0)
    self.kz = f(hinv @ (p.q_zmp * z_u.T))
    self.kv = f(hinv @ (p.q_vel * v_u.T))
    self.z_x, self.v_x = f(z_x), f(v_x)
    self.phi, self.gam = f(phi), f(gam)
    self.c = f(c)

  def solve(self, x0: torch.Tensor, zref: torch.Tensor, vref: torch.Tensor) -> torch.Tensor:
    """x0 (B, 3), zref (B, N), vref (B, N) -> predicted states (B, N, 3)."""
    zf = x0 @ self.z_x.T
    vf = x0 @ self.v_x.T
    u = (zref - zf) @ self.kz.T + (vref - vf) @ self.kv.T
    return torch.einsum("kij,bj->bki", self.phi, x0) + torch.einsum("kin,bn->bki", self.gam, u)


class BatchedZmpWalker:
  N_FOOTSTEPS = 14  # footstep positions P_{-1} .. P_12 around the current step

  def __init__(self, kin: RobotKinematics, num_envs: int, params: WalkParams | None = None):
    self.kin = kin
    self.p = params or WalkParams()
    self.n = num_envs
    dev = self.dev = kin.device
    z3 = lambda *s: torch.zeros(num_envs, *s, device=dev)
    self.t = z3()
    self.com_state = z3(2, 3)
    self.feet0 = z3(2, 3)
    self.foot_rot0 = torch.eye(3, device=dev).repeat(num_envs, 2, 1, 1)
    self.feet_start = z3(2, 3)  # both feet at the start of the current step
    self.landing = z3(3)
    self.nominal_landing = z3(3)
    self.last_step = torch.full((num_envs,), -3, dtype=torch.long, device=dev)
    self.q_home = z3(10)
    self.ik_base_pos = z3(3)
    self.ik_base_rot = torch.eye(3, device=dev).repeat(num_envs, 1, 1)
    self.ik_q = z3(10)
    self.com_height = z3()
    self.cmd = z3(2)  # vx, vy (heading frame of the start pose)
    self.contact = torch.ones(num_envs, 2, dtype=torch.bool, device=dev)
    self.preview: ZmpPreview | None = None
    self.omega = 1.0
    # Debug / observation outputs.
    self.out_q = z3(10)
    self.out_com = z3(3)
    self.out_feet = z3(2, 3)
    self.out_zmp = z3(2)
    self.dcm_err = z3(2)
    self.phase = z3()  # gait phase in [0, 1)
    self.bad = torch.zeros(num_envs, dtype=torch.bool, device=dev)  # non-finite input or IK this tick

  # ------------------------------------------------------------------ reset
  def reset(self, env_ids: torch.Tensor, base_pos: torch.Tensor, base_rot: torch.Tensor, q: torch.Tensor) -> None:
    """Start the walker of ``env_ids`` from a measured standing pose (feet on the floor)."""
    p = self.p
    ks = self.kin.fk(base_pos, base_rot, q)
    self.t[env_ids] = 0.0
    self.feet0[env_ids] = ks.site_pos
    self.foot_rot0[env_ids] = remove_yaw(ks.site_rot)
    self.feet_start[env_ids] = ks.site_pos
    self.landing[env_ids] = ks.site_pos[:, 1]
    self.nominal_landing[env_ids] = ks.site_pos[:, 1]
    self.com_state[env_ids] = 0.0
    self.com_state[env_ids, :, 0] = ks.com[:, :2]
    self.com_height[env_ids] = ks.com[:, 2]
    self.last_step[env_ids] = -3
    self.q_home[env_ids] = q
    self.ik_base_pos[env_ids] = base_pos
    self.ik_base_rot[env_ids] = base_rot
    self.ik_q[env_ids] = q
    self.contact[env_ids] = True
    self.dcm_err[env_ids] = 0.0
    if self.preview is None:
      h = float((ks.com[:, 2] - ks.site_pos[:, :, 2].mean(1)).mean())
      self.preview = ZmpPreview(p, h, self.dev)
      self.omega = float(np.sqrt(GRAVITY / h))

  # -------------------------------------------------------------- gait clock
  def _phase(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(step index, time into the step); -1 during the initial weight shift."""
    half = 0.5 * self.p.period
    tau = t - self.p.start_shift
    j = torch.floor(tau / half).long()
    local = tau - j * half
    pre = tau < 0
    j = torch.where(pre, torch.full_like(j, -1), j)
    local = torch.where(pre, t, local)
    return j, local

  @staticmethod
  def _stance(j: torch.Tensor) -> torch.Tensor:
    """Right foot swings on even steps (left stance = 0), left on odd."""
    return torch.remainder(j, 2)

  def _step_offset(self, swing: torch.Tensor) -> torch.Tensor:
    """Landing of foot ``swing`` relative to the stance foot, (B, 3)."""
    p = self.p
    half = 0.5 * p.period
    b = torch.arange(self.n, device=self.dev)
    y_sw = self.feet0[b, swing, 1]
    y_st = self.feet0[b, 1 - swing, 1]
    off = torch.zeros(self.n, 3, device=self.dev)
    off[:, 0] = self.cmd[:, 0] * half
    off[:, 1] = (y_sw - y_st) + self.cmd[:, 1] * half
    off[:, 2] = self.feet0[b, swing, 2] - self.feet0[b, 1 - swing, 2]
    return off

  def _footsteps(self, j_now: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Footstep positions P_i (B, N, 3) for i = -1 .. N-2 and the base step jb.

    The stance foot of step ``jb + i`` is at ``P_i``; ``P_{-1}`` is where the
    swing foot of step ``jb`` starts.
    """
    b = torch.arange(self.n, device=self.dev)
    jb = j_now.clamp_min(0)
    st = self._stance(jb)
    sw = 1 - st
    walking = j_now >= 0
    p0 = self.feet_start[b, st]
    pm1 = self.feet_start[b, sw]
    p1 = torch.where(walking[:, None], self.landing, p0 + self._step_offset(sw))
    pts = [pm1, p0, p1]
    # Steps jb+1, jb+2, ...: the swing side alternates starting with st.
    side = st.clone()
    for _ in range(self.N_FOOTSTEPS - 3):
      pts.append(pts[-1] + self._step_offset(side))
      side = 1 - side
    return torch.stack(pts, 1), jb

  # --------------------------------------------------------------- update
  def _zmp_point(self, foot: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
    inward = torch.where(side == 0, -1.0, 1.0)
    return foot[..., :2] + torch.stack(
      [torch.full_like(inward, self.p.zmp_offset_x), inward * self.p.zmp_offset_in], -1
    )

  def step(
    self,
    cmd: torch.Tensor,
    com_meas: torch.Tensor,
    vcom_meas: torch.Tensor,
    feet_meas: torch.Tensor,
  ) -> torch.Tensor:
    """Advance all walkers by one tick; return joint targets (B, 10).

    ``cmd`` (B, 2): vx, vy. ``com_meas``/``vcom_meas`` (B, 3) and ``feet_meas``
    (B, 2, 3): estimated CoM, its velocity and the estimated foot positions.
    """
    p = self.p
    dev = self.dev
    b = torch.arange(self.n, device=dev)
    half = 0.5 * p.period
    ds = p.double_support * p.period
    self.cmd = cmd
    t = self.t
    j_now, _ = self._phase(t)
    # Non-finite estimates (a simulation blow-up): fall back to the plan.
    good = torch.isfinite(com_meas).all(1) & torch.isfinite(vcom_meas).all(1) & torch.isfinite(feet_meas).flatten(1).all(1)
    plan_com = torch.cat([self.com_state[:, :, 0], self.com_height[:, None]], 1)
    com_meas = torch.where(good[:, None], com_meas, plan_com)
    vcom_meas = torch.where(good[:, None], vcom_meas, torch.zeros_like(vcom_meas))
    feet_meas = torch.where(good[:, None, None], feet_meas, self.feet_start)

    # 1. DCM error against the plan (before it advances).
    dcm_meas = com_meas[:, :2] + vcom_meas[:, :2] / self.omega
    dcm_plan = self.com_state[:, :, 0] + self.com_state[:, :, 1] / self.omega
    dcm_err = dcm_meas - dcm_plan

    # 2. New step: commit the landing, re-anchor x to the measured stance foot.
    new = (j_now >= 0) & (j_now != self.last_step)
    if new.any():
      st = self._stance(j_now)
      sw = 1 - st
      first = new & (j_now == 0)
      later = new & (j_now > 0)
      # Step j > 0: the previous swing foot (now stance) landed at `landing`.
      fs = self.feet_start.clone()
      fs[b, st] = torch.where(later[:, None], self.landing, fs[b, st])
      fs = torch.where(first[:, None, None], self.feet0, fs)
      land = fs[b, st] + self._step_offset(sw)
      self.feet_start = torch.where(new[:, None, None], fs, self.feet_start)
      self.landing = torch.where(new[:, None], land, self.landing)
      self.nominal_landing = torch.where(new[:, None], land, self.nominal_landing)
      if p.reanchor_x:
        shift = torch.where(new, feet_meas[b, st, 0] - self.feet_start[b, st, 0], torch.zeros_like(t))
        self.feet_start[:, :, 0] += shift[:, None]
        self.landing[:, 0] += shift
        self.nominal_landing[:, 0] += shift
        self.com_state[:, 0, 0] += shift
        dcm_err[:, 0] -= shift
    self.last_step = torch.where(j_now >= 0, j_now, torch.full_like(j_now, -1))

    # 3. Error relative to the support feet (estimate vs plan).
    support = self.contact.float()
    planned = torch.where((j_now >= 0)[:, None, None], self.feet_start, self.feet0)
    off = ((feet_meas[:, :, :2] - planned[:, :, :2]) * support[..., None]).sum(1) / support.sum(1, keepdim=True)
    dcm_err = dcm_err - off
    self.dcm_err = dcm_err

    # 4. Capture-point stepping on the landing of the step in progress.
    j_next, _ = self._phase(t + p.dt)
    adjust = (j_next >= 0) & (j_next == j_now)
    gain = torch.tensor(p.step_gain, device=dev)
    lim = torch.tensor(p.step_limit, device=dev)
    land = self.nominal_landing.clone()
    land[:, :2] += torch.clamp(gain * dcm_err, -lim, lim)
    st = self._stance(j_now.clamp_min(0))
    stance_y = self.feet_start[b, st, 1]
    land[:, 1] = torch.where(
      st == 0, torch.minimum(land[:, 1], stance_y - p.min_step_width),
      torch.maximum(land[:, 1], stance_y + p.min_step_width),
    )
    self.landing = torch.where(adjust[:, None], land, self.landing)

    # 5. ZMP reference over the horizon.
    pts, jb = self._footsteps(j_now)
    n = p.horizon
    tk = t[:, None] + p.dt * torch.arange(1, n + 1, device=dev)[None]
    jk, lk = self._phase(tk)  # (B, N)
    zref = self._zmp_reference(pts, jb, jk, lk)

    vref = self.cmd[:, None, :].expand(-1, n, -1)
    states = torch.stack([
      self.preview.solve(self.com_state[:, ax], zref[..., ax], vref[..., ax]) for ax in range(2)
    ], 1)  # (B, 2, N, 3)
    self.com_state = states[:, :, 0]
    self.plan_states = states

    # 6. Feet at t + dt and the swing trajectory.
    j1, l1 = self._phase(t + p.dt)
    feet, contact = self._feet_at(pts, jb, j1, l1)
    self.contact = contact

    # 7. IK.
    com = torch.cat([self.com_state[:, :, 0], self.com_height[:, None]], 1)
    first = self.ik_iterations_needed
    base_pos, base_rot, q, ks = self.kin.ik(
      self.ik_base_pos, self.ik_base_rot, self.ik_q, com, feet, self.foot_rot0, self.q_home,
      p.ik, iterations=p.ik.iterations if first else p.ik_iterations_warm,
    )
    self.ik_iterations_needed = False
    ok = torch.isfinite(q).all(1) & torch.isfinite(base_pos).all(1) & torch.isfinite(base_rot).flatten(1).all(1)
    self.bad = ~ok | ~good
    base_pos = torch.where(ok[:, None], base_pos, self.ik_base_pos)
    base_rot = torch.where(ok[:, None, None], base_rot, self.ik_base_rot)
    q = torch.where(ok[:, None], q, self.q_home)
    self.ik_base_pos, self.ik_base_rot, self.ik_q = base_pos, base_rot, q

    # 8. Load compensation.
    acc = self.com_state[:, :, 2]
    zmp = self.com_state[:, :, 0] - acc / self.omega**2
    if p.load > 0.0:
      tau = self._load_torques(ks, contact, feet, acc, zmp)
      q = q + p.load * tau / p.servo_kp

    self.t = t + p.dt
    self.out_q, self.out_com, self.out_feet, self.out_zmp = q, com, feet, zmp
    self.phase = torch.remainder((self.t - p.start_shift) / p.period, 1.0)
    return q

  ik_iterations_needed: bool = True

  def _zmp_reference(self, pts, jb, jk, lk) -> torch.Tensor:
    p = self.p
    ds = p.double_support * p.period
    b = torch.arange(self.n, device=self.dev)[:, None]
    i = (jk - jb[:, None]).clamp(0, self.N_FOOTSTEPS - 3)  # index into P_0.. (pts offset by 1)
    side = self._stance(jk.clamp_min(0))
    st = self._zmp_point(pts[b, i + 1], side)
    old = self._zmp_point(pts[b, i], 1 - side)
    s = torch.where(jk > 0, _smoothstep(lk / ds), torch.ones_like(lk))
    zref = torch.where((lk < ds)[..., None], (1 - s[..., None]) * old + s[..., None] * st, st)
    # Initial weight shift onto the first stance foot.
    f0 = self.feet0
    pre0 = self._zmp_point(f0[:, 0], torch.zeros(self.n, dtype=torch.long, device=self.dev))
    pre1 = self._zmp_point(f0[:, 1], torch.ones(self.n, dtype=torch.long, device=self.dev))
    mid = 0.5 * (pre0 + pre1)
    s0 = _smoothstep(lk / p.start_shift)[..., None]
    pre = (1 - s0) * mid[:, None] + s0 * pre0[:, None]  # step 0: left stance
    return torch.where((jk < 0)[..., None], pre, zref)

  def _feet_at(self, pts, jb, j, local) -> tuple[torch.Tensor, torch.Tensor]:
    p = self.p
    half = 0.5 * p.period
    ds = p.double_support * p.period
    b = torch.arange(self.n, device=self.dev)
    i = (j - jb).clamp(0, self.N_FOOTSTEPS - 3)
    st = self._stance(j.clamp_min(0))
    sw = 1 - st
    stance_pos = pts[b, i + 1]
    start = pts[b, i]
    end = pts[b, i + 2]
    s = ((local - ds) / (half - ds)).clamp(0.0, 1.0)
    swinging = (j >= 0) & (local >= ds)
    sw_pos = start + (end - start) * _smoothstep(s)[:, None]
    sw_pos[:, 2] = start[:, 2] + p.swing_height * 64.0 * s**3 * (1.0 - s) ** 3
    sw_pos = torch.where(swinging[:, None], sw_pos, start)
    feet = torch.zeros(self.n, 2, 3, device=self.dev)
    feet[b, st] = stance_pos
    feet[b, sw] = sw_pos
    feet = torch.where((j < 0)[:, None, None], self.feet0, feet)
    contact = torch.ones(self.n, 2, dtype=torch.bool, device=self.dev)
    contact[b, sw] = ~swinging
    return feet, contact

  def _load_torques(self, ks: KinState, contact, feet, acc, zmp) -> torch.Tensor:
    p = self.p
    m = self.kin.total_mass
    force = torch.stack([m * acc[:, 0], m * acc[:, 1], torch.full_like(acc[:, 0], m * GRAVITY)], 1)
    both = contact[:, 0] & contact[:, 1]
    span = feet[:, 0, 1] - feet[:, 1, 1]
    lam = ((zmp[:, 1] - feet[:, 1, 1]) / torch.where(span.abs() > 1e-6, span, torch.ones_like(span))).clamp(0, 1)
    share = torch.where(both[:, None], torch.stack([lam, 1 - lam], 1), contact.float())
    forces = share[..., None] * force[:, None, :]
    moments = None
    if p.cop_moment:
      cop = zmp[:, None, :] - feet[:, :, :2]
      cop[:, :, 1] = torch.where(both[:, None], torch.zeros_like(cop[:, :, 1]), cop[:, :, 1])
      cop[:, :, 0] = cop[:, :, 0].clamp(-p.zmp_box_back, p.zmp_box_front)
      cop[:, :, 1] = cop[:, :, 1].clamp(-p.zmp_box_half_width, p.zmp_box_half_width)
      r = torch.cat([cop, torch.zeros_like(cop[..., :1])], -1)
      moments = torch.cross(r, forces, dim=-1) * contact[..., None]
    return self.kin.leg_torques(ks, forces * contact[..., None], moments)


class LegOdometry:
  """Batched CoM estimate from the IMU orientation and joint angles."""

  def __init__(self, kin: RobotKinematics, num_envs: int, dt: float, vel_filter: float = 0.5):
    self.kin = kin
    self.dt = dt
    self.alpha = vel_filter
    dev = kin.device
    self.foot_world = torch.zeros(num_envs, 2, 3, device=dev)
    self.foot_z0 = torch.zeros(num_envs, 2, device=dev)
    self.contact = torch.ones(num_envs, 2, dtype=torch.bool, device=dev)
    self.com = torch.zeros(num_envs, 3, device=dev)
    self.vcom = torch.zeros(num_envs, 3, device=dev)
    self.fresh = torch.ones(num_envs, dtype=torch.bool, device=dev)

  def reset(self, env_ids: torch.Tensor, feet_world: torch.Tensor) -> None:
    self.foot_world[env_ids] = feet_world
    self.foot_z0[env_ids] = feet_world[:, :, 2]
    self.contact[env_ids] = True
    self.vcom[env_ids] = 0.0
    self.fresh[env_ids] = True

  def update(self, base_rot: torch.Tensor, q: torch.Tensor, contact: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    zero = torch.zeros(q.shape[0], 3, device=q.device)
    ks = self.kin.fk(zero, base_rot, q)
    anchor = self.contact & contact
    none = ~anchor.any(1)
    anchor = torch.where(none[:, None], self.contact, anchor)
    anchor = torch.where(~anchor.any(1)[:, None], torch.ones_like(anchor), anchor)
    w = anchor.float()
    base = ((self.foot_world - ks.site_pos) * w[..., None]).sum(1) / w.sum(1, keepdim=True)
    touchdown = contact & ~self.contact
    landed = base[:, None, :] + ks.site_pos
    landed[:, :, 2] = self.foot_z0
    self.foot_world = torch.where(touchdown[..., None], landed, self.foot_world)
    self.contact = contact.clone()
    com = base + ks.com
    v = (com - self.com) / self.dt
    v = torch.where(self.fresh[:, None], torch.zeros_like(v), v)
    self.vcom = (1 - self.alpha) * self.vcom + self.alpha * v
    self.com = com
    self.fresh[:] = False
    return com, self.vcom
