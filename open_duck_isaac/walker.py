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
3. Footstep correction: the landing of the swing foot moves by the DCM error
   (capture point) or the velocity error (Raibert), measured minus planned,
   in x and y. Optionally the measured CoM state is fed back into the MPC.
   The CoM is measured relative to the support feet by FK from the IMU
   orientation and the joint angles (assumed at their planned positions):
   balance only needs the CoM relative to the feet, so there is no world
   position estimate (leg odometry) to drift.
4. Swing trajectories, then whole-body IK (CoM, feet, base orientation,
   posture) to joint angles.
5. Load compensation for position servos: ``q += tau / kp`` with ``tau`` the
   torque the planned motion needs (gravity of the legs and the planned
   ground force and center-of-pressure moment on the stance feet).

The RL policy adds a residual to these joint targets.
"""

from __future__ import annotations

import dataclasses
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
  # Lateral ZMP reference from the foot site, + = outward. In CROUCH10 the soles are
  # rolled 10 deg and the robot stands on their outer edge (+-9.5 mm from the site), so the
  # center of pressure can only sit near that edge in single support.
  zmp_offset_out: float = 0.007
  zmp_box_back: float = 0.022
  zmp_box_front: float = 0.045
  zmp_box_half_width: float = 0.003  # around zmp_offset_out (the support strip along the edge)
  start_shift: float = 0.4
  # Landing correction of the step in progress, per axis (x, y):
  #   "cp":      gain * DCM error (position + velocity / omega, measured - planned)
  #   "raibert": (half period + raibert_gain) * velocity error (measured - planned)
  footstep: str = "cp"
  step_gain: tuple[float, float] = (0.4, 0.3)
  raibert_gain: tuple[float, float] = (0.0, 0.0)
  step_limit: tuple[float, float] = (0.04, 0.03)
  # State feedback into the ZMP-MPC: the initial state moves this fraction of the way from
  # the plan to the estimate (CoM position, velocity) every tick, per axis (x, y);
  # 0 = open-loop plan.
  mpc_fb_pos: tuple[float, float] = (0.0, 0.0)
  mpc_fb_vel: tuple[float, float] = (0.0, 0.2)  # x: the measured velocity is biased (the stance foot rolls), keep it open
  min_step_width: float = 0.12
  reanchor_x: bool = True
  vel_filter: float = 0.5  # low-pass of the measured CoM velocity (1 = raw finite difference)
  load: float = 1.0  # load compensation, 0..1
  cop_moment: bool = True  # include the planned CoP moment in the load term
  servo_kp: float = 13.37
  ik: IkWeights = field(default_factory=IkWeights)
  ik_iterations_warm: int = 1  # IK iterations per tick after the first (warm start)
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
    self.kz0, self.kv0 = self.kz[0], self.kv[0]
    self.a, self.b = f(a), f(b)
    self.c = f(c)

  def solve(self, x0: torch.Tensor, zref: torch.Tensor, vref: torch.Tensor) -> torch.Tensor:
    """x0 (B, 3), zref (B, N), vref (B, N) -> predicted states (B, N, 3)."""
    zf = x0 @ self.z_x.T
    vf = x0 @ self.v_x.T
    u = (zref - zf) @ self.kz.T + (vref - vf) @ self.kv.T
    return torch.einsum("kij,bj->bki", self.phi, x0) + torch.einsum("kin,bn->bki", self.gam, u)

  def step(self, x0: torch.Tensor, zref: torch.Tensor, vref: torch.Tensor) -> torch.Tensor:
    """Receding horizon: only the first jerk of the plan is applied, so only the first
    rows of the gains are needed. x0 (..., 3), zref/vref (..., N) -> next state (..., 3)."""
    u0 = ((zref - x0 @ self.z_x.T) * self.kz0).sum(-1) + ((vref - x0 @ self.v_x.T) * self.kv0).sum(-1)
    return x0 @ self.a.T + u0[..., None] * self.b


class BatchedZmpWalker:
  N_FOOTSTEPS = 14  # footstep positions P_{-1} .. P_12 around the current step

  def __init__(self, kin: RobotKinematics, num_envs: int, params: WalkParams | None = None):
    self.kin = kin
    self.p = params or WalkParams()
    self.n = num_envs
    dev = self.dev = kin.device
    z3 = lambda *s: torch.zeros(num_envs, *s, device=dev)
    self.t = z3()
    self.tick = torch.zeros(num_envs, dtype=torch.long, device=dev)  # t = tick * dt (no float drift)
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
    # Measurement (CoM relative to the feet by FK): last tick's values for the velocity.
    self.rel_prev = z3(2, 3)
    self.meas_contact = torch.ones(num_envs, 2, dtype=torch.bool, device=dev)
    self.meas_fresh = torch.ones(num_envs, dtype=torch.bool, device=dev)
    self.meas_good = torch.ones(num_envs, dtype=torch.bool, device=dev)
    self.com_meas = z3(3)  # measured CoM in the plan frame
    self.vcom_meas = z3(3)  # measured CoM velocity (relative to the stance feet)
    self.preview: ZmpPreview | None = None
    self.ik_ks: KinState | None = None  # FK of the IK warm start (ik_base_pos, ik_base_rot, ik_q)
    self._alternate = torch.arange(self.N_FOOTSTEPS - 3, device=dev) % 2
    self.omega = 1.0
    # Debug / observation outputs.
    self.out_q = z3(10)
    self.out_com = z3(3)
    self.out_feet = z3(2, 3)
    self.out_zmp = z3(2)
    self.dcm_err = z3(2)
    self.state_err = z3(4)  # CoM position and velocity error (x, y), estimate - plan
    self.phase = z3()  # gait phase in [0, 1)
    self.bad = torch.zeros(num_envs, dtype=torch.bool, device=dev)  # non-finite input or IK this tick

  # ------------------------------------------------------------------ reset
  def reset(self, env_ids: torch.Tensor, base_pos: torch.Tensor, base_rot: torch.Tensor, q: torch.Tensor) -> None:
    """Start the walker of ``env_ids`` from a measured standing pose (feet on the floor)."""
    p = self.p
    ks = self.kin.fk(base_pos, base_rot, q)
    self.t[env_ids] = 0.0
    self.tick[env_ids] = 0
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
    if self.ik_ks is not None:  # keep the cached warm-start FK in step with the reset pose
      for f in dataclasses.fields(ks):
        getattr(self.ik_ks, f.name)[env_ids] = getattr(ks, f.name)
    self.contact[env_ids] = True
    self.meas_contact[env_ids] = True
    self.meas_fresh[env_ids] = True
    self.vcom_meas[env_ids] = 0.0
    self.com_meas[env_ids] = ks.com
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
    eps = 1e-3 * self.p.dt  # t sits on the tick grid: round, don't truncate, at step boundaries
    j = torch.floor((tau + eps) / half).long()
    local = (tau - j * half).clamp_min(0.0)
    pre = tau < -eps
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
    # Steps jb+1, jb+2, ...: the swing side alternates starting with st, so the
    # offsets alternate between two values (the command is constant over the horizon).
    off = torch.stack([self._step_offset(st), self._step_offset(sw)], 1)  # (B, 2, 3)
    later = p1[:, None] + torch.cumsum(off[:, self._alternate], 1)
    return torch.cat([pm1[:, None], p0[:, None], p1[:, None], later], 1), jb

  # --------------------------------------------------------------- update
  def _zmp_point(self, foot: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
    outward = torch.where(side == 0, 1.0, -1.0)  # left foot: +y
    return foot[..., :2] + torch.stack(
      [torch.full_like(outward, self.p.zmp_offset_x), outward * self.p.zmp_offset_out], -1
    )

  def _measure(self, base_rot: torch.Tensor, q: torch.Tensor, j_now: torch.Tensor,
               vcom: torch.Tensor | None) -> torch.Tensor:
    """CoM relative to each foot site, (B, 2, 3), by FK from the IMU orientation and the
    joint angles; also updates ``vcom_meas`` (low-pass filtered), or sets it to ``vcom``.

    The velocity is the one of the CoM relative to the stance foot of the current step,
    if it was on the ground at the last tick too (else the feet that were). In double
    support the feet can slide against each other, and the foot that stays is the
    reference for the rest of the step; averaging both feet there loses that slide
    (under randomized friction this cost survival and roll in Isaac)."""
    ks = self.kin.fk(torch.zeros(self.n, 3, device=self.dev), base_rot, q)
    rel = ks.com[:, None, :] - ks.site_pos
    stay = self.contact & self.meas_contact
    b = torch.arange(self.n, device=self.dev)
    st = self._stance(j_now.clamp_min(0))
    only_st = torch.zeros_like(stay)
    only_st[b, st] = True
    use_st = (j_now >= 0) & stay[b, st]
    stay = torch.where(use_st[:, None], only_st, stay).float()
    v = ((rel - self.rel_prev) * stay[..., None]).sum(1) / stay.sum(1, keepdim=True).clamp_min(1.0) / self.p.dt
    good = torch.isfinite(rel).flatten(1).all(1)
    if vcom is not None:
      good &= torch.isfinite(vcom).all(1)
      v_new = vcom
    else:
      v = torch.where((good & ~self.meas_fresh)[:, None], v, torch.zeros_like(v))
      v_new = (1.0 - self.p.vel_filter) * self.vcom_meas + self.p.vel_filter * v
    self.vcom_meas = torch.where(good[:, None], v_new, torch.zeros_like(v_new))
    self.rel_prev = rel
    self.meas_contact = self.contact.clone()
    self.meas_fresh = ~good  # restart the velocity after a non-finite measurement
    self.meas_good = good
    return rel

  def step(
    self,
    cmd: torch.Tensor,
    base_rot: torch.Tensor,
    q: torch.Tensor,
    vcom: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Advance all walkers by one tick; return joint targets (B, 10).

    ``cmd`` (B, 2): vx, vy. ``base_rot`` (B, 3, 3): base orientation (IMU), ``q``
    (B, 10): measured joint angles. The CoM is measured relative to the support feet
    (FK only, no world position estimate); its velocity relative to the stance foot,
    unless ``vcom`` (B, 3) is given (e.g. the true CoM velocity in simulation).
    """
    p = self.p
    dev = self.dev
    b = torch.arange(self.n, device=dev)
    half = 0.5 * p.period
    ds = p.double_support * p.period
    self.cmd = cmd
    t = self.t
    j_now, _ = self._phase(t)
    rel = self._measure(base_rot, q, j_now, vcom)
    good = self.meas_good  # non-finite measurement (a simulation blow-up): fall back to the plan

    # 1. New step: commit the landing, re-anchor x to the measured stance foot.
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
        # The new stance foot just landed: measured from the previous stance foot (still on
        # the ground) by FK. Step 0 starts from the reset pose, no landing to measure.
        landed = fs[b, sw, 0] + rel[b, sw, 0] - rel[b, st, 0]
        shift = torch.where(later & good, landed - self.feet_start[b, st, 0], torch.zeros_like(t))
        self.feet_start[:, :, 0] += shift[:, None]
        self.landing[:, 0] += shift
        self.nominal_landing[:, 0] += shift
        self.com_state[:, 0, 0] += shift
    self.last_step = torch.where(j_now >= 0, j_now, torch.full_like(j_now, -1))

    # 2. Measured CoM in the plan frame: the support feet are where the plan put them.
    support = self.contact.float()
    planned = torch.where((j_now >= 0)[:, None, None], self.feet_start, self.feet0)
    com_meas = ((planned + rel) * support[..., None]).sum(1) / support.sum(1, keepdim=True)
    plan_com = torch.cat([self.com_state[:, :, 0], self.com_height[:, None]], 1)
    self.com_meas = torch.where(good[:, None], com_meas, plan_com)
    vcom_meas = self.vcom_meas

    # 3. DCM and CoM state error, measured - planned.
    dcm_plan = self.com_state[:, :, 0] + self.com_state[:, :, 1] / self.omega
    self.dcm_err = self.com_meas[:, :2] + vcom_meas[:, :2] / self.omega - dcm_plan
    dcm_err = self.dcm_err
    pos_err = self.com_meas[:, :2] - self.com_state[:, :, 0]
    vel_err = vcom_meas[:, :2] - self.com_state[:, :, 1]
    self.state_err = torch.cat([pos_err, vel_err], 1)

    # 4. Footstep correction of the landing of the step in progress.
    j_next, _ = self._phase(t + p.dt)
    adjust = (j_next >= 0) & (j_next == j_now)
    lim = torch.tensor(p.step_limit, device=dev)
    if p.footstep == "raibert":
      corr = (half + torch.tensor(p.raibert_gain, device=dev)) * vel_err
    else:
      corr = torch.tensor(p.step_gain, device=dev) * dcm_err
    land = self.nominal_landing.clone()
    land[:, :2] += torch.clamp(corr, -lim, lim)
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

    if any(p.mpc_fb_pos) or any(p.mpc_fb_vel):  # close the loop on the MPC state
      self.com_state[:, :, 0] += torch.tensor(p.mpc_fb_pos, device=dev) * pos_err
      self.com_state[:, :, 1] += torch.tensor(p.mpc_fb_vel, device=dev) * vel_err
    vref = self.cmd[:, None, :].expand(-1, n, -1)
    # Both axes at once: x0 (B, 2, 3), references (B, 2, N).
    self.com_state = self.preview.step(self.com_state, zref.transpose(1, 2), vref.transpose(1, 2))

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
      ks0=None if first else self.ik_ks,
    )
    self.ik_iterations_needed = False
    ok = torch.isfinite(q).all(1) & torch.isfinite(base_pos).all(1) & torch.isfinite(base_rot).flatten(1).all(1)
    self.bad = ~ok | ~good
    base_pos = torch.where(ok[:, None], base_pos, self.ik_base_pos)
    base_rot = torch.where(ok[:, None, None], base_rot, self.ik_base_rot)
    q = torch.where(ok[:, None], q, self.q_home)
    # FK of the warm start for the next tick (non-finite for a failed env: its next IK
    # step is skipped, and the FK after it is finite again).
    self.ik_base_pos, self.ik_base_rot, self.ik_q, self.ik_ks = base_pos, base_rot, q, ks

    # 8. Load compensation.
    acc = self.com_state[:, :, 2]
    zmp = self.com_state[:, :, 0] - acc / self.omega**2
    if p.load > 0.0:
      tau = self._load_torques(ks, contact, feet, acc, zmp)
      q = q + p.load * tau / p.servo_kp

    self.tick = self.tick + 1
    self.t = self.tick.float() * p.dt
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
      out = torch.tensor([p.zmp_offset_out, -p.zmp_offset_out], device=cop.device)  # left, right
      cop[:, :, 1] = torch.where(both[:, None], out.expand_as(cop[:, :, 1]), cop[:, :, 1])
      cop[:, :, 0] = cop[:, :, 0].clamp(-p.zmp_box_back, p.zmp_box_front)
      cop[:, :, 1] = (cop[:, :, 1] - out).clamp(-p.zmp_box_half_width, p.zmp_box_half_width) + out
      r = torch.cat([cop, torch.zeros_like(cop[..., :1])], -1)
      moments = torch.cross(r, forces, dim=-1) * contact[..., None]
    return self.kin.leg_torques(ks, forces * contact[..., None], moments)
