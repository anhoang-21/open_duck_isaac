"""Isaac Lab env: Open Duck walking with the GPU MPC walker plus an RL residual.

Every control step (50 Hz) and for every environment:

1. Sensing like the real robot: IMU orientation and gyro (with noise), joint
   angles; the CoM comes from leg odometry (``walker.LegOdometry``).
2. MPC walker (``walker.BatchedZmpWalker``): ZMP-MPC + capture point + IK +
   servo load compensation -> joint targets ``q_mpc``.
3. Policy residual: ``q_target = q_mpc + action_scale * action``.
4. The targets reach the STS3215 position servos (PhysX implicit PD with the
   MJCF servo parameters) after a random delay of 0..``max_delay`` physics
   steps, like the servo bus.

With a zero action the robot is driven by the MPC alone; the policy learns
the corrections the model misses (servo compliance, foot edge contact,
rotor inertia, delays).
"""

from __future__ import annotations

import math

import gymnasium as gym  # noqa: F401
import mujoco
import torch
from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat

from open_duck_isaac.constants import (
  JOINT_NAMES, MJCF_PATH, SERVO_ARMATURE, SERVO_DAMPING, SERVO_EFFORT, SERVO_FRICTION, SERVO_KP, START_POSE,
  USD_DIR,
)
from open_duck_isaac.kinematics import RobotKinematics
from open_duck_isaac.walker import BatchedZmpWalker, LegOdometry, WalkParams

ROBOT_USD = str(USD_DIR / "open_duck_import" / "open_duck_import.usda")
START_JOINTS = dict(zip(JOINT_NAMES, (float(x) for x in START_POSE)))
OBS_DIM = 54
PRIV_DIM = 8


@configclass
class EventCfg:
  physics_material = EventTerm(
    func=mdp.randomize_rigid_body_material,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
      "static_friction_range": (0.5, 1.2),
      "dynamic_friction_range": (0.4, 1.0),
      "restitution_range": (0.0, 0.0),
      "num_buckets": 64,
    },
  )
  base_mass = EventTerm(
    func=mdp.randomize_rigid_body_mass,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
      "mass_distribution_params": (-0.08, 0.15),
      "operation": "add",
    },
  )
  servo_gains = EventTerm(
    func=mdp.randomize_actuator_gains,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      "stiffness_distribution_params": (0.85, 1.15),
      "damping_distribution_params": (0.8, 1.2),
      "operation": "scale",
    },
  )
  joint_params = EventTerm(
    func=mdp.randomize_joint_parameters,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      "armature_distribution_params": (0.8, 1.3),
      "friction_distribution_params": (0.5, 1.5),
      "operation": "scale",
    },
  )
  push = EventTerm(
    func=mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(5.0, 9.0),
    params={"velocity_range": {"x": (-0.15, 0.15), "y": (-0.1, 0.1)}},
  )


@configclass
class OpenDuckSceneCfg(InteractiveSceneCfg):
  terrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="plane",
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
      friction_combine_mode="multiply", restitution_combine_mode="multiply",
      static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
    ),
    debug_vis=False,
  )
  robot: ArticulationCfg = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
      usd_path=ROBOT_USD,
      rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False, linear_damping=0.0, angular_damping=0.0,
        max_depenetration_velocity=1.0, enable_gyroscopic_forces=True,
      ),
      articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1,
        sleep_threshold=0.0, stabilization_threshold=0.0,
      ),
      collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25), joint_pos=START_JOINTS),
    actuators={
      # STS3215: tau = kp (target - q) - damping qdot, |tau| <= 3.23 N m
      "servos": ImplicitActuatorCfg(
        joint_names_expr=[".*"], stiffness=SERVO_KP, damping=SERVO_DAMPING,
        effort_limit_sim=SERVO_EFFORT, velocity_limit_sim=10.0,
        armature=SERVO_ARMATURE, friction=SERVO_FRICTION,
      ),
    },
  )
  light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))


@configclass
class OpenDuckEnvCfg(DirectRLEnvCfg):
  episode_length_s = 20.0
  decimation = 4  # 200 Hz physics, 50 Hz control (= walker dt)
  action_space = 10
  observation_space = OBS_DIM
  state_space = 0
  action_scale = 0.1  # rad of residual per unit action
  max_delay_substeps = 4  # servo bus delay, 0..20 ms
  gyro_noise = 0.05  # rad/s
  joint_noise = 0.005  # rad
  # commands (m/s), heading frame of the start pose
  vx_range = (-0.05, 0.15)
  vy_range = (-0.05, 0.05)
  zero_cmd_prob = 0.1
  # reward weights
  w_track_vel = 2.0
  vel_sigma = 0.05  # m/s
  w_yaw_rate = 0.5
  w_alive = 0.5
  w_flat = -5.0
  w_ang_vel_xy = -0.05
  w_lin_vel_z = -2.0
  w_residual = -0.1
  w_action_rate = -0.05
  w_torque = -1e-3
  w_termination = -20.0
  # termination
  min_base_height = 0.12
  max_tilt = 0.7

  sim: SimulationCfg = SimulationCfg(
    dt=1 / 200,
    render_interval=decimation,
    physics=PhysxCfg(gpu_max_rigid_patch_count=2**21),
    physics_material=sim_utils.RigidBodyMaterialCfg(
      friction_combine_mode="multiply", restitution_combine_mode="multiply",
      static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
    ),
  )
  scene: OpenDuckSceneCfg = OpenDuckSceneCfg(num_envs=4096, env_spacing=1.0, replicate_physics=True)
  events: EventCfg = EventCfg()
  viewer: ViewerCfg = ViewerCfg(eye=(0.8, 0.8, 0.4), lookat=(0.0, 0.0, 0.15), origin_type="env", env_index=0)


class OpenDuckEnv(DirectRLEnv):
  cfg: OpenDuckEnvCfg

  def __init__(self, cfg: OpenDuckEnvCfg, render_mode: str | None = None, **kwargs):
    super().__init__(cfg, render_mode, **kwargs)
    n, dev = self.num_envs, self.device
    self.robot = self.scene["robot"]
    self.jid = torch.as_tensor(self.robot.find_joints(list(JOINT_NAMES), preserve_order=True)[0], device=dev)
    self.base_id = self.robot.find_bodies("base_link")[0][0]

    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    self.kin = RobotKinematics(model, dev)
    self.wp = WalkParams(dt=self.step_dt)
    self.walker = BatchedZmpWalker(self.kin, n, self.wp)
    self.odo = LegOdometry(self.kin, n, self.step_dt)
    self.q0 = torch.as_tensor(START_POSE, dtype=torch.float32, device=dev)
    # Base height that puts the soles on the floor in the start pose.
    ks = self.kin.fk(torch.zeros(1, 3, device=dev), torch.eye(3, device=dev)[None], self.q0[None])
    self.base_z0 = float(-ks.site_pos[0, :, 2].min()) + 0.003

    self.actions = torch.zeros(n, 10, device=dev)
    self.prev_actions = torch.zeros(n, 10, device=dev)
    self.cmd = torch.zeros(n, 2, device=dev)
    self.q_mpc = self.q0.repeat(n, 1)
    self.target_old = self.q0.repeat(n, 1)
    self.target_new = self.q0.repeat(n, 1)
    self.delay = torch.zeros(n, dtype=torch.long, device=dev)
    self.substep = 0
    self.full_targets = torch.zeros(n, self.robot.num_joints, device=dev)
    self.qmin, self.qmax = self.kin.qmin, self.kin.qmax
    keys = ["track_vel", "yaw_rate", "alive", "flat", "ang_vel_xy", "lin_vel_z", "residual", "action_rate", "torque",
            "termination"]
    self.episode_sums = {k: torch.zeros(n, device=dev) for k in keys}
    self.vel_err_sum = torch.zeros(n, device=dev)
    self.steps = torch.zeros(n, device=dev)

  # ---------------------------------------------------------------- sensing
  def _joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos.torch[:, self.jid]

  def _base_rot(self) -> torch.Tensor:
    return matrix_from_quat(self.robot.data.root_quat_w.torch)

  # ---------------------------------------------------------------- control
  def _pre_physics_step(self, actions: torch.Tensor):
    self.prev_actions = self.actions
    self.actions = actions.clamp(-5.0, 5.0)
    q = self._joint_pos() + self.cfg.joint_noise * torch.randn_like(self.actions)
    com, vcom = self.odo.update(self._base_rot(), q, self.walker.contact)
    self.q_mpc = self.walker.step(self.cmd, com, vcom, self.odo.foot_world.clone())
    target = (self.q_mpc + self.cfg.action_scale * self.actions).clamp(self.qmin, self.qmax)
    self.target_old = self.target_new
    self.target_new = target
    self.delay = torch.randint(0, self.cfg.max_delay_substeps + 1, (self.num_envs,), device=self.device)
    self.substep = 0

  def _apply_action(self):
    use_new = (self.substep >= self.delay)[:, None]
    tgt = torch.where(use_new, self.target_new, self.target_old)
    self.full_targets[:, self.jid] = tgt
    self.robot.set_joint_position_target_index(target=self.full_targets)
    self.substep += 1

  # ------------------------------------------------------------ observations
  def _get_observations(self) -> dict:
    d = self.robot.data
    gyro = d.root_ang_vel_b.torch + self.cfg.gyro_noise * torch.randn(self.num_envs, 3, device=self.device)
    q = self._joint_pos()
    phase = 2.0 * math.pi * self.walker.phase
    obs = torch.cat([
      d.projected_gravity_b.torch,
      gyro * 0.25,
      q - self.q0,
      d.joint_vel.torch[:, self.jid] * 0.05,
      self.actions,
      self.cmd * 5.0,
      torch.sin(phase)[:, None], torch.cos(phase)[:, None],
      self.q_mpc - self.q0,
      self.walker.contact.float(),
      self.walker.dcm_err * 10.0,
    ], 1)
    base_h = (d.root_pos_w.torch[:, 2] - self.scene.env_origins[:, 2])[:, None]
    priv = torch.cat([d.root_lin_vel_b.torch * 2.0, d.root_ang_vel_b.torch * 0.25, (base_h - self.base_z0) * 10.0,
                      (self.odo.com[:, 2:3] - self.walker.com_height[:, None]) * 10.0], 1)
    return {"policy": obs, "critic": torch.cat([obs, priv], 1)}

  # ----------------------------------------------------------------- rewards
  def _get_rewards(self) -> torch.Tensor:
    c = self.cfg
    d = self.robot.data
    vel = d.root_lin_vel_w.torch[:, :2]  # heading frame of the start pose = world
    err = torch.sum(torch.square(vel - self.cmd), 1)
    terms = {
      "track_vel": c.w_track_vel * torch.exp(-err / c.vel_sigma**2),
      "yaw_rate": c.w_yaw_rate * torch.exp(-torch.square(d.root_ang_vel_w.torch[:, 2]) / 0.25),
      "alive": torch.full_like(err, c.w_alive),
      "flat": c.w_flat * torch.sum(torch.square(d.projected_gravity_b.torch[:, :2]), 1),
      "ang_vel_xy": c.w_ang_vel_xy * torch.sum(torch.square(d.root_ang_vel_b.torch[:, :2]), 1),
      "lin_vel_z": c.w_lin_vel_z * torch.square(d.root_lin_vel_w.torch[:, 2]),
      "residual": c.w_residual * torch.sum(torch.square(self.actions * c.action_scale / 0.1), 1),
      "action_rate": c.w_action_rate * torch.sum(torch.square(self.actions - self.prev_actions), 1),
      "torque": c.w_torque * torch.sum(torch.square(self.robot.actuators.applied_effort.torch), 1),
      "termination": c.w_termination * self.reset_terminated.float(),
    }
    self.vel_err_sum += torch.sqrt(err)
    self.steps += 1
    reward = torch.zeros_like(err)
    for k, v in terms.items():
      v = v * self.step_dt
      self.episode_sums[k] += v
      reward += v
    return reward

  def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
    time_out = self.episode_length_buf >= self.max_episode_length - 1
    d = self.robot.data
    height = d.root_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]
    g = d.projected_gravity_b.torch
    tilt = torch.acos((-g[:, 2]).clamp(-1.0, 1.0))
    died = (height < self.cfg.min_base_height) | (tilt > self.cfg.max_tilt) | ~torch.isfinite(height)
    died |= self.walker.bad | ~torch.isfinite(d.joint_pos.torch).all(1)
    return died, time_out

  # ------------------------------------------------------------------- reset
  def _reset_idx(self, env_ids: torch.Tensor | None):
    if env_ids is None or len(env_ids) == self.num_envs:
      env_ids = torch.arange(self.num_envs, device=self.device)
    env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
    if len(env_ids) > 0:
      steps = self.steps[env_ids].clamp_min(1.0)
      log = {f"Episode_Reward/{k}": (v[env_ids].mean() / self.max_episode_length_s).item()
             for k, v in self.episode_sums.items()}
      log["Metrics/vel_error"] = (self.vel_err_sum[env_ids] / steps).mean().item()
      log["Metrics/episode_s"] = (self.steps[env_ids] * self.step_dt).mean().item()
      log["Episode_Termination/fall"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
      self.extras["log"] = log
    self.robot.reset(env_ids)
    super()._reset_idx(env_ids)
    if len(env_ids) == self.num_envs:
      self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
    for v in self.episode_sums.values():
      v[env_ids] = 0.0
    self.vel_err_sum[env_ids] = 0.0
    self.steps[env_ids] = 0.0
    k = len(env_ids)
    dev = self.device
    cmd = torch.empty(k, 2, device=dev)
    cmd[:, 0].uniform_(*self.cfg.vx_range)
    cmd[:, 1].uniform_(*self.cfg.vy_range)
    cmd[torch.rand(k, device=dev) < self.cfg.zero_cmd_prob] = 0.0
    self.cmd[env_ids] = cmd

    pose = torch.zeros(k, 7, device=dev)
    pose[:, :3] = self.scene.env_origins[env_ids]
    pose[:, 2] += self.base_z0
    pose[:, 6] = 1.0  # quaternion (x, y, z, w)
    self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(k, 6, device=dev), env_ids=env_ids)
    q_full = self.robot.data.default_joint_pos.torch[env_ids].clone()
    q_full[:, self.jid] = self.q0
    self.robot.write_joint_position_to_sim_index(position=q_full, env_ids=env_ids)
    self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q_full), env_ids=env_ids)
    self.full_targets[env_ids] = q_full
    self.target_old[env_ids] = self.q0
    self.target_new[env_ids] = self.q0
    self.q_mpc[env_ids] = self.q0
    self.actions[env_ids] = 0.0
    self.prev_actions[env_ids] = 0.0

    # Walker and odometry start from the reset pose (env-local frame).
    base_pos = torch.zeros(k, 3, device=dev)
    base_pos[:, 2] = self.base_z0
    rot = torch.eye(3, device=dev).repeat(k, 1, 1)
    self.walker.reset(env_ids, base_pos, rot, self.q0.repeat(k, 1))
    self.odo.reset(env_ids, self.walker.feet0[env_ids].clone())
