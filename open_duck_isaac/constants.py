"""Open Duck Mini V2 constants shared by the walker, the Isaac Lab env and the tools."""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MJCF_PATH = ROOT / "assets" / "mjcf" / "open_duck_mini_v4.xml"
USD_DIR = ROOT / "assets" / "usd"

JOINT_NAMES = (
  "L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle",
  "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle",
)
FOOT_SITES = ("L6_IK", "R6_IK")  # sole frame: x forward, y left, z up
FOOT_BODIES = ("L5_ankle", "R5_ankle")
BASE_BODY = "base_link"

# STS3215 servo model (MJCF class "sts3215").
SERVO_KP = 13.37  # N m / rad
SERVO_DAMPING = 0.56  # N m s / rad (joint damping)
SERVO_ARMATURE = 0.027  # kg m^2
SERVO_FRICTION = 0.068  # N m
SERVO_EFFORT = 3.23  # N m

_r = np.radians
# Start pose: hip roll 10 deg inward, hip pitch 20, knee 40, ankle 20.
CROUCH10 = {
  "L_hip_yaw": 0.0, "L_hip_roll": -_r(10), "L_hip_pitch": _r(20), "L_knee": -_r(40), "L_ankle": _r(20),
  "R_hip_yaw": 0.0, "R_hip_roll": _r(10), "R_hip_pitch": -_r(20), "R_knee": _r(40), "R_ankle": -_r(20),
}
START_POSE = np.array([CROUCH10[n] for n in JOINT_NAMES])
