"""Convert the Open Duck MJCF to USD for Isaac Lab (Isaac Sim 6 MJCF importer).

1. Writes ``assets/mjcf/open_duck_import.xml``: the robot with explicit body
   inertias (MuJoCo computes them from the meshes at compile time; written
   out so the importer cannot recompute them differently), the foot
   colliders only, no actuators or sensors (the servos are set up as Isaac
   Lab implicit actuators in the env).
2. Converts it to ``assets/usd/open_duck/open_duck.usda``.
3. Prints bodies, masses and joints of the USD for a check against MuJoCo.

  run.bat scripts/convert_usd.py
"""

import os
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
import torch  # noqa: F401  (Windows: load torch DLLs before Kit)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402

from open_duck_isaac.constants import MJCF_PATH, USD_DIR  # noqa: E402

IMPORT_XML = MJCF_PATH.parent / "open_duck_import.xml"
FOOT_GEOMS = ("left_foot_collision", "right_foot_collision")


def build_import_mjcf() -> mujoco.MjModel:
  spec = mujoco.MjSpec.from_file(str(MJCF_PATH))
  model = spec.compile()
  for body in spec.bodies[1:]:
    b = model.body(body.name)
    body.explicitinertial = True
    body.mass = float(b.mass[0])
    body.ipos = b.ipos.copy()
    body.iquat = b.iquat.copy()
    body.inertia = b.inertia.copy()
  spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_FALSE
  for g in spec.geoms:
    if g.name in FOOT_GEOMS:
      g.contype, g.conaffinity, g.condim = 1, 1, 3
      g.friction = [0.8, 0.005, 0.0001]
    else:
      g.contype, g.conaffinity = 0, 0
  for a in list(spec.actuators):
    spec.delete(a)
  for s in list(spec.sensors):
    spec.delete(s)
  check = spec.compile()
  for b in range(1, model.nbody):
    assert abs(check.body_mass[b] - model.body_mass[b]) < 1e-9
    assert abs(check.body_inertia[b] - model.body_inertia[b]).max() < 1e-12
  IMPORT_XML.write_text(spec.to_xml(), encoding="utf-8")
  print(f"[MJCF] {IMPORT_XML}  total mass {model.body_subtreemass[1]:.3f} kg")
  return model


def main():
  model = build_import_mjcf()
  import argparse

  # DirectX 12 instead of Vulkan, like the other Isaac Lab project on this machine (fewer driver resets).
  if not any(a.startswith("--/app/vulkan") for a in sys.argv):
    sys.argv.append("--/app/vulkan=false")
  from isaaclab.app import AppLauncher

  parser = argparse.ArgumentParser()
  AppLauncher.add_app_launcher_args(parser)
  args, _ = parser.parse_known_args()
  args.headless = True
  app = AppLauncher(args).app
  from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

  out = USD_DIR / "open_duck"
  shutil.rmtree(out, ignore_errors=True)  # the importer never overwrites
  cfg = MjcfConverterCfg(
    asset_path=str(IMPORT_XML), usd_dir=str(USD_DIR), fix_base=False, force_usd_conversion=True,
    self_collision=False, run_asset_transformer=False,
  )
  usd_path = MjcfConverter(cfg).usd_path
  print(f"[USD] {usd_path}")

  from pxr import Usd, UsdPhysics

  stage = Usd.Stage.Open(usd_path)
  print("bodies (USD mass vs MuJoCo):")
  for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.MassAPI):
      name = prim.GetName()
      mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
      try:
        ref = model.body(name).mass[0]
      except KeyError:
        ref = float("nan")
      print(f"  {prim.GetPath()}: {mass} (mujoco {ref:.4f})")
    if prim.IsA(UsdPhysics.RevoluteJoint):
      j = UsdPhysics.RevoluteJoint(prim)
      print(f"  joint {prim.GetName()}: axis {j.GetAxisAttr().Get()} limits {j.GetLowerLimitAttr().Get()}..{j.GetUpperLimitAttr().Get()}")
    if prim.HasAPI(UsdPhysics.CollisionAPI):
      print(f"  collider {prim.GetPath()}")
  app.close()


if __name__ == "__main__":
  main()
