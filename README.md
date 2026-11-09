# Open Duck Mini: MPC + RL residual in Isaac Lab

A model-based walker (MPC) for the Open Duck Mini biped, running batched on the GPU,
plus a PPO policy that learns a small joint-angle correction on top of it.
Trained in Isaac Sim 6.2 / Isaac Lab 3.0 with 16384 parallel robots.

```
q_target = q_mpc + 0.1 * a        (a = policy output, 10 joints, rad)
```

## How it works

**MPC walker** (`open_duck_isaac/walker.py`, `kinematics.py`), every control step for every env:

1. Footstep planner from the velocity command (vx, vy).
2. Capture-point / DCM step adjustment in x and y to recover from pushes
   (`footstep="raibert"` uses the Raibert velocity heuristic instead).
3. ZMP preview MPC, horizon 80 steps (1.6 s), for the centre-of-mass trajectory. The ZMP
   reference sits 7 mm outward of the foot site: in the start pose the soles are rolled 10°
   and the robot stands on their outer edge. The measured lateral CoM velocity is fed back
   into the MPC state (`mpc_fb_vel`).
4. Swing-foot trajectory.
5. Damped-least-squares whole-body IK to joint targets `q_mpc` (one warm-started iteration
   per tick), with load compensation.
6. CoM measurement by FK from the IMU orientation and joint angles, relative to the support
   feet (no world position estimate): position from the feet at their planned positions,
   velocity relative to the stance foot of the current step.

**RL residual** (`open_duck_isaac/env.py`, `agent_cfg.py`): rsl_rl PPO, MLP [256, 128, 128],
54 observations, 10 outputs scaled by 0.1 rad. A residual-size penalty in the reward keeps
the gait the MPC's own; the policy only fixes what the model gets wrong (servo delay,
model mismatch, domain randomization).

## Results

1024 robots, 20 s, with domain randomization (`scripts/mpc_only.py`):

| Controller | vx command (m/s) | Survived | Actual vx (m/s) | Roll RMS |
| --- | --- | --- | --- | --- |
| MPC only | 0.10 | 44.4 % | 0.061 | 6.5° |
| MPC only, current walker (outward ZMP, CP in x and y, lateral MPC feedback) | 0.10 | 65.1 % | 0.064 | 7.2° |
| MPC + RL | 0.10 | 98.6 % | 0.106 | 3.3° |
| MPC + RL | 0 | 99.8 % | 0.004 | |
| MPC + RL | 0.15 | 98.9 % | 0.121 | |

The MPC + RL rows and `logs/run1/model_1649.pt` were trained on the earlier walker
(inward ZMP offset, CP in y only, open-loop MPC); retrain for the current one.

Walker cost on a CPU, one robot, one thread: about 2 ms per 50 Hz tick on a desktop core
(`kin.fk` batched per tree depth, first-step MPC update, one IK iteration).

Training: about 590 M environment steps (1650 PPO iterations, mostly at 16384 envs),
about 1.5 h on a laptop RTX 5080. The final policy is `logs/run1/model_1649.pt`.

## Usage (Windows)

Edit the Isaac Lab python path in `run.bat` / `play.bat` first.

```bat
:: watch the policy in the Isaac Sim window (free camera: right mouse + WASDQE)
play.bat --num-envs 64 --vx 0.1
play.bat --mpc-only --vx 0.1          :: MPC alone, for comparison
play.bat --follow --no-dr             :: camera follows robot 0, no randomization

:: train (restarts from the last checkpoint after a crash)
run.bat scripts\train_supervised.py --num-envs 16384 --iterations 1650 --run-dir logs/run1

:: evaluate survival / velocity tracking
run.bat scripts\mpc_only.py --checkpoint logs/run1/model_1649.pt --vx 0.1 --num-envs 1024
```

`scripts/watch_train.ps1` prints GPU temperature/power and the rsl_rl progress while training.

## Note on GPU driver resets (TDR)

On this laptop, training at 8192+ envs kept crashing with `CUDA error: unknown error` and a
black screen (Windows TDR, `nvlddmkm` event 153). Drivers, heat, power, VRAM and the code were
ruled out with `scripts/diagnose_tdr.ps1`. The fix was turning off Windows
"Hardware-accelerated GPU scheduling" (Settings > System > Display > Graphics) and rebooting.

## Credits

Robot model and meshes from [Open Duck Mini](https://github.com/apirrone/Open_Duck_Mini).
