"""CUDA load test without Isaac Sim, to tell a GPU/driver problem from an Isaac one.

  burst:  many small batched kernels with a sync per step, like the MPC walker
          + PPO in training (launch-bound, the GPU load goes up and down fast).
  steady: back-to-back large matmuls, like FurMark-style full load.

  python scripts/cuda_burst_test.py --mode burst --minutes 10 --batch 16384

Prints a progress line every 30 s and exits 0 if it ran the full time.
A driver reset shows up as "CUDA error: unknown error" and exit code 1.
"""

import argparse
import sys
import time

import torch


def burst_step(x, A, v):
  # Shapes and op mix similar to the walker: batched small matmuls,
  # small linear solves, elementwise math and reductions.
  for _ in range(8):
    J = A + 0.01 * torch.tanh(x[:, :, None] * x[:, None, :])
    H = J.transpose(1, 2) @ J + 1e-3 * torch.eye(J.shape[-1], device=x.device)
    dx, _ = torch.linalg.solve_ex(H, (J.transpose(1, 2) @ v[..., None]))
    x = x + 0.01 * dx.squeeze(-1)
    x = torch.where(torch.isfinite(x), x, torch.zeros_like(x)).clamp(-3, 3)
  r = (x.square().sum(-1) + torch.sin(x).mean(-1)).mean()
  return x, r


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--mode", choices=["burst", "steady"], default="burst")
  ap.add_argument("--minutes", type=float, default=10.0)
  ap.add_argument("--batch", type=int, default=16384)
  args = ap.parse_args()

  dev = torch.device("cuda")
  print(f"[cuda_test] {torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
        f"mode {args.mode}, {args.minutes} min", flush=True)
  torch.manual_seed(0)
  if args.mode == "burst":
    n = 16
    x = torch.randn(args.batch, n, device=dev)
    A = torch.randn(args.batch, n, n, device=dev) * 0.3
    v = torch.randn(args.batch, n, device=dev)
  else:
    a = torch.randn(8192, 8192, device=dev, dtype=torch.float16)
    b = torch.randn(8192, 8192, device=dev, dtype=torch.float16)

  t0 = last = time.time()
  steps = 0
  try:
    while time.time() - t0 < args.minutes * 60:
      if args.mode == "burst":
        x, r = burst_step(x, A, v)
        r.item()  # sync every step, like reading back results each control tick
      else:
        for _ in range(20):
          c = a @ b
        c[0, 0].item()
      steps += 1
      if time.time() - last >= 30:
        last = time.time()
        print(f"[cuda_test] {last - t0:6.0f} s  {steps} steps  ok", flush=True)
  except RuntimeError as e:
    print(f"[cuda_test] FAILED after {time.time() - t0:.0f} s: {e}", flush=True)
    sys.exit(1)
  print(f"[cuda_test] PASSED {args.minutes} min, {steps} steps", flush=True)


if __name__ == "__main__":
  main()
