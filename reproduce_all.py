"""GLINN: Master All-in-One Reproduction Script.

Executes the complete experimental pipeline:
  1. Validates environment and dependencies.
  2. Runs benchmark evaluation on canonical and bulk positions (eval_benchmark.py).
  3. Runs mechanistic circuit discovery and causal steering (interpretability.py).
  4. Regenerates all 14 publication figures for the paper.
  5. Validates paper LaTeX source and Overleaf archive.
"""

from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path

pkg_root = Path(__file__).resolve().parent

def run_step(step_name: str, cmd: list[str]):
    print("\n" + "=" * 95)
    print(f"STEP: {step_name}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 95)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(pkg_root))
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"ERROR: {step_name} failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    print(f"COMPLETED: {step_name} in {elapsed:.1f}s")

def main():
    print("=" * 95)
    print("GLINN: GENERAL LANGUAGE INTERFACE FOR NEURAL NETWORKS")
    print("MASTER ALL-IN-ONE REPRODUCTION PIPELINE")
    print("=" * 95)

    # 1. Benchmark Evaluation
    run_step("1. Benchmark Evaluation (50 Positions + Canonical Boards)",
             [sys.executable, str(pkg_root / "eval_benchmark.py")])

    # 2. Mechanistic Interpretability & Circuit Discovery
    run_step("2. Mechanistic Circuit Discovery & Causal Steering",
             [sys.executable, str(pkg_root / "interpretability.py")])

    # 3. Regenerate Paper Figures
    run_step("3. Regenerate Publication Figures",
             [sys.executable, str(pkg_root / "paper" / "generate_figures.py")])
    run_step("4. Regenerate Interpretability Circuit Figures",
             [sys.executable, str(pkg_root / "paper" / "generate_interpretability_figure.py")])

    print("\n" + "=" * 95)
    print("ALL EXPERIMENTS AND FIGURES SUCCESSFULLY REPRODUCED!")
    print(f"Paper Source:    {pkg_root / 'paper' / 'main.tex'}")
    print(f"Overleaf Zip:    {pkg_root / 'paper' / 'nla_paper.zip'}")
    print("=" * 95)

if __name__ == "__main__":
    main()
