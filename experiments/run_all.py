"""
Master script: Run all experiments and generate all figures.
Usage: python run_all.py
"""

import subprocess
import sys
import os
import time

EXPERIMENTS = [
    ('exp1_transfer_gap.py', 'Exp 1: Transfer Gap Decomposition'),
    ('exp2_stability_radius.py', 'Exp 2: Stability Radius'),
    ('exp3_lqr_quadratic.py', 'Exp 3: LQR Quadratic Degradation'),
    ('exp4_gymnasium.py', 'Exp 4: Control Environments'),
    ('exp5_sample_complexity.py', 'Exp 5: Sample Complexity'),
    ('exp6_baselines.py', 'Exp 6: Baseline Comparison'),
    ('exp7_mujoco.py', 'Exp 7: MuJoCo Surrogate'),
    ('exp8_dqn_transfer.py', 'Exp 8: DQN Model-Agnostic Transfer'),
    ('exp9_gymnasium_benchmark.py', 'Exp 9: Gymnasium Benchmark'),
    ('exp10_mujoco_transfer.py plot', 'Exp 10: MuJoCo Transfer (plot only)'),
]

def run_experiment(script, name, timeout=900):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable] + script.split(),
            timeout=timeout,
            capture_output=True,
            text=True
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            print(f"  DONE ({elapsed:.1f}s)")
            # Print last few lines
            lines = result.stdout.strip().split('\n')
            for line in lines[-5:]:
                print(f"  {line}")
        else:
            print(f"  FAILED ({elapsed:.1f}s)")
            print(f"  {result.stderr[-500:]}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT (>{timeout}s)")
        return False


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    results = []
    for script, name in EXPERIMENTS:
        ok = run_experiment(script, name)
        results.append((name, ok))

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for name, ok in results:
        status = 'PASS' if ok else 'FAIL'
        print(f"  [{status}] {name}")

    # Check figures
    fig_dir = '../figures'
    if os.path.exists(fig_dir):
        pdfs = [f for f in os.listdir(fig_dir) if f.endswith('.pdf')]
        print(f"\n  Generated {len(pdfs)} PDF figures in {fig_dir}/")
        for f in sorted(pdfs):
            print(f"    {f}")
