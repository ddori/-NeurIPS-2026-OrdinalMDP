"""
Wait for Hopper/HalfCheetah to finish, then run Ant, Walker2d, Swimmer sequentially.
Usage: python run_remaining_mujoco.py
"""
import os
import time
import subprocess
import sys

CACHE_DIR = '../cache_exp10'
REQUIRED = [
    ('Hopper-v4', 'hopper', [5.0, 7.0, 9.81, 12.0, 15.0]),
    ('HalfCheetah-v4', 'halfcheetah', [5.0, 7.0, 9.81, 12.0, 15.0]),
]
TO_RUN = ['ant', 'walker2d', 'swimmer']


def check_done(env_id, gravities):
    """Check if all gravity models + DR are cached."""
    for g in gravities:
        path = os.path.join(CACHE_DIR, f'sac_{env_id}_g{g:.2f}_s42_t500000.zip')
        if not os.path.exists(path):
            return False
    # Check DR
    dr_path = os.path.join(CACHE_DIR, f'sac_{env_id}_DR_{min(gravities)}-{max(gravities)}_s42_t500000.zip')
    if not os.path.exists(dr_path):
        return False
    return True


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Wait for Hopper + HalfCheetah
    print("Waiting for Hopper and HalfCheetah to finish...")
    while True:
        all_done = all(check_done(eid, gs) for eid, _, gs in REQUIRED)
        if all_done:
            break
        # Print status
        for eid, name, gs in REQUIRED:
            done_gs = sum(1 for g in gs if os.path.exists(
                os.path.join(CACHE_DIR, f'sac_{eid}_g{g:.2f}_s42_t500000.zip')))
            dr_done = os.path.exists(
                os.path.join(CACHE_DIR, f'sac_{eid}_DR_{min(gs)}-{max(gs)}_s42_t500000.zip'))
            print(f"  {name}: {done_gs}/{len(gs)} gravities, DR={'yes' if dr_done else 'no'}")
        print(f"  Checking again in 5 min...")
        time.sleep(300)

    print("\n=== Hopper + HalfCheetah done! Starting remaining environments ===\n")

    # Run remaining sequentially
    for env_name in TO_RUN:
        print(f"\n{'='*60}")
        print(f"  Starting {env_name}")
        print(f"{'='*60}")
        result = subprocess.run(
            [sys.executable, '-u', 'exp10_mujoco_transfer.py', env_name],
            timeout=None
        )
        if result.returncode != 0:
            print(f"  WARNING: {env_name} failed with code {result.returncode}")

    # Final plot
    print("\n=== All done! Generating final figure ===")
    subprocess.run([sys.executable, 'exp10_mujoco_transfer.py', 'plot'])
