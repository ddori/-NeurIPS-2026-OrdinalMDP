"""
Multi-seed training for HalfCheetah (exp10).
Trains seed=123 and seed=456 for all 5 gravities + DR.
Seed=42 is already cached.
"""
import sys
sys.path.insert(0, '.')
from exp10_mujoco_transfer import train_agent, train_dr_agent, CACHE_DIR
import time

ENV_ID = 'HalfCheetah-v4'
GRAVITIES = [5.0, 7.0, 9.81, 12.0, 15.0]
N_TIMESTEPS = 500_000
NEW_SEEDS = [123, 456]

print("=" * 60)
print("  Multi-seed training: HalfCheetah-v4")
print(f"  Seeds: {NEW_SEEDS}, Gravities: {GRAVITIES}")
print("=" * 60)

total_t0 = time.time()
count = 0
total = len(NEW_SEEDS) * (len(GRAVITIES) + 1)  # +1 for DR

for seed in NEW_SEEDS:
    for g in GRAVITIES:
        count += 1
        print(f"\n[{count}/{total}] seed={seed}, g={g}")
        model, env = train_agent(ENV_ID, g, N_TIMESTEPS, seed=seed)
        env.close()
        del model

    # DR baseline
    count += 1
    print(f"\n[{count}/{total}] seed={seed}, DR")
    dr_model = train_dr_agent(ENV_ID, (5.0, 15.0), N_TIMESTEPS, seed=seed)
    del dr_model

elapsed = time.time() - total_t0
print(f"\n{'=' * 60}")
print(f"  All done in {elapsed/3600:.1f} hours")
print(f"{'=' * 60}")
