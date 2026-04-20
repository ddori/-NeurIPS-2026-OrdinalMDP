"""
Train a single seed across all gravities for one env.
Used to parallelize exp10_multiseed.py training on multi-GPU / large-VRAM setups.

Usage:
  python run_seed.py <env> <seed>
  e.g. python run_seed.py ant 42

Launch 5 in parallel on bash:
  for s in 42 123 7 2024 31; do python run_seed.py ant $s & done; wait
"""
import sys
import exp10_multiseed as base

if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(1)

env_name = sys.argv[1].lower()
seed = int(sys.argv[2])

if env_name not in base.ENV_CONFIGS:
    print(f"Unknown env '{env_name}'. Choose from {list(base.ENV_CONFIGS)}.")
    sys.exit(1)

env_id = base.ENV_CONFIGS[env_name]
print(f"[run_seed] env={env_id}, seed={seed}, gravities={base.GRAVITIES}")
for g in base.GRAVITIES:
    base.train_agent(env_id, g, seed)
print(f"[run_seed] env={env_id} seed={seed} done.")
