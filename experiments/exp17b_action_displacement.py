"""
Experiment 17b: Per-State Action-Displacement Distribution
==========================================================
For each MuJoCo env (HalfCheetah, Ant, Hopper), measure the per-state
deterministic-action displacement

    Delta a(s, g) = || pi_g(s) - pi_{9.81}(s) ||_2

across target gravities g in {5, 7, 9.81, 12, 15}, using the same SAC
checkpoints as exp10. Distribution of Delta a(s, g) is the continuous-action
analog of the action-gap CDF F_kappa: states with small Delta a stay within
their original basin (quadratic regime); states with large Delta a have
switched basins (super-quadratic).

Prediction: if basin-switching is the mechanism for HalfCheetah/Ant exceeding
the quadratic slope at large |Delta g|, those envs should show a heavy-tailed
displacement distribution (most states small, a few very large), while Hopper
stays tight.

Decision rule: if Ant (and ideally HalfCheetah) shows clearly heavier tail
than Hopper, ship as basin-switching evidence. Otherwise discard.

Usage:
  python exp17b_action_displacement.py sanity  # 50 states, source + 1 target
  python exp17b_action_displacement.py run     # full run, save pkl
  python exp17b_action_displacement.py plot    # generate figure from cached pkl
"""

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
import os
import sys
import pickle
import time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'font.size': 13, 'font.family': 'serif',
    'axes.labelsize': 14, 'axes.titlesize': 14,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 10, 'figure.dpi': 150,
})
import warnings
warnings.filterwarnings('ignore')

CACHE_DIR = '../cache_exp10'
OUT_PKL = os.path.join(CACHE_DIR, 'results_action_displacement.pkl')
SEED = 42
GRAVITIES = [5.0, 7.0, 9.81, 12.0, 15.0]
SRC_GRAVITY = 9.81
N_STATES = 300

ENV_CONFIGS = {
    'HalfCheetah-v4': {'label': 'HalfCheetah', 'color': '#1f77b4', 'act_dim': 6},
    'Ant-v4':         {'label': 'Ant',         'color': '#ff7f0e', 'act_dim': 8},
    'Hopper-v4':      {'label': 'Hopper',      'color': '#2ca02c', 'act_dim': 3},
}


def make_env(env_id, gravity=9.81):
    env = gym.make(env_id)
    env.unwrapped.model.opt.gravity[:] = [0, 0, -gravity]
    return env


def load_model(env_id, gravity, seed=SEED):
    path = os.path.join(CACHE_DIR, f'sac_{env_id}_g{gravity:.2f}_s{seed}_t500000.zip')
    env = make_env(env_id, gravity)
    model = SAC.load(path, env=env, device='cuda')
    return model, env


def collect_states(env_id, gravity, n_states, model):
    env = make_env(env_id, gravity)
    states = []
    s, _ = env.reset(seed=SEED + 999)
    for t in range(20000):
        a, _ = model.predict(s, deterministic=True)
        s, r, term, trunc, _ = env.step(a)
        if t >= 200 and t % 5 == 0:
            states.append(s.copy())
        if term or trunc:
            s, _ = env.reset(seed=SEED + 999 + t)
        if len(states) >= n_states:
            break
    env.close()
    return np.array(states[:n_states])


def get_actions(model, states):
    actions = np.empty((len(states), states.shape[-1] if False else 0), dtype=np.float32) if False else None
    out = []
    for s in states:
        a, _ = model.predict(s, deterministic=True)
        out.append(a)
    return np.array(out)


def measure_env(env_id, gravities, n_states, sanity=False):
    """Returns dict with 'states', 'actions_by_g' (g -> (N, act_dim)),
    'displacements_by_g' (g -> (N,)) measured against SRC_GRAVITY."""
    print(f"\n[{env_id}] loading source model and collecting {n_states} states ...")
    src_model, _ = load_model(env_id, SRC_GRAVITY)
    states = collect_states(env_id, SRC_GRAVITY, n_states, src_model)
    print(f"  collected {len(states)} states")

    src_actions = get_actions(src_model, states)
    actions_by_g = {SRC_GRAVITY: src_actions}
    displacements_by_g = {SRC_GRAVITY: np.zeros(len(states), dtype=np.float32)}

    targets = gravities if not sanity else [g for g in gravities if g != SRC_GRAVITY][:1]
    for g in targets:
        if g == SRC_GRAVITY:
            continue
        print(f"  loading model g={g:.2f} and computing per-state actions ...")
        model, env = load_model(env_id, g)
        acts = get_actions(model, states)
        env.close()
        del model
        actions_by_g[g] = acts
        diffs = np.linalg.norm(acts - src_actions, axis=1).astype(np.float32)
        displacements_by_g[g] = diffs
        print(f"    g={g:.2f}: median={np.median(diffs):.4f}, "
              f"mean={diffs.mean():.4f}, p90={np.percentile(diffs, 90):.4f}, "
              f"max={diffs.max():.4f}")

    return {
        'env_id': env_id,
        'gravities': sorted(actions_by_g.keys()),
        'states': states,
        'actions_by_g': actions_by_g,
        'displacements_by_g': displacements_by_g,
    }


def run_all(sanity=False):
    n_states = 50 if sanity else N_STATES
    results = {}
    t0 = time.time()
    for env_id in ENV_CONFIGS:
        results[env_id] = measure_env(env_id, GRAVITIES, n_states, sanity=sanity)
    print(f"\n[done] elapsed {time.time()-t0:.1f}s")

    if sanity:
        # print but do not save
        return results

    with open(OUT_PKL, 'wb') as f:
        pickle.dump(results, f)
    print(f"[saved] {OUT_PKL}")
    return results


def summarize(results):
    """Print summary table; useful in both sanity and run modes."""
    print("\n=== Per-state action displacement summary ===")
    header = f"{'env':<14}{'g':>7}  {'|dg|':>6}  {'med':>8}{'mean':>8}{'p90':>8}{'p99':>8}{'max':>8}"
    print(header)
    print('-' * len(header))
    for env_id, r in results.items():
        for g in r['gravities']:
            d = r['displacements_by_g'][g]
            print(f"{env_id:<14}{g:>7.2f}  {abs(g-SRC_GRAVITY):>6.2f}  "
                  f"{np.median(d):>8.4f}{d.mean():>8.4f}{np.percentile(d,90):>8.4f}"
                  f"{np.percentile(d,99):>8.4f}{d.max():>8.4f}")


def plot(results, out_path=None):
    """3 panels:
      (a) CDF of Delta a at the most extreme target gravity (g=5) per env.
      (b) Median Delta a vs |Delta g| per env.
      (c) p90 / p99 Delta a vs |Delta g| per env (tail).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) CDF at extreme target
    extreme = 5.0
    ax = axes[0]
    for env_id, cfg in ENV_CONFIGS.items():
        if env_id not in results:
            continue
        d = results[env_id]['displacements_by_g'].get(extreme)
        if d is None:
            continue
        xs = np.sort(d)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, color=cfg['color'], lw=2.0, label=cfg['label'])
    ax.set_xlabel(r'$\|\pi_g(s) - \pi_{9.81}(s)\|_2$')
    ax.set_ylabel('empirical CDF')
    ax.set_title(f'(a) Displacement CDF at g={extreme}')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (b) median vs |Delta g|
    ax = axes[1]
    for env_id, cfg in ENV_CONFIGS.items():
        if env_id not in results:
            continue
        gs, meds = [], []
        for g in results[env_id]['gravities']:
            gs.append(abs(g - SRC_GRAVITY))
            meds.append(np.median(results[env_id]['displacements_by_g'][g]))
        order = np.argsort(gs)
        gs = np.array(gs)[order]; meds = np.array(meds)[order]
        ax.plot(gs, meds, 'o-', color=cfg['color'], lw=2, label=cfg['label'])
    ax.set_xlabel(r'$|\Delta g|$')
    ax.set_ylabel(r'median $\|\Delta a\|_2$')
    ax.set_title('(b) Median displacement')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (c) p90, p99 tails
    ax = axes[2]
    for env_id, cfg in ENV_CONFIGS.items():
        if env_id not in results:
            continue
        gs, p90s, p99s = [], [], []
        for g in results[env_id]['gravities']:
            gs.append(abs(g - SRC_GRAVITY))
            d = results[env_id]['displacements_by_g'][g]
            p90s.append(np.percentile(d, 90))
            p99s.append(np.percentile(d, 99))
        order = np.argsort(gs)
        gs = np.array(gs)[order]
        p90s = np.array(p90s)[order]; p99s = np.array(p99s)[order]
        ax.plot(gs, p90s, 'o-', color=cfg['color'], lw=2, label=f"{cfg['label']} p90")
        ax.plot(gs, p99s, 's--', color=cfg['color'], lw=1.5, alpha=0.7,
                label=f"{cfg['label']} p99")
    ax.set_xlabel(r'$|\Delta g|$')
    ax.set_ylabel(r'$\|\Delta a\|_2$')
    ax.set_title('(c) Tail (p90, p99)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    if out_path is None:
        out_path = os.path.join(CACHE_DIR, 'fig_action_displacement.pdf')
    plt.savefig(out_path)
    plt.close()
    print(f"[saved] {out_path}")


def discard_run_artifacts():
    """Walk-back option: remove pkl and figure if results don't support claim."""
    for p in [OUT_PKL, os.path.join(CACHE_DIR, 'fig_action_displacement.pdf')]:
        if os.path.exists(p):
            os.remove(p)
            print(f"[removed] {p}")


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'sanity'
    if mode == 'sanity':
        results = run_all(sanity=True)
        summarize(results)
        print("\n[sanity] no pkl/figure saved.")
    elif mode == 'run':
        results = run_all(sanity=False)
        summarize(results)
        plot(results)
    elif mode == 'plot':
        with open(OUT_PKL, 'rb') as f:
            results = pickle.load(f)
        summarize(results)
        plot(results)
    elif mode == 'discard':
        discard_run_artifacts()
    else:
        print(f"unknown mode: {mode}")
        sys.exit(1)
