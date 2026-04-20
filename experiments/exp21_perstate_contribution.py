"""
Experiment 21: Per-state gap-contribution vs local concavity margin.

This is the mechanistic counterpart to exp20: instead of cell-level statistics,
it pairs each on-policy state with two scalars,

    G(s) = V^{pi*_g}_g(s) - V^{pi_src}_g(s)        [per-state gap contribution]
    mu(s)= -lambda_max(grad^2_aa Q_g(s, a*(s)))    [strong-concavity margin]

at the same on-policy target state s. The per-state gap contribution decomposes
the cell-level transfer gap by Theorem 1 (state-wise sum). Theorem 4's
upper bound has coefficient L_aa Lambda^2 / (2 mu^2 (1-gamma)) and is therefore
informative only at states with mu(s) > 0.

If a substantial fraction of an environment's gap is concentrated at
mu(s) <= 0 states (precondition failures), that environment's cell-level slope
is allowed to exceed the quadratic upper bound by Theorem 4 alone (because the
bound never applied to those states), without contradicting the theorem.

Inputs:
  - cache_exp10/sac_<env>_g<g>_s42_t500000.zip          source policy g=9.81
  - cache_exp10/sac_<env>_g<g'>_s42_t500000.zip         target policy g'

Outputs:
  - cache_exp10/exp21_perstate.pkl
  - figures/exp21_perstate_contribution.{pdf,png}
"""

import os, sys, pickle, time
import numpy as np
import gymnasium as gym
import torch
import matplotlib
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from exp16_hessian_diagnostic import compute_hessian_stats  # reuse Hessian protocol

matplotlib.rcParams.update({
    'font.size': 12, 'font.family': 'serif',
    'axes.labelsize': 13, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'figure.dpi': 150,
})

import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(THIS_DIR)
CACHE_DIR = os.path.join(ROOT, 'cache_exp10')
OUT_PKL = os.path.join(CACHE_DIR, 'exp21_perstate.pkl')

SEED = 42
SRC_GRAVITY = 9.81
GRAVITIES_SHIFTED = [5.0, 7.0, 12.0, 15.0]
N_STATES = 60          # per (env, g); kept moderate for runtime
ROLLOUT_T = 400        # MC truncation; gamma^400 = 0.018 for gamma=0.99
GAMMA = 0.99
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ENV_CONFIGS = [
    ('HalfCheetah', 'HalfCheetah-v4', '#1f77b4'),
    ('Ant',         'Ant-v4',         '#d62728'),
    ('Hopper',      'Hopper-v4',      '#2ca02c'),
]


def make_env(env_id, gravity):
    env = gym.make(env_id)
    env.unwrapped.model.opt.gravity[:] = [0, 0, -gravity]
    return env


def load_sac(env_id, gravity, seed=SEED):
    path = os.path.join(CACHE_DIR, f'sac_{env_id}_g{gravity:.2f}_s{seed}_t500000.zip')
    env = make_env(env_id, gravity)
    model = SAC.load(path, env=env, device=DEVICE)
    return model


def collect_states_with_mjstate(env_id, gravity, model, n_states=N_STATES):
    """Roll target policy at target gravity; record obs + (qpos, qvel) snapshots
    so we can later restore the underlying simulator state for MC rollouts."""
    env = make_env(env_id, gravity)
    obs_list, qpos_list, qvel_list = [], [], []
    s, _ = env.reset(seed=SEED + 999)
    for t in range(20000):
        a, _ = model.predict(s, deterministic=True)
        s, r, term, trunc, _ = env.step(a)
        if t >= 200 and t % 5 == 0:
            obs_list.append(s.copy())
            qpos_list.append(env.unwrapped.data.qpos.copy())
            qvel_list.append(env.unwrapped.data.qvel.copy())
        if term or trunc:
            s, _ = env.reset(seed=SEED + 999 + t)
        if len(obs_list) >= n_states:
            break
    env.close()
    return (np.array(obs_list[:n_states]),
            np.array(qpos_list[:n_states]),
            np.array(qvel_list[:n_states]))


def mc_return_from(env, model, qpos, qvel, T=ROLLOUT_T, gamma=GAMMA):
    """Reset env (reuses caller's env handle), set qpos/qvel, run model T steps,
    return sum_{t=0}^{T-1} gamma^t r_t."""
    env.reset(seed=SEED + 12345)
    env.unwrapped.set_state(qpos, qvel)
    s = env.unwrapped._get_obs()
    R = 0.0; discount = 1.0
    for t in range(T):
        a, _ = model.predict(s, deterministic=True)
        s, r, term, trunc, _ = env.step(a)
        R += discount * r
        discount *= gamma
        if term or trunc:
            break
    return float(R)


def per_state_gap(env_id, gravity, src_model, tgt_model, qpos_arr, qvel_arr):
    """For each state, compute V^{pi_tgt}_g(s) and V^{pi_src}_g(s) by Monte Carlo
    in the target environment, return G(s) = V_tgt - V_src."""
    env = make_env(env_id, gravity)
    G = np.empty(len(qpos_arr))
    V_src = np.empty(len(qpos_arr))
    V_tgt = np.empty(len(qpos_arr))
    for i, (qp, qv) in enumerate(zip(qpos_arr, qvel_arr)):
        V_tgt[i] = mc_return_from(env, tgt_model, qp, qv)
        V_src[i] = mc_return_from(env, src_model, qp, qv)
        G[i] = V_tgt[i] - V_src[i]
    env.close()
    return G, V_tgt, V_src


def run():
    os.makedirs(CACHE_DIR, exist_ok=True)
    results = {}
    t_start = time.time()

    for label, env_id, color in ENV_CONFIGS:
        print(f"\n{'='*60}\n  {label} ({env_id})\n{'='*60}", flush=True)
        src_model = load_sac(env_id, SRC_GRAVITY)
        cell_results = {}
        for g in GRAVITIES_SHIFTED:
            print(f"  g={g:.2f}: loading target SAC, collecting {N_STATES} states...",
                  flush=True)
            tgt_model = load_sac(env_id, g)
            obs, qpos, qvel = collect_states_with_mjstate(env_id, g, tgt_model)

            print(f"           computing mu(s) at {len(obs)} states...", flush=True)
            mu = compute_hessian_stats(tgt_model, obs, device=DEVICE)

            print(f"           Monte-Carlo gap rollouts (T={ROLLOUT_T} per state, "
                  f"x2 policies)...", flush=True)
            t0 = time.time()
            G, V_tgt, V_src = per_state_gap(env_id, g, src_model, tgt_model, qpos, qvel)
            dt = time.time() - t0

            cell_results[g] = {
                'mu': mu, 'G': G, 'V_tgt': V_tgt, 'V_src': V_src,
                'qpos': qpos, 'qvel': qvel,
            }
            print(f"           done in {dt:.0f}s | "
                  f"mu med={np.median(mu):+.3f}, frac<0={np.mean(mu<0):.2f} | "
                  f"G mean={np.mean(G):.1f}, max={np.max(G):.1f}",
                  flush=True)

        results[label] = {'env_id': env_id, 'color': color, 'cells': cell_results}

    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed/60:.1f} min")

    with open(OUT_PKL, 'wb') as f:
        pickle.dump(results, f)
    print(f"[saved] {OUT_PKL}")


def plot():
    if not os.path.exists(OUT_PKL):
        print(f"No cache at {OUT_PKL}; run first."); return
    with open(OUT_PKL, 'rb') as f:
        results = pickle.load(f)

    # ── Aggregate per-state pairs across shifted gravities ──
    per_env = {}
    for label, d in results.items():
        mu_all, G_all = [], []
        for g, c in d['cells'].items():
            mu_all.append(c['mu']); G_all.append(c['G'])
        per_env[label] = {
            'mu': np.concatenate(mu_all),
            'G':  np.concatenate(G_all),
            'color': d['color'],
        }

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    # (a) per-state scatter: mu(s) vs G(s)
    ax = axes[0]
    for label, e in per_env.items():
        ax.scatter(e['mu'], e['G'], s=22, c=e['color'], alpha=0.55,
                   edgecolor='none', label=label)
    ax.axvline(0, color='red', ls='--', lw=1.4, alpha=0.7,
               label=r'Precondition limit ($\mu=0$)')
    ax.axhline(0, color='k', lw=0.6, alpha=0.4)
    ax.set_xlabel(r'Strong-concavity margin $\mu(s)$')
    ax.set_ylabel(r'Per-state gap contribution $G(s)$')
    ax.set_title('(a) Per-state gap concentrates at small $\\mu$ on Ant')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) bar: fraction of total positive-gap mass coming from mu<=0 states
    ax = axes[1]
    labels, fracs, colors = [], [], []
    for label, e in per_env.items():
        Gpos = np.maximum(e['G'], 0.0)
        total = Gpos.sum()
        frac_low_mu = float(Gpos[e['mu'] <= 0].sum() / max(total, 1e-9))
        labels.append(label); fracs.append(frac_low_mu); colors.append(e['color'])
    xs = np.arange(len(labels))
    ax.bar(xs, fracs, color=colors, edgecolor='black', lw=0.6, alpha=0.85)
    for x, f in zip(xs, fracs):
        ax.text(x, f + 0.01, f'{f:.2f}', ha='center', va='bottom', fontsize=11)
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel(r'Share of total $G^+$ from $\mu(s)\leq 0$ states')
    ax.set_title('(b) Precondition-failure share of the cell-level gap')
    ax.set_ylim(0, max(max(fracs) * 1.25, 0.2))
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    out_pdf = os.path.join(ROOT, 'figures', 'exp21_perstate_contribution.pdf')
    out_png = os.path.join(ROOT, 'figures', 'exp21_perstate_contribution.png')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()
    print(f"[saved] {out_pdf}")

    # Summary print
    print("\nPer-env summary across shifted gravities:")
    print(f"{'env':<14}{'n':>6}{'frac_mu<=0':>12}{'mean_G':>10}{'mean_G(mu<=0)':>16}{'mean_G(mu>0)':>16}")
    for label, e in per_env.items():
        n = len(e['mu'])
        m0 = e['G'][e['mu'] <= 0]
        m1 = e['G'][e['mu'] >  0]
        print(f"{label:<14}{n:>6d}{np.mean(e['mu']<=0):>12.2f}"
              f"{np.mean(e['G']):>10.1f}"
              f"{(np.mean(m0) if len(m0) else 0):>16.1f}"
              f"{(np.mean(m1) if len(m1) else 0):>16.1f}")


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if mode == 'run':   run()
    elif mode == 'plot': plot()
    elif mode == 'both': run(); plot()
    else: print(f"Unknown mode: {mode}"); sys.exit(1)
