"""
Experiment 18: Source-Side L_Q Estimator (Proposition prop:lq-estimator)
=======================================================================
Validates the certified L_Q upper bound

    hat L_Q^cert = max_{s,a,k} [ (|hat Q_{theta_{k+1}}(s,a) - hat Q_{theta_k}(s,a)|
                                  + 2 epsilon_Q) / h_k  +  H_Q * h_k ]

on existing MuJoCo SAC checkpoints (HalfCheetah, Ant, Hopper) at the gravity
ensemble theta_k in {5.0, 7.0, 9.81, 12.0, 15.0}, treating gravity as the 1-d
source-side parameter.

For each env:
  - Load 5 SAC checkpoints (seed 42).
  - Collect ~N_STATES on-policy states from the source gravity (9.81).
  - For each (state s, source-policy action a), evaluate the twin-Q minimum
    of every checkpoint, giving hat Q_{theta_k}(s, a) for k = 1..5.
  - Compute first-order finite differences (lower bound on local |dQ/dtheta|).
  - Compute second-order finite differences as a data-driven H_Q lower bound.
  - Combine into hat L_Q^cert at user-supplied (H_Q, epsilon_Q).
  - Compare against the implied stability radius hat rho = hat kappa / (2 hat L_Q^cert)
    using a representative kappa derived from the action gap distribution.

This experiment validates the protocol; it does NOT verify the C^2 assumption
(taken as input). Results report the FD-based numbers and the H_Q lower bound
from the data so a reader can sanity-check the inflation term.

Usage:
  python exp18_lq_estimator.py sanity   # 50 states, no save
  python exp18_lq_estimator.py run      # full N_STATES, save pkl
  python exp18_lq_estimator.py plot     # plot from cached pkl
  python exp18_lq_estimator.py discard  # remove pkl + figure
"""

import numpy as np
import torch
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
OUT_PKL = os.path.join(CACHE_DIR, 'results_lq_estimator.pkl')
OUT_FIG = os.path.join(CACHE_DIR, 'fig_lq_estimator.pdf')
SEED = 42
GRAVITIES = [5.0, 7.0, 9.81, 12.0, 15.0]
SRC_GRAVITY = 9.81
N_STATES = 300

# Defaults for the inflation-term inputs (user supplies these in practice).
# We read epsilon_Q from a coarse SAC twin-Q gap estimate, and H_Q from second-
# order FD as a data-driven lower bound (the user must check / inflate).
DEFAULT_EPS_Q = 1.0   # critic error magnitude (placeholder; report side-by-side)
DEFAULT_H_Q   = None  # if None, take the second-order FD lower bound from data

ENV_CONFIGS = {
    'HalfCheetah-v4': {'label': 'HalfCheetah', 'color': '#1f77b4'},
    'Ant-v4':         {'label': 'Ant',         'color': '#ff7f0e'},
    'Hopper-v4':      {'label': 'Hopper',      'color': '#2ca02c'},
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


def critic_min_q(model, states, actions):
    """Twin-Q minimum: SAC stores two critics in model.critic.qf{0,1}."""
    device = model.device
    s_t = torch.as_tensor(states, dtype=torch.float32, device=device)
    a_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
    with torch.no_grad():
        q1, q2 = model.critic(s_t, a_t)
        q = torch.minimum(q1, q2).squeeze(-1).cpu().numpy()
    return q


def measure_env(env_id, gravities, n_states):
    """Returns dict with per-(state,action) Q values across gravity ensemble."""
    print(f"\n[{env_id}] loading source model and collecting {n_states} states ...")
    src_model, _ = load_model(env_id, SRC_GRAVITY)
    states = collect_states(env_id, SRC_GRAVITY, n_states, src_model)
    print(f"  collected {len(states)} states")

    actions = []
    for s in states:
        a, _ = src_model.predict(s, deterministic=True)
        actions.append(a)
    actions = np.array(actions)

    q_values = np.zeros((len(gravities), len(states)), dtype=np.float64)
    for ki, g in enumerate(gravities):
        if g == SRC_GRAVITY:
            model = src_model
        else:
            model, env = load_model(env_id, g)
        q = critic_min_q(model, states, actions)
        q_values[ki] = q
        if g != SRC_GRAVITY:
            env.close()
            del model
        print(f"  g={g:.2f}: Q stats min={q.min():.3f} median={np.median(q):.3f} max={q.max():.3f}")

    return {
        'env_id': env_id,
        'gravities': np.array(gravities),
        'states': states,
        'actions': actions,
        'q_values': q_values,  # shape (K, N)
    }


def compute_lq_stats(record, eps_q, H_Q=None):
    """Compute first-order FD distribution and second-order FD H_Q lower bound.
    Returns dict with arrays/scalars used by the figure and Appendix table."""
    gravities = record['gravities']
    Q = record['q_values']  # (K, N)
    K, N = Q.shape

    # First-order FD per consecutive pair.
    dQ = np.diff(Q, axis=0)             # (K-1, N)
    h = np.diff(gravities)              # (K-1,)
    fd1 = np.abs(dQ) / h[:, None]       # (K-1, N) lower-bound on local |dQ/dtheta|

    # Second-order FD as H_Q lower bound on consecutive triples.
    # Use uneven-spacing formula:
    #   Q''_k ≈ 2 * (h_{k} * (Q_{k+1} - Q_k) - h_{k-1} * (Q_k - Q_{k-1}))
    #              / (h_{k-1} * h_{k} * (h_{k-1} + h_{k}))
    fd2 = []
    for k in range(1, K - 1):
        hL = gravities[k] - gravities[k - 1]
        hR = gravities[k + 1] - gravities[k]
        num = 2.0 * (hL * (Q[k + 1] - Q[k]) - hR * (Q[k] - Q[k - 1]))
        den = hL * hR * (hL + hR)
        fd2.append(np.abs(num / den))
    fd2 = np.array(fd2)  # (K-2, N)

    H_Q_data = float(fd2.max()) if fd2.size else 0.0
    H_Q_used = H_Q if H_Q is not None else H_Q_data

    # hat L_Q^cert per pair k, max over states.
    cert_per_pair = (fd1.max(axis=1) + 2 * eps_q / h) + H_Q_used * h  # (K-1,)
    L_Q_cert = float(cert_per_pair.max())
    L_Q_fd_only = float((np.abs(dQ).max(axis=1) / h).max())

    return {
        'fd1': fd1,
        'fd2': fd2,
        'h': h,
        'gravities': gravities,
        'eps_q': eps_q,
        'H_Q_used': H_Q_used,
        'H_Q_data_lb': H_Q_data,
        'L_Q_fd_only': L_Q_fd_only,
        'L_Q_cert': L_Q_cert,
        'cert_per_pair': cert_per_pair,
    }


def summarize(results, eps_q, H_Q=None):
    print("\n=== Source-side L_Q estimator summary (eps_Q={:.2f}, H_Q={}) ===".format(
        eps_q, 'data-LB' if H_Q is None else f'{H_Q:.3f}'))
    header = f"{'env':<14}{'L_Q^fd':>12}{'H_Q_LB':>12}{'L_Q^cert':>12}{'h_max':>8}"
    print(header)
    print('-' * len(header))
    summary = {}
    for env_id, rec in results.items():
        s = compute_lq_stats(rec, eps_q=eps_q, H_Q=H_Q)
        summary[env_id] = s
        print(f"{env_id:<14}{s['L_Q_fd_only']:>12.3f}{s['H_Q_data_lb']:>12.3f}"
              f"{s['L_Q_cert']:>12.3f}{s['h'].max():>8.2f}")
    return summary


def run_all(sanity=False):
    n_states = 50 if sanity else N_STATES
    results = {}
    t0 = time.time()
    for env_id in ENV_CONFIGS:
        results[env_id] = measure_env(env_id, GRAVITIES, n_states)
    print(f"\n[done] elapsed {time.time()-t0:.1f}s")
    if not sanity:
        with open(OUT_PKL, 'wb') as f:
            pickle.dump(results, f)
        print(f"[saved] {OUT_PKL}")
    else:
        print("[sanity] no pkl saved.")
    return results


def plot(results, eps_q=DEFAULT_EPS_Q, H_Q=DEFAULT_H_Q, out_path=None):
    """3-panel figure:
      (a) Per-pair first-order FD distribution (boxplot per env, per pair).
      (b) Convergence: L_Q^cert as ensemble grows (drop pairs from the ends).
      (c) Inflation breakdown at the worst pair: FD term vs eps_Q/h vs H_Q*h.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # ---- (a) per-pair FD distribution
    ax = axes[0]
    pair_labels = [f"{GRAVITIES[k]:.1f}-{GRAVITIES[k+1]:.1f}" for k in range(len(GRAVITIES)-1)]
    width = 0.25
    x_base = np.arange(len(pair_labels))
    for i, (env_id, cfg) in enumerate(ENV_CONFIGS.items()):
        if env_id not in results:
            continue
        rec = results[env_id]
        Q = rec['q_values']; h = np.diff(rec['gravities'])
        fd1 = np.abs(np.diff(Q, axis=0)) / h[:, None]
        meds = np.median(fd1, axis=1)
        p90s = np.percentile(fd1, 90, axis=1)
        ax.bar(x_base + (i - 1) * width, meds, width, color=cfg['color'],
               alpha=0.7, label=f"{cfg['label']} (median)")
        ax.scatter(x_base + (i - 1) * width, p90s, color=cfg['color'],
                   marker='_', s=80, zorder=3)
    ax.set_xticks(x_base); ax.set_xticklabels(pair_labels, rotation=15)
    ax.set_xlabel('gravity pair $(\\theta_k, \\theta_{k+1})$')
    ax.set_ylabel(r'$|\Delta\hat Q|/h_k$')
    ax.set_title('(a) First-order FD per pair (median bars, p90 ticks)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ---- (b) L_Q^cert vs ensemble subset size
    ax = axes[1]
    for env_id, cfg in ENV_CONFIGS.items():
        if env_id not in results:
            continue
        rec = results[env_id]
        K = len(rec['gravities'])
        Ks = list(range(2, K + 1))
        cert_vals = []
        for K_use in Ks:
            sub = {
                'env_id': env_id,
                'gravities': rec['gravities'][:K_use],
                'states': rec['states'],
                'actions': rec['actions'],
                'q_values': rec['q_values'][:K_use],
            }
            s = compute_lq_stats(sub, eps_q=eps_q, H_Q=H_Q)
            cert_vals.append(s['L_Q_cert'])
        ax.plot(Ks, cert_vals, 'o-', color=cfg['color'], lw=2, label=cfg['label'])
    ax.set_xlabel('ensemble size $K$ (using leftmost $K$ gravities)')
    ax.set_ylabel(r'$\hat L_Q^{\mathrm{cert}}$')
    ax.set_title('(b) Convergence with ensemble size')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ---- (c) inflation breakdown at the worst pair
    ax = axes[2]
    env_names = []
    fd_terms = []; eps_terms = []; H_terms = []
    for env_id, cfg in ENV_CONFIGS.items():
        if env_id not in results:
            continue
        rec = results[env_id]
        s = compute_lq_stats(rec, eps_q=eps_q, H_Q=H_Q)
        kstar = int(np.argmax(s['cert_per_pair']))
        h_k = s['h'][kstar]
        Q = rec['q_values']
        dQ = np.abs(Q[kstar + 1] - Q[kstar])
        fd_part = dQ.max() / h_k
        eps_part = 2 * eps_q / h_k
        H_part = s['H_Q_used'] * h_k
        env_names.append(cfg['label'])
        fd_terms.append(fd_part); eps_terms.append(eps_part); H_terms.append(H_part)
    x = np.arange(len(env_names))
    ax.bar(x, fd_terms, label='FD: $|\\Delta\\hat Q|/h_k$', color='#4c72b0')
    ax.bar(x, eps_terms, bottom=fd_terms, label='$2\\varepsilon_Q/h_k$', color='#dd8452')
    ax.bar(x, H_terms, bottom=np.array(fd_terms) + np.array(eps_terms),
           label='$H_Q\\,h_k$', color='#55a868')
    ax.set_xticks(x); ax.set_xticklabels(env_names)
    ax.set_ylabel(r'contribution to $\hat L_Q^{\mathrm{cert}}$')
    ax.set_title('(c) Worst-pair inflation breakdown')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if out_path is None:
        out_path = OUT_FIG
    plt.savefig(out_path)
    plt.close()
    print(f"[saved] {out_path}")


def discard_artifacts():
    for p in [OUT_PKL, OUT_FIG]:
        if os.path.exists(p):
            os.remove(p)
            print(f"[removed] {p}")


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'sanity'
    if mode == 'sanity':
        results = run_all(sanity=True)
        summarize(results, eps_q=DEFAULT_EPS_Q, H_Q=DEFAULT_H_Q)
    elif mode == 'run':
        results = run_all(sanity=False)
        summarize(results, eps_q=DEFAULT_EPS_Q, H_Q=DEFAULT_H_Q)
        plot(results, eps_q=DEFAULT_EPS_Q, H_Q=DEFAULT_H_Q)
    elif mode == 'plot':
        with open(OUT_PKL, 'rb') as f:
            results = pickle.load(f)
        summarize(results, eps_q=DEFAULT_EPS_Q, H_Q=DEFAULT_H_Q)
        plot(results, eps_q=DEFAULT_EPS_Q, H_Q=DEFAULT_H_Q)
    elif mode == 'discard':
        discard_artifacts()
    else:
        print(f"unknown mode: {mode}")
        sys.exit(1)
