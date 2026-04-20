"""
Experiment 16: Hessian Diagnostic for Strong Concavity Assumption
==================================================================
For each MuJoCo environment (HalfCheetah, Ant, Hopper) at each gravity,
measure the top eigenvalue of the action-Hessian of the critic:

    mu(s) = -lambda_max( nabla^2_aa Q(s, a*(s)) )

Under Assumption (strong concavity), mu(s) >= mu_bar > 0 uniformly.
If mu(s) collapses to near zero at extreme gravities, this directly
corroborates the basin-dominance / concavity-weakening explanation for
super-quadratic transfer gap slopes on HalfCheetah/Ant.

Hopper should stay closer to near-quadratic, so mu should remain
well-separated from zero across gravities.

Usage:
  python exp16_hessian_diagnostic.py run    # compute Hessian stats, save pkl
  python exp16_hessian_diagnostic.py plot   # generate figure from cached pkl
"""

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
import torch
from torch.autograd.functional import hessian
import os
import sys
import pickle
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
OUT_PKL = os.path.join(CACHE_DIR, 'results_hessian.pkl')
SEED = 42
GRAVITIES = [5.0, 7.0, 9.81, 12.0, 15.0]
N_STATES = 150  # per (env, gravity); Hessian is O(d^2) autograd calls

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
    cache_path = os.path.join(CACHE_DIR, f'sac_{env_id}_g{gravity:.2f}_s{seed}_t500000.zip')
    env = make_env(env_id, gravity)
    model = SAC.load(cache_path, env=env, device='cuda')
    return model, env


def collect_states(env_id, gravity, n_states=N_STATES, model=None):
    """Rollout the target-trained policy (if given) at target gravity to collect on-policy states."""
    env = make_env(env_id, gravity)
    states = []
    s, _ = env.reset(seed=SEED + 999)
    for t in range(20000):
        if model is not None:
            a, _ = model.predict(s, deterministic=True)
        else:
            a = env.action_space.sample()
        s, r, term, trunc, _ = env.step(a)
        if t >= 200 and t % 5 == 0:
            states.append(s.copy())
        if term or trunc:
            s, _ = env.reset(seed=SEED + 999 + t)
        if len(states) >= n_states:
            break
    env.close()
    return np.array(states[:n_states])


def compute_hessian_stats(model, states, device='cuda', eps=0.05, n_dirs=None, seed=0):
    """
    For each state, estimate the action-Hessian of Q via finite differences
    along random directions, then report

        mu(s) = -lambda_max( H(s) ).

    Note: SAC's critic is a ReLU MLP (piecewise linear), so analytic autograd
    gives Hessian = 0 almost everywhere. Finite differences at a non-infinitesimal
    scale eps (set below as a fraction of the action range) recover the
    *effective* curvature seen over a neighborhood spanning multiple linear
    pieces, which is exactly what Assumption (strong concavity) is about at a
    non-asymptotic scale.

    Returns np.ndarray of mu(s) values.
    """
    critic = model.critic
    actor = model.actor
    critic.eval()
    actor.eval()

    # Determine action dim from one forward pass
    s0 = torch.as_tensor(states[0], dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        a0 = actor(s0).squeeze(0)
    d = a0.numel()
    if n_dirs is None:
        n_dirs = max(4 * d * (d + 1) // 2, 60)  # over-determined LSQ

    # Action scale: use SAC default [-1, 1] squashed range; eps in units of that range
    rng = np.random.default_rng(seed)

    # Pre-sample directions (shared across states for efficiency, resampled per env call)
    dirs = rng.standard_normal((n_dirs, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    dirs_t = torch.as_tensor(dirs, dtype=torch.float32, device=device)

    # Design matrix for LSQ: d_val = v^T H v with H symmetric
    # Unknowns in upper triangle: H[i,i] gets coeff v_i^2, H[i,j] (i<j) gets 2 v_i v_j
    n_params = d * (d + 1) // 2
    A = np.zeros((n_dirs, n_params))
    for k in range(n_dirs):
        v = dirs[k]
        col = 0
        for i in range(d):
            for j in range(i, d):
                A[k, col] = v[i] * v[j] if i == j else 2.0 * v[i] * v[j]
                col += 1

    mu_values = []
    for s_np in states:
        s_tensor = torch.as_tensor(s_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            a_init = actor(s_tensor).squeeze(0)

        # Refine a* by gradient ascent on Q1 starting from SAC's output
        a_opt = a_init.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([a_opt], lr=5e-3)
        for _ in range(80):
            opt.zero_grad()
            loss = -critic(s_tensor, a_opt.unsqueeze(0))[0].squeeze()
            loss.backward()
            opt.step()
            with torch.no_grad():
                a_opt.clamp_(-1.0, 1.0)
        a_opt = a_opt.detach()

        with torch.no_grad():
            # Q at a*
            q0 = critic(s_tensor, a_opt.unsqueeze(0))[0].item()

            # Q at a* ± eps * v_k, batched
            a_plus = a_opt.unsqueeze(0) + eps * dirs_t    # (n_dirs, d)
            a_minus = a_opt.unsqueeze(0) - eps * dirs_t
            s_rep = s_tensor.expand(n_dirs, -1)
            qp = critic(s_rep, a_plus)[0].squeeze(-1).cpu().numpy()
            qm = critic(s_rep, a_minus)[0].squeeze(-1).cpu().numpy()

        d_vals = (qp + qm - 2.0 * q0) / (eps ** 2)   # (n_dirs,)
        # Solve A x = d_vals for upper-triangular H entries
        x, *_ = np.linalg.lstsq(A, d_vals, rcond=None)
        # Assemble symmetric H
        H = np.zeros((d, d))
        col = 0
        for i in range(d):
            for j in range(i, d):
                H[i, j] = x[col]
                H[j, i] = x[col]
                col += 1
        eigs = np.linalg.eigvalsh(H)
        lambda_max = float(eigs[-1])
        mu_values.append(-lambda_max)

    return np.array(mu_values)


def run():
    os.makedirs(CACHE_DIR, exist_ok=True)
    results = {}

    for env_id, cfg in ENV_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  {env_id}")
        print(f"{'='*60}")
        results[env_id] = {'gravities': GRAVITIES, 'label': cfg['label'], 'color': cfg['color'],
                           'mu_stats': {}}
        for g in GRAVITIES:
            print(f"  g={g:5.2f}: loading model, collecting states...", flush=True)
            model, env = load_model(env_id, g)
            env.close()

            # Collect on-policy states at this gravity using the target-trained model
            states = collect_states(env_id, g, n_states=N_STATES, model=model)

            print(f"           computing Hessian at {len(states)} states...", flush=True)
            mu = compute_hessian_stats(model, states)
            mu_valid = mu[~np.isnan(mu)]

            stats = {
                'median': float(np.median(mu_valid)),
                'q25': float(np.percentile(mu_valid, 25)),
                'q75': float(np.percentile(mu_valid, 75)),
                'frac_near_zero': float(np.mean(np.abs(mu_valid) < 0.1)),
                'frac_nonconcave': float(np.mean(mu_valid < 0)),
                'raw': mu_valid.tolist(),
            }
            results[env_id]['mu_stats'][g] = stats
            print(f"           mu: median={stats['median']:.3f}, "
                  f"IQR=[{stats['q25']:.3f},{stats['q75']:.3f}], "
                  f"frac(|mu|<0.1)={stats['frac_near_zero']:.2f}, "
                  f"frac(mu<0)={stats['frac_nonconcave']:.2f}")

    with open(OUT_PKL, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved Hessian diagnostic results to {OUT_PKL}")


def plot(save_dir='../figures'):
    os.makedirs(save_dir, exist_ok=True)
    if not os.path.exists(OUT_PKL):
        print(f"No results at {OUT_PKL}. Run 'run' first.")
        return

    with open(OUT_PKL, 'rb') as f:
        results = pickle.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Panel (a): median mu(s) vs gravity, per env, with IQR band
    ax = axes[0]
    for env_id, data in results.items():
        gs = data['gravities']
        label = data['label']
        color = data['color']
        meds = [data['mu_stats'][g]['median'] for g in gs]
        q25 = [data['mu_stats'][g]['q25'] for g in gs]
        q75 = [data['mu_stats'][g]['q75'] for g in gs]
        ax.plot(gs, meds, 'o-', color=color, lw=2, ms=7, label=label)
        ax.fill_between(gs, q25, q75, color=color, alpha=0.2)
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.7)
    ax.axvline(9.81, color='gray', lw=0.8, ls=':', alpha=0.7)
    ax.set_xlabel('Gravity (m/s$^2$)')
    ax.set_ylabel(r'Strong-concavity margin $\mu(s) = -\lambda_{\max}(\nabla^2_{aa} Q)$')
    ax.set_title(r'(a) Curvature of $Q$ at $a^*$ across gravities')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    # Panel (b): fraction of states with near-zero or non-concave curvature
    ax = axes[1]
    width = 0.25
    x = np.arange(len(GRAVITIES))
    for i, (env_id, data) in enumerate(results.items()):
        fracs = [data['mu_stats'][g]['frac_near_zero'] for g in data['gravities']]
        ax.bar(x + (i - 1) * width, fracs, width, color=data['color'],
               label=data['label'], alpha=0.85, edgecolor='black', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{g}' for g in GRAVITIES])
    ax.axvline(np.where(np.array(GRAVITIES) == 9.81)[0][0], color='gray', lw=0.8, ls=':', alpha=0.7)
    ax.set_xlabel('Gravity (m/s$^2$)')
    ax.set_ylabel(r'Fraction of states with $|\mu(s)| < 0.1$')
    ax.set_title('(b) Prevalence of flat curvature regions')
    ax.set_ylim(0, 1.02)
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_pdf = os.path.join(save_dir, 'exp16_hessian_diagnostic.pdf')
    out_png = os.path.join(save_dir, 'exp16_hessian_diagnostic.png')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved figure to {out_pdf}")

    print("\nSummary:")
    print(f"{'Env':<14}" + ''.join(f' g={g:<5.2f}' for g in GRAVITIES))
    for env_id, data in results.items():
        row = f"{data['label']:<14}"
        for g in data['gravities']:
            row += f' {data["mu_stats"][g]["median"]:+.3f} '
        print(row)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'run'
    if mode == 'run':
        run()
    elif mode == 'plot':
        plot()
    elif mode == 'both':
        run()
        plot()
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
