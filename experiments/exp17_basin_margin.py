"""
Experiment 17: Basin-Margin Distribution F_{delta_basin}
========================================================
For each MuJoCo env (HalfCheetah, Ant, Hopper) at each gravity, measure the
per-state basin margin

    delta_basin(s) = Q(s, a*(s)) - Q(s, a**(s)),

where a* is the global Q-maximizer and a** is the best Q value attainable from
a *different* basin, found by multi-start gradient ascent on the critic from
random initial actions, retaining only candidates that ended up far from a*.

Under Assumption (basin-dominance), delta_basin(s) > 0 with a uniform lower
bound. The empirical CDF F_{delta_basin}(x) over states is the continuous-action
analog of the action-gap CDF F_kappa: states with delta_basin(s) > c||delta theta||
stay within their original basin and the quadratic transfer bound applies; states
with delta_basin(s) <= c||delta theta|| undergo basin-switching jumps that push
the transfer-gap exponent above 2.

Prediction: Hopper's F_{delta_basin} is supported well above tested ||delta theta||
(no basin-switching) -> quadratic regime. HalfCheetah/Ant's F_{delta_basin}
intersects tested ||delta theta|| (basin-switching states) -> super-quadratic.

Usage:
  python exp17_basin_margin.py sanity   # tiny run to verify pipeline
  python exp17_basin_margin.py run      # full run, save pkl
  python exp17_basin_margin.py plot     # generate figure from cached pkl
"""

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
import torch
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
OUT_PKL = os.path.join(CACHE_DIR, 'results_basin_margin.pkl')
SEED = 42
GRAVITIES = [5.0, 7.0, 9.81, 12.0, 15.0]
SRC_GRAVITY = 9.81

N_STATES = 150
N_STARTS = 64           # random starts per state for runner-up search
GA_STEPS = 80           # gradient-ascent steps per start
GA_LR = 5e-3
FAR_THRESHOLD = 0.3     # ||a - a*||_inf > this counts as different basin (action range [-1,1])

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


def grad_ascent_batched(critic, s_tensor, a_init, lr=GA_LR, steps=GA_STEPS):
    """
    Run gradient ascent on critic Q1 jointly for a batch of (s replicated, a_init) pairs.

    Args
    ----
    s_tensor: (1, ds) torch float
    a_init:   (B, da) torch float, initial actions in [-1, 1]
    Returns
    -------
    a_final: (B, da) refined actions, clamped to [-1, 1]
    q_final: (B,) Q values at refined actions
    """
    B = a_init.shape[0]
    s_rep = s_tensor.expand(B, -1)
    a = a_init.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([a], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        q = critic(s_rep, a)[0].squeeze(-1)
        loss = -q.sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            a.clamp_(-1.0, 1.0)
    with torch.no_grad():
        q_final = critic(s_rep, a)[0].squeeze(-1).cpu().numpy()
        a_final = a.detach().cpu().numpy()
    return a_final, q_final


def compute_basin_margins(model, states, n_starts=N_STARTS, far_threshold=FAR_THRESHOLD,
                          device='cuda', seed=0):
    """
    For each state s in `states`, return delta_basin(s) = Q(s, a*) - max Q(s, a)
    over actions a obtained via gradient ascent from random initializations and
    ending up far (||a - a*||_inf > far_threshold) from a*.

    Returns a dict with arrays:
      delta_basin: (N,) margin per state
      q_star:      (N,) Q at a*
      q_runnerup:  (N,) Q at a** (best far candidate)
      n_far:       (N,) number of starts that ended up far from a*
    """
    critic = model.critic
    actor = model.actor
    critic.eval()
    actor.eval()

    s0 = torch.as_tensor(states[0], dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        a0 = actor(s0).squeeze(0)
    d_a = a0.numel()

    rng = np.random.default_rng(seed)

    delta_basin = np.full(len(states), np.nan)
    q_star_arr = np.full(len(states), np.nan)
    q_runner_arr = np.full(len(states), np.nan)
    n_far_arr = np.zeros(len(states), dtype=np.int32)

    for idx, s_np in enumerate(states):
        s_tensor = torch.as_tensor(s_np, dtype=torch.float32, device=device).unsqueeze(0)

        # Refine a* by gradient ascent from SAC's deterministic output
        with torch.no_grad():
            a_init = actor(s_tensor).squeeze(0)
        a_star_init = a_init.unsqueeze(0)  # (1, d_a)
        a_star_arr, q_star_vals = grad_ascent_batched(critic, s_tensor, a_star_init)
        a_star = a_star_arr[0]
        q_star = float(q_star_vals[0])

        # Multi-start gradient ascent from random initial actions in [-1, 1]^d
        a_random = rng.uniform(-1.0, 1.0, size=(n_starts, d_a)).astype(np.float32)
        a_random_t = torch.as_tensor(a_random, device=device)
        a_local, q_local = grad_ascent_batched(critic, s_tensor, a_random_t)

        # Keep only candidates whose final action is far from a* in L_inf
        dist_inf = np.max(np.abs(a_local - a_star[None, :]), axis=1)
        far_mask = dist_inf > far_threshold
        n_far = int(far_mask.sum())
        n_far_arr[idx] = n_far

        if n_far > 0:
            q_runner = float(np.max(q_local[far_mask]))
        else:
            # No alternative basin found; basin is uniquely dominant in the search range.
            # Record max Q over all (necessarily near-a*) candidates as a conservative proxy
            q_runner = float(np.max(q_local))

        q_star_arr[idx] = q_star
        q_runner_arr[idx] = q_runner
        delta_basin[idx] = q_star - q_runner

    return {
        'delta_basin': delta_basin,
        'q_star': q_star_arr,
        'q_runnerup': q_runner_arr,
        'n_far': n_far_arr,
    }


def run(sanity=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    if sanity:
        n_states = 20
        n_starts = 16
        gravities = [SRC_GRAVITY]
        out_pkl = os.path.join(CACHE_DIR, 'results_basin_margin_sanity.pkl')
    else:
        n_states = N_STATES
        n_starts = N_STARTS
        gravities = GRAVITIES
        out_pkl = OUT_PKL

    results = {}
    for env_id, cfg in ENV_CONFIGS.items():
        print(f"\n{'='*60}\n  {env_id}\n{'='*60}", flush=True)
        results[env_id] = {'gravities': gravities, 'label': cfg['label'], 'color': cfg['color'],
                           'stats': {}}
        for g in gravities:
            t0 = time.time()
            print(f"  g={g:5.2f}: load model + collect {n_states} states ...", flush=True)
            model, env = load_model(env_id, g)
            env.close()
            states = collect_states(env_id, g, n_states=n_states, model=model)

            print(f"           multi-start basin search (K={n_starts}) ...", flush=True)
            stats = compute_basin_margins(model, states, n_starts=n_starts)
            db = stats['delta_basin']

            summary = {
                'delta_basin': db.tolist(),
                'q_star': stats['q_star'].tolist(),
                'q_runnerup': stats['q_runnerup'].tolist(),
                'n_far': stats['n_far'].tolist(),
                'median': float(np.nanmedian(db)),
                'q25': float(np.nanpercentile(db, 25)),
                'q75': float(np.nanpercentile(db, 75)),
                'frac_unique_basin': float(np.mean(stats['n_far'] == 0)),
            }
            results[env_id]['stats'][g] = summary
            print(f"           median delta_basin={summary['median']:+.3f}  "
                  f"IQR=[{summary['q25']:+.3f},{summary['q75']:+.3f}]  "
                  f"frac_unique_basin={summary['frac_unique_basin']:.2f}  "
                  f"({time.time()-t0:.1f}s)", flush=True)

    with open(out_pkl, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved to {out_pkl}")


def plot(save_dir='../figures', sanity=False):
    pkl_path = os.path.join(CACHE_DIR, 'results_basin_margin_sanity.pkl' if sanity else 'results_basin_margin.pkl')
    if not os.path.exists(pkl_path):
        print(f"No results at {pkl_path}. Run first.")
        return
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)

    os.makedirs(save_dir, exist_ok=True)

    # Panel (a): F_{delta_basin} CDF at source gravity per env
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    for env_id, data in results.items():
        if SRC_GRAVITY not in data['stats']:
            continue
        db = np.array(data['stats'][SRC_GRAVITY]['delta_basin'])
        db_sorted = np.sort(db)
        cdf = np.arange(1, len(db_sorted) + 1) / len(db_sorted)
        ax.plot(db_sorted, cdf, '-', color=data['color'], lw=2.2, label=data['label'])
    ax.axvline(0, color='k', lw=0.7, ls='--', alpha=0.6)
    ax.set_xlabel(r'Basin margin $\delta_{\mathrm{basin}}(s)$')
    ax.set_ylabel(r'$F_{\delta_{\mathrm{basin}}}(x)$ (fraction of states $\leq x$)')
    ax.set_title(f'(a) Basin-margin CDF at source $g={SRC_GRAVITY}$')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    # Panel (b): median delta_basin vs gravity
    ax = axes[1]
    for env_id, data in results.items():
        gs = sorted(data['stats'].keys())
        meds = [data['stats'][g]['median'] for g in gs]
        q25 = [data['stats'][g]['q25'] for g in gs]
        q75 = [data['stats'][g]['q75'] for g in gs]
        ax.plot(gs, meds, 'o-', color=data['color'], lw=2, ms=7, label=data['label'])
        ax.fill_between(gs, q25, q75, color=data['color'], alpha=0.2)
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.7)
    ax.axvline(SRC_GRAVITY, color='gray', lw=0.8, ls=':', alpha=0.7)
    ax.set_xlabel('Gravity (m/s$^2$)')
    ax.set_ylabel(r'Median $\delta_{\mathrm{basin}}(s)$')
    ax.set_title(r'(b) Basin margin vs gravity')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    # Panel (c): fraction of states with unique basin (no far candidate found)
    ax = axes[2]
    width = 0.25
    gs = sorted(next(iter(results.values()))['stats'].keys())
    x = np.arange(len(gs))
    for i, (env_id, data) in enumerate(results.items()):
        fracs = [data['stats'][g]['frac_unique_basin'] for g in gs]
        ax.bar(x + (i - 1) * width, fracs, width, color=data['color'],
               label=data['label'], alpha=0.85, edgecolor='black', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{g}' for g in gs])
    if SRC_GRAVITY in gs:
        ax.axvline(gs.index(SRC_GRAVITY), color='gray', lw=0.8, ls=':', alpha=0.7)
    ax.set_xlabel('Gravity (m/s$^2$)')
    ax.set_ylabel('Fraction of states with uniquely dominant basin')
    ax.set_title('(c) Basin uniqueness across gravities')
    ax.set_ylim(0, 1.02)
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_pdf = os.path.join(save_dir, 'exp17_basin_margin.pdf')
    out_png = os.path.join(save_dir, 'exp17_basin_margin.png')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved figure to {out_pdf}")

    # Console summary
    print("\nSummary (median delta_basin per gravity):")
    print(f"{'Env':<14}" + ''.join(f' g={g:<5.2f}' for g in gs))
    for env_id, data in results.items():
        row = f"{data['label']:<14}"
        for g in gs:
            row += f' {data["stats"][g]["median"]:+.3f} '
        print(row)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'sanity'
    if mode == 'sanity':
        run(sanity=True)
        plot(sanity=True)
    elif mode == 'run':
        run(sanity=False)
    elif mode == 'plot':
        plot(sanity=False)
    elif mode == 'both':
        run(sanity=False)
        plot(sanity=False)
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
