"""
Experiment 14: F_kappa as Predictive Quantity in Neural Settings
================================================================
For each MuJoCo environment (HalfCheetah, Ant, Hopper), estimate the empirical
action-gap distribution F_kappa from the trained SAC model at source gravity,
then show that F_kappa shape predicts the OC decay pattern across environments.

- Steep F_kappa near origin → many near-tie actions → fast OC decay (Hopper)
- Flat F_kappa near origin → large gaps → slow OC decay (Ant)
"""

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
import torch
import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'font.size': 13, 'font.family': 'serif',
    'axes.labelsize': 14, 'axes.titlesize': 14,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 10, 'figure.dpi': 150,
})

CACHE_DIR = '../cache_exp10'
SEED = 42
SRC_GRAVITY = 9.81
GRAVITIES = [5.0, 7.0, 9.81, 12.0, 15.0]
N_EVAL_EPISODES = 20
N_SAMPLE_STATES = 500
N_ACTION_PERTURBATIONS = 50
PERTURB_SCALE = 0.3  # std of Gaussian perturbation relative to action range

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


def collect_states(model, env_id, n_states=N_SAMPLE_STATES):
    """Collect diverse states from rollouts."""
    env = make_env(env_id, SRC_GRAVITY)
    states = []
    s, _ = env.reset(seed=SEED + 999)
    for t in range(10000):
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


def estimate_action_gaps(model, states, env_id):
    """
    Estimate per-state action gap for continuous actions.

    For each state s:
    1. Get optimal action a* = pi(s)
    2. Sample N perturbed actions around a*
    3. Compute Q(s, a*) - Q(s, a_perturbed) for each
    4. Gap = min of these differences (how close the nearest competitor is)

    This measures the "curvature" of the Q-landscape: flat Q → small gap → fragile.
    """
    env = make_env(env_id, SRC_GRAVITY)
    action_low = env.action_space.low
    action_high = env.action_space.high
    action_range = action_high - action_low
    env.close()

    gaps = []

    # Get Q-function from SAC
    critic = model.critic
    actor = model.actor

    for s in states:
        s_tensor = torch.FloatTensor(s).unsqueeze(0).to(model.device)

        # Optimal action
        with torch.no_grad():
            a_opt = actor(s_tensor)  # deterministic
            # SAC has two critics, take minimum (as SAC does)
            q1_opt, q2_opt = critic(s_tensor, a_opt)
            q_opt = torch.min(q1_opt, q2_opt).item()

        # Perturbed actions
        min_gap = float('inf')
        for _ in range(N_ACTION_PERTURBATIONS):
            noise = np.random.randn(len(action_low)) * PERTURB_SCALE * action_range
            a_pert = np.clip(a_opt.cpu().numpy().flatten() + noise, action_low, action_high)
            a_pert_tensor = torch.FloatTensor(a_pert).unsqueeze(0).to(model.device)

            with torch.no_grad():
                q1_p, q2_p = critic(s_tensor, a_pert_tensor)
                q_pert = torch.min(q1_p, q2_p).item()

            gap = q_opt - q_pert
            if gap > 0:  # only count actions that are worse
                min_gap = min(min_gap, gap)

        if min_gap == float('inf'):
            min_gap = 0.0  # all perturbations were equally good → zero gap
        gaps.append(min_gap)

    return np.array(gaps)


def compute_oc_curve(src_model, env_id, states):
    """Compute OC at each target gravity."""
    src_actions = []
    for s in states:
        a, _ = src_model.predict(s, deterministic=True)
        src_actions.append(a)
    src_actions = np.array(src_actions)

    oc_values = []
    for g in GRAVITIES:
        tgt_model, tgt_env = load_model(env_id, g)
        tgt_env.close()

        tgt_actions = []
        for s in states:
            a, _ = tgt_model.predict(s, deterministic=True)
            tgt_actions.append(a)
        tgt_actions = np.array(tgt_actions)

        # Directional OC
        s1, s2 = np.sign(src_actions), np.sign(tgt_actions)
        mask = (np.abs(src_actions) > 0.05) & (np.abs(tgt_actions) > 0.05)
        if mask.sum() == 0:
            oc = 1.0
        else:
            oc = np.mean(s1[mask] == s2[mask])
        oc_values.append(oc)

    return oc_values


def run_experiment():
    save_dir = '../figures'
    os.makedirs(save_dir, exist_ok=True)

    results = {}

    for env_id, cfg in ENV_CONFIGS.items():
        print(f"\n{'='*50}")
        print(f"  {cfg['label']}: estimating F_kappa")
        print(f"{'='*50}")

        # Check if model exists
        cache_path = os.path.join(CACHE_DIR, f'sac_{env_id}_g{SRC_GRAVITY:.2f}_s{SEED}_t500000.zip')
        if not os.path.exists(cache_path):
            print(f"  [SKIP] No model at {cache_path}")
            continue

        src_model, src_env = load_model(env_id, SRC_GRAVITY)
        src_env.close()

        # Collect states
        print(f"  Collecting {N_SAMPLE_STATES} states...")
        states = collect_states(src_model, env_id)
        print(f"  Got {len(states)} states")

        # Estimate action gaps
        print(f"  Estimating action gaps ({N_ACTION_PERTURBATIONS} perturbations per state)...")
        gaps = estimate_action_gaps(src_model, states, env_id)
        print(f"  Gap stats: mean={np.mean(gaps):.4f}, median={np.median(gaps):.4f}, "
              f"std={np.std(gaps):.4f}, min={np.min(gaps):.4f}, max={np.max(gaps):.4f}")

        # Compute OC curve
        print(f"  Computing OC curve...")
        oc_values = compute_oc_curve(src_model, env_id, states)
        for g, oc in zip(GRAVITIES, oc_values):
            print(f"    g={g:.2f}: OC={oc:.3f}")

        # Steepness metric: fraction of states with gap below median/4
        median_gap = np.median(gaps)
        steepness = np.mean(gaps <= median_gap / 4) if median_gap > 0 else 1.0

        # OC drop: average OC drop from source
        oc_drop = 1.0 - np.mean([oc for g, oc in zip(GRAVITIES, oc_values) if g != SRC_GRAVITY])

        results[env_id] = {
            'label': cfg['label'],
            'color': cfg['color'],
            'gaps': gaps,
            'oc_values': oc_values,
            'steepness': steepness,
            'oc_drop': oc_drop,
            'median_gap': median_gap,
        }
        print(f"  Steepness (F_κ at median/4) = {steepness:.3f}")
        print(f"  Mean OC drop = {oc_drop:.3f}")

    if len(results) == 0:
        print("No models found!")
        return

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Panel (a): F_kappa CDFs on absolute scale (log-x to show Hopper's small gaps)
    ax = axes[0]
    for env_id, r in results.items():
        sorted_gaps = np.sort(r['gaps'])
        sorted_gaps = sorted_gaps[sorted_gaps > 0]  # exclude exact zeros for log
        cdf = np.arange(1, len(sorted_gaps)+1) / len(r['gaps'])
        ax.plot(sorted_gaps, cdf, lw=2.5, color=r['color'],
                label=f"{r['label']} (med={r['median_gap']:.3f})")
    ax.set_xscale('log')
    ax.set_xlabel('Action gap $\\kappa(s)$ (log scale)')
    ax.set_ylabel('$F_\\kappa(x)$')
    ax.set_title('(a) Empirical $F_\\kappa$ from SAC')
    ax.legend(fontsize=9)
    ax.set_ylim(-0.02, 1.05)

    # Panel (b): OC decay curves
    ax = axes[1]
    delta_gs = [abs(g - SRC_GRAVITY) for g in GRAVITIES]
    for env_id, r in results.items():
        ax.plot(delta_gs, r['oc_values'], 'o-', lw=2.5, color=r['color'],
                label=r['label'], markersize=8)
    ax.set_xlabel('$|\\Delta g|$ (m/s²)')
    ax.set_ylabel('Directional OC')
    ax.set_title('(b) OC Decay')
    ax.legend(fontsize=9)
    ax.set_ylim(0.45, 1.05)

    # Panel (c): Median gap vs OC drop — the prediction
    ax = axes[2]
    for env_id, r in results.items():
        ax.scatter(r['median_gap'], r['oc_drop'], s=200, color=r['color'],
                   edgecolors='black', lw=1.5, zorder=5)
        ax.annotate(r['label'], (r['median_gap'], r['oc_drop']),
                    textcoords="offset points", xytext=(10, 5), fontsize=12)

    # Fit line if 3 points
    if len(results) >= 3:
        xs = [r['median_gap'] for r in results.values()]
        ys = [r['oc_drop'] for r in results.values()]
        if np.std(xs) > 1e-8:
            c = np.polyfit(xs, ys, 1)
            xf = np.linspace(0, max(xs)*1.2, 50)
            ax.plot(xf, np.polyval(c, xf), 'k--', lw=1.5, alpha=0.5)
            corr = np.corrcoef(xs, ys)[0, 1]
            ax.text(0.95, 0.95, f'r = {corr:.2f}', transform=ax.transAxes,
                    fontsize=12, va='top', ha='right')

    ax.set_xlabel('Median action gap $\\tilde{\\kappa}$')
    ax.set_ylabel('Mean OC drop')
    ax.set_title('(c) Larger Gaps $\\Rightarrow$ More Robust Transfer')

    plt.suptitle('Neural $F_\\kappa$: Action-Gap Distribution Predicts Transfer Robustness',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'exp14_fkappa_neural.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'exp14_fkappa_neural.png'), bbox_inches='tight')
    plt.close()

    # Save results
    with open(os.path.join(save_dir, 'exp14_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    print(f"\n{'='*50}")
    print("  Summary")
    print(f"{'='*50}")
    print(f"  {'Env':<15} {'Steepness':>10} {'OC Drop':>10} {'Median Gap':>12}")
    for env_id, r in results.items():
        print(f"  {r['label']:<15} {r['steepness']:>10.3f} {r['oc_drop']:>10.3f} {r['median_gap']:>12.4f}")
    print(f"\n  Prediction: steeper F_κ → larger OC drop")
    print(f"  Saved to {save_dir}/exp14_fkappa_neural.*")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_experiment()
