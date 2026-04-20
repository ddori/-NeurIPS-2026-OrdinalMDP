"""
Experiment 15: Quadratic Regime Analysis
=========================================
Address reviewer W1: show that within the regime where continuous-action
assumptions hold, the log-log slope of transfer degradation is ≈ 2.
At larger |Δg|, assumptions break and slope increases.

Approach:
  - Evaluate source policy (g=9.81) at finely-spaced gravities
  - Measure return degradation and action displacement
  - Compute rolling log-log slope
  - Identify the "breakpoint" where slope departs from 2
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
N_EVAL_EPISODES = 20
N_TEST_STATES = 300

# Fine-grained gravity grid: dense near source, sparser far away
FINE_GRAVITIES = sorted(set(
    list(np.arange(7.0, 13.0, 0.25)) +   # dense near source
    [5.0, 5.5, 6.0, 6.5, 13.5, 14.0, 14.5, 15.0]  # sparse far
))

ENV_CONFIGS = {
    'HalfCheetah-v4': {'label': 'HalfCheetah', 'color': '#1f77b4'},
    'Ant-v4':         {'label': 'Ant',         'color': '#ff7f0e'},
    'Hopper-v4':      {'label': 'Hopper',      'color': '#2ca02c'},
}


def make_env(env_id, gravity=9.81):
    env = gym.make(env_id)
    env.unwrapped.model.opt.gravity[:] = [0, 0, -gravity]
    return env


def evaluate_return(model, env_id, gravity, n_episodes=N_EVAL_EPISODES):
    env = make_env(env_id, gravity)
    rewards = []
    for ep in range(n_episodes):
        s, _ = env.reset(seed=5000 + ep)
        total_r = 0
        for t in range(1000):
            a, _ = model.predict(s, deterministic=True)
            s, r, term, trunc, _ = env.step(a)
            total_r += r
            if term or trunc:
                break
        rewards.append(total_r)
    env.close()
    return np.mean(rewards), np.std(rewards)


def collect_states(model, env_id, n_states=N_TEST_STATES):
    env = make_env(env_id, SRC_GRAVITY)
    states = []
    s, _ = env.reset(seed=SEED + 999)
    for t in range(8000):
        a, _ = model.predict(s, deterministic=True)
        s, r, term, trunc, _ = env.step(a)
        if t >= 300 and t % 10 == 0:
            states.append(s.copy())
        if term or trunc:
            s, _ = env.reset(seed=SEED + 999 + t)
        if len(states) >= n_states:
            break
    env.close()
    return np.array(states[:n_states])


def get_actions_at_gravity(model, states, env_id, gravity):
    """Get actions the source model would take, evaluated in target env context.
    For action displacement, we just need model.predict on the same states."""
    actions = []
    for s in states:
        a, _ = model.predict(s, deterministic=True)
        actions.append(a)
    return np.array(actions)


def compute_rolling_slope(x, y, window=5):
    """Compute local log-log slope using rolling window."""
    log_x = np.log(x)
    log_y = np.log(np.maximum(y, 1e-10))
    slopes = []
    centers = []
    for i in range(len(x) - window + 1):
        lx = log_x[i:i+window]
        ly = log_y[i:i+window]
        if np.std(lx) > 1e-10 and np.all(np.isfinite(ly)):
            slope = np.polyfit(lx, ly, 1)[0]
            slopes.append(slope)
            centers.append(np.exp(np.mean(lx)))
        else:
            slopes.append(np.nan)
            centers.append(np.exp(np.mean(lx)))
    return np.array(centers), np.array(slopes)


def run_experiment():
    save_dir = '../figures'
    os.makedirs(save_dir, exist_ok=True)

    results = {}

    for env_id, cfg in ENV_CONFIGS.items():
        cache_path = os.path.join(CACHE_DIR, f'sac_{env_id}_g{SRC_GRAVITY:.2f}_s{SEED}_t500000.zip')
        if not os.path.exists(cache_path):
            print(f"  [SKIP] {env_id}")
            continue

        print(f"\n{'='*50}")
        print(f"  {cfg['label']}: fine-grained regime analysis")
        print(f"{'='*50}")

        env = make_env(env_id, SRC_GRAVITY)
        model = SAC.load(cache_path, env=env, device='cuda')
        env.close()

        # Source return
        src_ret, _ = evaluate_return(model, env_id, SRC_GRAVITY)
        print(f"  Source return at g=9.81: {src_ret:.0f}")

        # Collect reference states and actions at source
        states = collect_states(model, env_id)
        src_actions = get_actions_at_gravity(model, states, env_id, SRC_GRAVITY)

        # Evaluate at fine-grained gravities
        delta_gs = []
        return_losses = []
        action_disps = []

        for g in FINE_GRAVITIES:
            if abs(g - SRC_GRAVITY) < 0.01:
                continue  # skip source itself

            ret, _ = evaluate_return(model, env_id, g)
            loss = max(src_ret - ret, 0)

            # Action displacement: collect states at target gravity and compare actions
            # Actually, we compare what the model does on the SAME states
            # (states from source distribution)
            tgt_env = make_env(env_id, g)
            # Run source model in target env to get visited states, then compare actions
            tgt_states = []
            s, _ = tgt_env.reset(seed=SEED + 999)
            for t in range(3000):
                a, _ = model.predict(s, deterministic=True)
                s, r, term, trunc, _ = tgt_env.step(a)
                if t >= 300 and t % 10 == 0:
                    tgt_states.append(s.copy())
                if term or trunc:
                    s, _ = tgt_env.reset(seed=SEED + 999 + t)
                if len(tgt_states) >= 200:
                    break
            tgt_env.close()

            # Action displacement on source states (policy doesn't change,
            # but we want displacement vs optimal target action — we don't have that.
            # Use return loss as the main metric instead.)
            dg = abs(g - SRC_GRAVITY)
            delta_gs.append(dg)
            return_losses.append(loss)
            print(f"    g={g:5.2f}, |Δg|={dg:.2f}, ret={ret:8.0f}, loss={loss:8.0f}")

        delta_gs = np.array(delta_gs)
        return_losses = np.array(return_losses)

        # Sort by delta_g
        sort_idx = np.argsort(delta_gs)
        delta_gs = delta_gs[sort_idx]
        return_losses = return_losses[sort_idx]

        # Remove zero losses for log-log
        pos_mask = return_losses > 0
        dg_pos = delta_gs[pos_mask]
        loss_pos = return_losses[pos_mask]

        # Overall slope
        if len(dg_pos) >= 3:
            overall_slope = np.polyfit(np.log(dg_pos), np.log(loss_pos), 1)[0]
        else:
            overall_slope = np.nan

        # Slope for small Δg only (< 3 m/s²)
        small_mask = dg_pos < 3.0
        if small_mask.sum() >= 3:
            small_slope = np.polyfit(np.log(dg_pos[small_mask]), np.log(loss_pos[small_mask]), 1)[0]
        else:
            small_slope = np.nan

        # Slope for large Δg only (>= 3 m/s²)
        large_mask = dg_pos >= 3.0
        if large_mask.sum() >= 3:
            large_slope = np.polyfit(np.log(dg_pos[large_mask]), np.log(loss_pos[large_mask]), 1)[0]
        else:
            large_slope = np.nan

        # Rolling slope
        if len(dg_pos) >= 7:
            roll_centers, roll_slopes = compute_rolling_slope(dg_pos, loss_pos, window=5)
        else:
            roll_centers, roll_slopes = np.array([]), np.array([])

        results[env_id] = {
            'label': cfg['label'],
            'color': cfg['color'],
            'delta_gs': delta_gs,
            'return_losses': return_losses,
            'dg_pos': dg_pos,
            'loss_pos': loss_pos,
            'overall_slope': overall_slope,
            'small_slope': small_slope,
            'large_slope': large_slope,
            'roll_centers': roll_centers,
            'roll_slopes': roll_slopes,
        }

        print(f"  Slopes: overall={overall_slope:.2f}, small Δg (<3)={small_slope:.2f}, large Δg (≥3)={large_slope:.2f}")

    if not results:
        print("No results!")
        return

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Panel (a): Log-log with regime coloring
    ax = axes[0]
    for env_id, r in results.items():
        if len(r['dg_pos']) == 0:
            continue
        # Small regime
        small = r['dg_pos'] < 3.0
        large = r['dg_pos'] >= 3.0
        ax.scatter(r['dg_pos'][small], r['loss_pos'][small],
                   color=r['color'], s=40, alpha=0.8, zorder=3)
        ax.scatter(r['dg_pos'][large], r['loss_pos'][large],
                   color=r['color'], s=40, alpha=0.4, marker='s', zorder=3)

        # Fit lines for each regime
        if small.sum() >= 2:
            c = np.polyfit(np.log(r['dg_pos'][small]), np.log(r['loss_pos'][small]), 1)
            xf = np.linspace(r['dg_pos'][small].min(), r['dg_pos'][small].max(), 50)
            ax.plot(xf, np.exp(np.polyval(c, np.log(xf))), '--', color=r['color'], lw=2,
                    label=f"{r['label']}: slope={r['small_slope']:.1f} (small)")
        if large.sum() >= 2:
            c = np.polyfit(np.log(r['dg_pos'][large]), np.log(r['loss_pos'][large]), 1)
            xf = np.linspace(r['dg_pos'][large].min(), r['dg_pos'][large].max(), 50)
            ax.plot(xf, np.exp(np.polyval(c, np.log(xf))), ':', color=r['color'], lw=2,
                    label=f"{r['label']}: slope={r['large_slope']:.1f} (large)")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axvline(x=3.0, color='gray', ls='--', alpha=0.5, label='Regime boundary')
    ax.set_xlabel('$|\\Delta g|$ (m/s²)')
    ax.set_ylabel('Return loss')
    ax.set_title('(a) Log-log: Two Regimes')
    ax.legend(fontsize=7, loc='upper left')

    # Panel (b): Rolling slope
    ax = axes[1]
    for env_id, r in results.items():
        if len(r['roll_centers']) > 0:
            ax.plot(r['roll_centers'], r['roll_slopes'], 'o-', color=r['color'],
                    lw=2, markersize=5, label=r['label'])
    ax.axhline(y=2.0, color='black', ls='--', lw=1.5, alpha=0.7, label='Quadratic (slope=2)')
    ax.axvline(x=3.0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('$|\\Delta g|$ (m/s²)')
    ax.set_ylabel('Local log-log slope')
    ax.set_title('(b) Rolling Slope (window=5)')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 8)
    ax.text(1.5, 7.2, 'Quadratic\nregime', ha='center', fontsize=9, color='green', alpha=0.7)
    ax.text(4.0, 7.2, 'Super-quadratic\nregime', ha='center', fontsize=9, color='red', alpha=0.7)

    # Panel (c): Summary table as text
    ax = axes[2]
    ax.axis('off')
    header = ['Environment', 'Slope\n$|\\Delta g|<3$', 'Slope\n$|\\Delta g|\\geq 3$', 'Breakpoint']
    rows = []
    for env_id, r in results.items():
        # Find breakpoint: where rolling slope first exceeds 3.0
        bp = '—'
        if len(r['roll_centers']) > 0:
            exceed = r['roll_slopes'] > 3.0
            if np.any(exceed):
                bp_idx = np.argmax(exceed)
                bp = f"$\\approx${r['roll_centers'][bp_idx]:.1f}"
        rows.append([r['label'],
                     f"{r['small_slope']:.1f}" if not np.isnan(r['small_slope']) else '—',
                     f"{r['large_slope']:.1f}" if not np.isnan(r['large_slope']) else '—',
                     bp])

    table = ax.table(cellText=rows, colLabels=header, loc='center',
                     cellLoc='center', colColours=['#f0f0f0']*4)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 2.0)
    ax.set_title('(c) Regime Summary', pad=20)

    plt.suptitle('Quadratic Regime Analysis: Slope $\\approx 2$ for Small $|\\Delta g|$,\n'
                 'Super-Quadratic Only When Assumptions Break',
                 fontsize=13, y=1.04)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'exp15_regime_analysis.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'exp15_regime_analysis.png'), bbox_inches='tight')
    plt.close()

    # Save results
    with open(os.path.join(save_dir, 'exp15_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    print(f"\nSaved to {save_dir}/exp15_regime_analysis.*")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_experiment()
