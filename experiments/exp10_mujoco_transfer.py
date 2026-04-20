"""
Experiment 10: MuJoCo Transfer (Hopper-v4 + HalfCheetah-v4)
============================================================
High-dimensional continuous control with physics parameter sweeps.
Uses Stable-Baselines3 SAC on actual MuJoCo environments.

Usage:
  python exp10_mujoco_transfer.py hopper
  python exp10_mujoco_transfer.py halfcheetah
  python exp10_mujoco_transfer.py plot        # plot from cached results
"""

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import SAC
import os
import sys
import time
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

SEED = 42
CACHE_DIR = '../cache_exp10'

# ── Environment configs ──
ENV_CONFIGS = {
    'hopper': {
        'env_id': 'Hopper-v4',
        'gravities': [5.0, 7.0, 9.81, 12.0, 15.0],
        'src_gravity': 9.81,
        'n_timesteps': 500_000,
        'eval_gravities_n': 11,
        'n_eval_episodes': 20,
        'label': 'Hopper-v4',
        'obs_dim': 11,
        'act_dim': 3,
    },
    'halfcheetah': {
        'env_id': 'HalfCheetah-v4',
        'gravities': [5.0, 7.0, 9.81, 12.0, 15.0],
        'src_gravity': 9.81,
        'n_timesteps': 500_000,
        'eval_gravities_n': 11,
        'n_eval_episodes': 20,
        'label': 'HalfCheetah-v4',
        'obs_dim': 17,
        'act_dim': 6,
    },
    'ant': {
        'env_id': 'Ant-v4',
        'gravities': [5.0, 7.0, 9.81, 12.0, 15.0],
        'src_gravity': 9.81,
        'n_timesteps': 500_000,
        'eval_gravities_n': 11,
        'n_eval_episodes': 20,
        'label': 'Ant-v4',
        'obs_dim': 27,
        'act_dim': 8,
    },
    'walker2d': {
        'env_id': 'Walker2d-v4',
        'gravities': [5.0, 7.0, 9.81, 12.0, 15.0],
        'src_gravity': 9.81,
        'n_timesteps': 500_000,
        'eval_gravities_n': 11,
        'n_eval_episodes': 20,
        'label': 'Walker2d-v4',
        'obs_dim': 17,
        'act_dim': 6,
    },
    'swimmer': {
        'env_id': 'Swimmer-v4',
        'gravities': [5.0, 7.0, 9.81, 12.0, 15.0],
        'src_gravity': 9.81,
        'n_timesteps': 500_000,
        'eval_gravities_n': 11,
        'n_eval_episodes': 20,
        'label': 'Swimmer-v4',
        'obs_dim': 8,
        'act_dim': 2,
    },
}


def make_env(env_id, gravity=9.81):
    env = gym.make(env_id)
    env.unwrapped.model.opt.gravity[:] = [0, 0, -gravity]
    return env


def train_agent(env_id, gravity, n_timesteps, seed=SEED):
    """Train SAC agent. Cache to disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f'sac_{env_id}_g{gravity:.2f}_s{seed}_t{n_timesteps}.zip')

    if os.path.exists(cache_path):
        print(f'    [cached] {cache_path}')
        env = make_env(env_id, gravity)
        model = SAC.load(cache_path, env=env, device='cuda')
        return model, env

    print(f'    Training SAC {env_id} g={gravity:.2f} ({n_timesteps} steps)...')
    env = make_env(env_id, gravity)
    model = SAC(
        'MlpPolicy', env, verbose=0, device='cuda',
        learning_rate=3e-4, batch_size=256, buffer_size=300_000,
        learning_starts=10000, tau=0.005, gamma=0.99,
        seed=seed
    )
    t0 = time.time()
    model.learn(total_timesteps=n_timesteps)
    elapsed = time.time() - t0
    print(f'    Done in {elapsed:.0f}s ({elapsed/60:.1f}min)')
    model.save(cache_path)
    return model, env


def train_dr_agent(env_id, gravity_range, n_timesteps, seed=SEED):
    """Train SAC with domain-randomized gravity."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag = f'{gravity_range[0]}-{gravity_range[1]}'
    cache_path = os.path.join(CACHE_DIR, f'sac_{env_id}_DR_{tag}_s{seed}_t{n_timesteps}.zip')

    if os.path.exists(cache_path):
        print(f'    [cached DR] {cache_path}')
        env = make_env(env_id, 9.81)
        model = SAC.load(cache_path, env=env, device='cuda')
        return model

    print(f'    Training DR {env_id} g~U[{gravity_range[0]}, {gravity_range[1]}]...')
    env = make_env(env_id, 9.81)
    rng = np.random.RandomState(seed)
    original_reset = env.reset

    def randomized_reset(**kwargs):
        result = original_reset(**kwargs)
        g = rng.uniform(gravity_range[0], gravity_range[1])
        env.unwrapped.model.opt.gravity[:] = [0, 0, -g]
        return result

    env.reset = randomized_reset

    model = SAC(
        'MlpPolicy', env, verbose=0, device='cuda',
        learning_rate=3e-4, batch_size=256, buffer_size=300_000,
        learning_starts=10000, seed=seed
    )
    t0 = time.time()
    model.learn(total_timesteps=n_timesteps)
    elapsed = time.time() - t0
    print(f'    DR done in {elapsed:.0f}s ({elapsed/60:.1f}min)')
    model.save(cache_path)
    return model


def evaluate_at_gravity(model, env_id, gravity, n_episodes=20, seed_base=5000):
    env = make_env(env_id, gravity)
    rewards = []
    for ep in range(n_episodes):
        s, _ = env.reset(seed=seed_base + ep)
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


def get_actions(model, states):
    actions = []
    for s in states:
        a, _ = model.predict(s, deterministic=True)
        actions.append(a)
    return np.array(actions)


def directional_oc(actions1, actions2):
    """Fraction of (state, dim) pairs where action signs agree."""
    signs1 = np.sign(actions1)
    signs2 = np.sign(actions2)
    mask = (np.abs(actions1) > 0.05) & (np.abs(actions2) > 0.05)
    if mask.sum() == 0:
        return 1.0
    return np.mean(signs1[mask] == signs2[mask])


def run_single_env(env_name):
    cfg = ENV_CONFIGS[env_name]
    env_id = cfg['env_id']
    gravities = cfg['gravities']
    src_g = cfg['src_gravity']
    n_ts = cfg['n_timesteps']
    n_eval = cfg['n_eval_episodes']

    print(f"\n{'='*60}")
    print(f"  {cfg['label']} ({cfg['obs_dim']}D obs, {cfg['act_dim']}D act)")
    print(f"{'='*60}")

    # ── Train agents ──
    print("\n[1/7] Training agents at each gravity...")
    models = {}
    for g in gravities:
        model, env = train_agent(env_id, g, n_ts, seed=SEED)
        models[g] = model
        env.close()

    # ── Train DR baseline ──
    print("\n[2/7] Training DR baseline...")
    dr_model = train_dr_agent(env_id, (min(gravities), max(gravities)), n_ts, seed=SEED)

    # ── Test states ──
    print("\n[3/7] Collecting test states...")
    env = make_env(env_id, src_g)
    test_states = []
    s, _ = env.reset(seed=SEED + 999)
    for t in range(5000):
        a, _ = models[src_g].predict(s, deterministic=True)
        s, r, term, trunc, _ = env.step(a)
        if t >= 500 and t % 10 == 0:
            test_states.append(s.copy())
        if term or trunc:
            s, _ = env.reset(seed=SEED + 999 + t)
    test_states = np.array(test_states[:300])
    env.close()
    print(f"  Collected {len(test_states)} test states")

    # ── Transfer returns ──
    print("\n[4/7] Evaluating transfer returns...")
    eval_gravities = np.linspace(min(gravities), max(gravities), cfg['eval_gravities_n'])
    src_returns, opt_returns, dr_returns = [], [], []
    for g in eval_gravities:
        sr, _ = evaluate_at_gravity(models[src_g], env_id, g, n_eval)
        closest_g = min(gravities, key=lambda x: abs(x - g))
        orr, _ = evaluate_at_gravity(models[closest_g], env_id, g, n_eval)
        drr, _ = evaluate_at_gravity(dr_model, env_id, g, n_eval)
        src_returns.append(sr)
        opt_returns.append(orr)
        dr_returns.append(drr)
        print(f"  g={g:5.2f}: src={sr:7.0f}, target-trained={orr:7.0f}, DR={drr:7.0f}")

    # ── OC ──
    print("\n[5/7] Measuring ordinal consistency...")
    src_actions = get_actions(models[src_g], test_states)
    oc_values = []
    for g in gravities:
        tgt_actions = get_actions(models[g], test_states)
        doc = directional_oc(src_actions, tgt_actions)
        oc_values.append(doc)
        print(f"  OC(g={g:.2f}): {doc:.3f}")

    # ── Scale-invariance ──
    print("\n[6/7] Scale-invariance test...")
    all_actions = [get_actions(m, test_states) for m in models.values()]

    base_signs = np.sign(np.mean([np.sign(a) for a in all_actions], axis=0))
    base_mean = np.mean(all_actions, axis=0)

    scale_ranges = [1, 2, 5, 10, 20, 50, 100]
    mv_inv, qa_inv = [], []
    n_trials = 20

    for sr in scale_ranges:
        mv_t, qa_t = [], []
        for trial in range(n_trials):
            np.random.seed(SEED + trial * 77 + int(sr * 10))
            log_s = np.random.uniform(-np.log(max(sr, 1.01)), np.log(max(sr, 1.01)), len(all_actions))
            scales = np.exp(log_s)

            scaled = [a * s for a, s in zip(all_actions, scales)]
            scaled_signs = np.sign(np.mean([np.sign(a) for a in scaled], axis=0))
            mv_t.append(np.mean(scaled_signs == base_signs))

            scaled_mean = np.mean(scaled, axis=0)
            cos_sims = []
            for i in range(len(test_states)):
                n1, n2 = np.linalg.norm(base_mean[i]), np.linalg.norm(scaled_mean[i])
                if n1 > 1e-6 and n2 > 1e-6:
                    cos_sims.append(np.dot(base_mean[i], scaled_mean[i]) / (n1 * n2))
            qa_t.append(np.mean(cos_sims) if cos_sims else 1.0)

        mv_inv.append(np.mean(mv_t))
        qa_inv.append(np.mean(qa_t))
        print(f"  Scale {sr:3d}x: MV={mv_inv[-1]:.3f}, Mean-Act={qa_inv[-1]:.3f}")

    # ── Action displacement ──
    print("\n[7/7] Action displacement...")
    displacements = []
    for g in gravities:
        tgt_actions = get_actions(models[g], test_states)
        disp = np.mean(np.linalg.norm(src_actions - tgt_actions, axis=1))
        displacements.append(disp)
        print(f"  g={g:.2f}: disp={disp:.3f}")

    results = {
        'env_name': env_name,
        'label': cfg['label'],
        'gravities': gravities,
        'src_gravity': src_g,
        'eval_gravities': eval_gravities.tolist(),
        'src_returns': src_returns,
        'opt_returns': opt_returns,
        'dr_returns': dr_returns,
        'oc_values': oc_values,
        'scale_ranges': scale_ranges,
        'mv_invariance': mv_inv,
        'qa_invariance': qa_inv,
        'displacements': displacements,
    }

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, f'results_{env_name}.pkl'), 'wb') as f:
        pickle.dump(results, f)

    return results


ALL_ENVS = ['halfcheetah', 'ant', 'walker2d', 'swimmer', 'hopper']  # order: expected good → bad OC


def plot_combined(save_dir='../figures'):
    """Plot 5-env figure: 5 rows × 4 cols."""
    os.makedirs(save_dir, exist_ok=True)

    available = []
    results = {}
    for name in ALL_ENVS:
        path = os.path.join(CACHE_DIR, f'results_{name}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                results[name] = pickle.load(f)
            available.append(name)

    if not available:
        print("No results found. Run environments first.")
        return

    n_envs = len(available)
    fig, axes = plt.subplots(n_envs, 4, figsize=(20, 4.2 * n_envs))
    if n_envs == 1:
        axes = axes[np.newaxis, :]

    panels = 'abcdefghijklmnopqrst'

    for row, name in enumerate(available):
        r = results[name]
        eg = r['eval_gravities']
        sg = r['src_gravity']
        gs = r['gravities']
        label = r['label']
        pi = row * 4  # panel index

        # Col 0: Transfer return + DR
        ax = axes[row, 0]
        ax.plot(eg, r['opt_returns'], 'go-', ms=4, lw=2, label='Target-trained')
        ax.plot(eg, r['src_returns'], 'b^--', ms=4, lw=2, label=f'Source ($g={sg}$)')
        ax.plot(eg, r['dr_returns'], 'rs-.', ms=4, lw=1.5, label='Domain Rand.')
        ax.fill_between(eg, r['src_returns'], r['opt_returns'], alpha=0.1, color='red')
        ax.axvline(sg, color='gray', ls='--', alpha=0.5)
        ax.set_xlabel('Gravity (m/s$^2$)')
        ax.set_ylabel('Episode return')
        ax.set_title(f'({panels[pi]}) {label}: Transfer Return')
        ax.legend(fontsize=7, loc='best')

        # Col 1: OC decay
        ax = axes[row, 1]
        dg = [abs(g - sg) for g in gs]
        oc = r['oc_values']
        ax.scatter(dg, oc, c='steelblue', s=60, zorder=3, edgecolors='black', lw=0.5)
        if len(dg) >= 3:
            coeffs = np.polyfit(dg, oc, 2)
            xfit = np.linspace(0, max(dg), 50)
            ax.plot(xfit, np.clip(np.polyval(coeffs, xfit), 0.35, 1.05), 'b-', lw=2.5, alpha=0.7, label='Trend')
        ax.axhline(0.5, color='red', ls=':', alpha=0.4, label='Random')
        ax.set_xlabel(r'$|\Delta g|$ from source')
        ax.set_ylabel('Directional OC')
        ax.set_title(f'({panels[pi+1]}) {label}: OC Decay')
        ax.set_ylim(0.35, 1.05)
        ax.legend(fontsize=9)

        # Col 2: Scale-invariance
        ax = axes[row, 2]
        ax.semilogx(r['scale_ranges'], r['mv_invariance'], 'bo-', ms=6, lw=2.5,
                     label='Ordinal (sign vote)')
        ax.semilogx(r['scale_ranges'], r['qa_invariance'], 's--', color='purple',
                     ms=6, lw=2, label='Mean-action')
        ax.axhline(1.0, color='green', ls='--', alpha=0.5)
        ax.set_xlabel('Scale heterogeneity ($r$)')
        ax.set_ylabel('Agreement with unscaled')
        ax.set_title(f'({panels[pi+2]}) {label}: Scale-Invariance')
        ax.set_ylim(0.5, 1.05)
        ax.legend(fontsize=9)

        # Col 3: Transfer Gap (log-log) — validates Theorem 4
        ax = axes[row, 3]
        # Transfer gap = opt_return - src_return at each eval gravity
        transfer_gaps = [max(o - s, 1.0) for o, s in zip(r['opt_returns'], r['src_returns'])]
        dg_eval = [abs(g - sg) for g in eg]
        # Filter positive |Δg|
        pos_mask = [i for i, d in enumerate(dg_eval) if d > 0.3]
        if pos_mask:
            dg_pos = [dg_eval[i] for i in pos_mask]
            gap_pos = [transfer_gaps[i] for i in pos_mask]
            ax.loglog(dg_pos, gap_pos, 'b^-', ms=6, lw=2, label='Transfer gap')
            # Fit log-log slope
            if len(dg_pos) >= 3:
                log_dg = np.log(dg_pos)
                log_gap = np.log(gap_pos)
                coeffs = np.polyfit(log_dg, log_gap, 1)
                slope = coeffs[0]
                xfit = np.logspace(np.log10(min(dg_pos) * 0.8),
                                   np.log10(max(dg_pos) * 1.2), 20)
                ax.loglog(xfit, np.exp(np.polyval(coeffs, np.log(xfit))),
                          'r--', lw=2, label=f'Slope $\\approx {slope:.2f}$')
            # DR gap for comparison
            dr_gaps = [max(o - d, 1.0) for o, d in zip(r['opt_returns'], r['dr_returns'])]
            dr_pos = [dr_gaps[i] for i in pos_mask]
            ax.loglog(dg_pos, dr_pos, 'gs--', ms=5, lw=1.5, alpha=0.7, label='DR gap')
            ax.legend(fontsize=8)
        ax.set_xlabel(r'$|\Delta g|$ from source')
        ax.set_ylabel('Transfer gap')
        ax.set_title(f'({panels[pi+3]}) {label}: Gap Scaling')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'exp10_mujoco_transfer.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'exp10_mujoco_transfer.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved exp10 figure ({n_envs} envs) to {save_dir}")

    # Also print summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY: OC and Scale-Invariance across {n_envs} MuJoCo environments")
    print(f"{'='*70}")
    print(f"  {'Env':<20} {'Obs':>4} {'Act':>4} {'OC(near)':>9} {'OC(far)':>8} {'MV@100x':>8} {'MA@100x':>8}")
    print(f"  {'-'*64}")
    for name in available:
        r = results[name]
        cfg = ENV_CONFIGS[name]
        oc_arr = r['oc_values']
        # near = smallest |Δg| > 0, far = largest |Δg|
        dgs = [abs(g - r['src_gravity']) for g in r['gravities']]
        near_idx = sorted(range(len(dgs)), key=lambda i: dgs[i])[1]  # skip self
        far_idx = sorted(range(len(dgs)), key=lambda i: -dgs[i])[0]
        mv100 = r['mv_invariance'][-1] if r['mv_invariance'] else 0
        qa100 = r['qa_invariance'][-1] if r['qa_invariance'] else 0
        print(f"  {cfg['label']:<20} {cfg['obs_dim']:>4} {cfg['act_dim']:>4} "
              f"{oc_arr[near_idx]:>9.3f} {oc_arr[far_idx]:>8.3f} {mv100:>8.3f} {qa100:>8.3f}")


if __name__ == '__main__':
    usage = "Usage: python exp10_mujoco_transfer.py [hopper|halfcheetah|ant|walker2d|swimmer|all|plot]"

    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)

    arg = sys.argv[1].lower()

    if arg == 'plot':
        plot_combined()
    elif arg == 'all':
        for name in ALL_ENVS:
            run_single_env(name)
        plot_combined()
    elif arg in ENV_CONFIGS:
        run_single_env(arg)
        plot_combined()  # plot whatever is available so far
    else:
        print(f"Unknown argument: {arg}")
        print(usage)
        sys.exit(1)
