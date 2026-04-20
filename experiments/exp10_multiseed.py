"""
Experiment 10b: Multi-seed MuJoCo Transfer
==========================================
Train SAC at 5 gravities with 3 seeds (42, 123, 7) for multiple envs.

Usage:
  python exp10_multiseed.py train halfcheetah
  python exp10_multiseed.py train ant
  python exp10_multiseed.py train hopper
  python exp10_multiseed.py eval halfcheetah
  python exp10_multiseed.py eval ant
  python exp10_multiseed.py eval hopper
  python exp10_multiseed.py plot
"""

import numpy as np
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

CACHE_DIR = '../cache_exp10'
SEEDS = [42, 123, 7, 2024, 31]
GRAVITIES = [5.0, 7.0, 9.81, 12.0, 15.0]
SRC_GRAVITY = 9.81
N_TIMESTEPS = 500_000
N_EVAL_EPISODES = 20
N_TEST_STATES = 300

ENV_CONFIGS = {
    'halfcheetah': 'HalfCheetah-v4',
    'ant': 'Ant-v4',
    'hopper': 'Hopper-v4',
}


def make_env(env_id, gravity=9.81):
    env = gym.make(env_id)
    env.unwrapped.model.opt.gravity[:] = [0, 0, -gravity]
    return env


def train_agent(env_id, gravity, seed):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f'sac_{env_id}_g{gravity:.2f}_s{seed}_t{N_TIMESTEPS}.zip')

    if os.path.exists(cache_path):
        print(f'  [cached] {env_id} seed={seed}, g={gravity:.2f}')
        return cache_path

    print(f'  Training {env_id} seed={seed}, g={gravity:.2f} ...')
    env = make_env(env_id, gravity)
    model = SAC(
        'MlpPolicy', env, verbose=0, device='cuda',
        learning_rate=3e-4, batch_size=256, buffer_size=300_000,
        learning_starts=10000, tau=0.005, gamma=0.99,
        seed=seed
    )
    t0 = time.time()
    model.learn(total_timesteps=N_TIMESTEPS)
    elapsed = time.time() - t0
    print(f'    Done in {elapsed/60:.1f} min')
    model.save(cache_path)
    env.close()
    return cache_path


def load_model(env_id, gravity, seed):
    cache_path = os.path.join(CACHE_DIR, f'sac_{env_id}_g{gravity:.2f}_s{seed}_t{N_TIMESTEPS}.zip')
    env = make_env(env_id, gravity)
    model = SAC.load(cache_path, env=env, device='cuda')
    return model, env


def evaluate_at_gravity(model, env_id, gravity, n_episodes=N_EVAL_EPISODES, seed_base=5000):
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


def directional_oc(a1, a2):
    s1, s2 = np.sign(a1), np.sign(a2)
    mask = (np.abs(a1) > 0.05) & (np.abs(a2) > 0.05)
    if mask.sum() == 0:
        return 1.0
    return np.mean(s1[mask] == s2[mask])


def run_training(env_name):
    env_id = ENV_CONFIGS[env_name]
    print(f"\n{'='*60}")
    print(f"  Multi-seed Training: {env_id}")
    print(f"  Seeds: {SEEDS}, Gravities: {GRAVITIES}")
    print(f"{'='*60}")
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        for g in GRAVITIES:
            train_agent(env_id, g, seed)


def run_eval(env_name):
    env_id = ENV_CONFIGS[env_name]
    print(f"\n{'='*60}")
    print(f"  Multi-seed Evaluation: {env_id}")
    print(f"{'='*60}")

    all_results = {}

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        src_model, src_env = load_model(env_id, SRC_GRAVITY, seed)
        src_env.close()

        # Collect test states
        env = make_env(env_id, SRC_GRAVITY)
        test_states = []
        s, _ = env.reset(seed=seed + 999)
        for t in range(5000):
            a, _ = src_model.predict(s, deterministic=True)
            s, r, term, trunc, _ = env.step(a)
            if t >= 500 and t % 10 == 0:
                test_states.append(s.copy())
            if term or trunc:
                s, _ = env.reset(seed=seed + 999 + t)
        test_states = np.array(test_states[:N_TEST_STATES])
        env.close()

        src_actions = get_actions(src_model, test_states)
        seed_results = {'returns': {}, 'oc': {}, 'disp': {}}

        for g in GRAVITIES:
            sr, _ = evaluate_at_gravity(src_model, env_id, g)
            tgt_model, tgt_env = load_model(env_id, g, seed)
            tgt_env.close()
            tr, _ = evaluate_at_gravity(tgt_model, env_id, g)
            gap = max(tr - sr, 0)

            tgt_actions = get_actions(tgt_model, test_states)
            oc = directional_oc(src_actions, tgt_actions)
            disp = np.mean(np.linalg.norm(src_actions - tgt_actions, axis=1))

            seed_results['returns'][g] = {'src': sr, 'tgt': tr, 'gap': gap}
            seed_results['oc'][g] = oc
            seed_results['disp'][g] = disp

            print(f"  g={g:5.2f}: src={sr:7.0f}, tgt={tr:7.0f}, gap={gap:5.0f}, OC={oc:.3f}, disp={disp:.3f}")

        all_results[seed] = seed_results

    # Aggregate
    print(f"\n{'='*60}")
    print(f"  Aggregated: {env_id} (mean ± std, {len(SEEDS)} seeds)")
    print(f"{'='*60}")
    print(f"  {'g':>6}  {'Src Return':>14}  {'Tgt Return':>14}  {'Gap':>12}  {'OC':>12}  {'Disp':>12}")

    agg = {}
    for g in GRAVITIES:
        src_rets = [all_results[s]['returns'][g]['src'] for s in SEEDS]
        tgt_rets = [all_results[s]['returns'][g]['tgt'] for s in SEEDS]
        gaps = [all_results[s]['returns'][g]['gap'] for s in SEEDS]
        ocs = [all_results[s]['oc'][g] for s in SEEDS]
        disps = [all_results[s]['disp'][g] for s in SEEDS]

        agg[g] = {
            'src_mean': np.mean(src_rets), 'src_std': np.std(src_rets),
            'tgt_mean': np.mean(tgt_rets), 'tgt_std': np.std(tgt_rets),
            'gap_mean': np.mean(gaps), 'gap_std': np.std(gaps),
            'oc_mean': np.mean(ocs), 'oc_std': np.std(ocs),
            'disp_mean': np.mean(disps), 'disp_std': np.std(disps),
        }
        a = agg[g]
        print(f"  {g:6.2f}  {a['src_mean']:6.0f}±{a['src_std']:5.0f}  "
              f"{a['tgt_mean']:6.0f}±{a['tgt_std']:5.0f}  "
              f"{a['gap_mean']:5.0f}±{a['gap_std']:4.0f}  "
              f"{a['oc_mean']:.3f}±{a['oc_std']:.3f}  "
              f"{a['disp_mean']:.3f}±{a['disp_std']:.3f}")

    save_data = {'env_name': env_name, 'env_id': env_id, 'seeds': SEEDS,
                 'per_seed': all_results, 'aggregated': agg}
    pkl_path = os.path.join(CACHE_DIR, f'results_multiseed_{env_name}.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nSaved to {pkl_path}")
    return save_data


def plot_results(save_dir='../figures'):
    os.makedirs(save_dir, exist_ok=True)

    # Load all available envs
    all_data = {}
    for env_name in ENV_CONFIGS:
        pkl_path = os.path.join(CACHE_DIR, f'results_multiseed_{env_name}.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                all_data[env_name] = pickle.load(f)
    # Also check old format (halfcheetah only)
    old_path = os.path.join(CACHE_DIR, 'results_multiseed.pkl')
    if 'halfcheetah' not in all_data and os.path.exists(old_path):
        with open(old_path, 'rb') as f:
            d = pickle.load(f)
            d['env_name'] = 'halfcheetah'
            d['env_id'] = 'HalfCheetah-v4'
            all_data['halfcheetah'] = d

    if not all_data:
        print("No results. Run 'eval' first.")
        return

    n_envs = len(all_data)
    fig, axes = plt.subplots(n_envs, 4, figsize=(20, 4.5 * n_envs))
    if n_envs == 1:
        axes = axes[np.newaxis, :]

    for row, env_name in enumerate(sorted(all_data.keys())):
        data = all_data[env_name]
        agg = data['aggregated']
        env_id = data.get('env_id', ENV_CONFIGS.get(env_name, env_name))
        gs = sorted(agg.keys())
        dgs = [abs(g - SRC_GRAVITY) for g in gs]

        # (a) Transfer return
        ax = axes[row, 0]
        src_m = [agg[g]['src_mean'] for g in gs]
        src_s = [agg[g]['src_std'] for g in gs]
        tgt_m = [agg[g]['tgt_mean'] for g in gs]
        tgt_s = [agg[g]['tgt_std'] for g in gs]
        ax.errorbar(gs, src_m, yerr=src_s, fmt='b^--', ms=6, lw=2, capsize=4, label='Source $\\pi_{\\theta_0}$')
        ax.errorbar(gs, tgt_m, yerr=tgt_s, fmt='go-', ms=6, lw=2, capsize=4, label='Target $\\pi^*_{\\theta}$')
        ax.axvline(SRC_GRAVITY, color='gray', ls='--', alpha=0.5)
        ax.set_xlabel('Gravity (m/s$^2$)')
        ax.set_ylabel('Episode Return')
        ax.set_title(f'{env_id}: Transfer (3 seeds)')
        ax.legend(fontsize=8)

        # (b) OC
        ax = axes[row, 1]
        oc_m = [agg[g]['oc_mean'] for g in gs]
        oc_s = [agg[g]['oc_std'] for g in gs]
        ax.errorbar(dgs, oc_m, yerr=oc_s, fmt='o', color='steelblue', ms=8, capsize=4,
                    ecolor='black', elinewidth=1, zorder=3)
        ax.axhline(0.5, color='red', ls=':', alpha=0.4, label='Random')
        ax.set_xlabel(r'$|\Delta g|$ from source')
        ax.set_ylabel('Directional OC')
        ax.set_title(f'{env_id}: OC (3 seeds)')
        ax.set_ylim(0.35, 1.05)
        ax.legend()

        # (c) Transfer gap (log-log)
        ax = axes[row, 2]
        pos = [i for i, d in enumerate(dgs) if d > 0.3]
        if pos:
            dg_pos = [dgs[i] for i in pos]
            gap_m = [max(agg[gs[i]]['gap_mean'], 1.0) for i in pos]
            gap_s = [agg[gs[i]]['gap_std'] for i in pos]
            ax.errorbar(dg_pos, gap_m, yerr=gap_s, fmt='b^-', ms=6, lw=2, capsize=4, label='Transfer gap')
            if len(dg_pos) >= 3:
                c = np.polyfit(np.log(dg_pos), np.log(gap_m), 1)
                xf = np.logspace(np.log10(min(dg_pos)*0.8), np.log10(max(dg_pos)*1.2), 20)
                ax.loglog(xf, np.exp(np.polyval(c, np.log(xf))), 'r--', lw=2,
                          label=f'Slope $\\approx {c[0]:.1f}$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$|\Delta g|$')
        ax.set_ylabel('Transfer gap')
        ax.set_title(f'{env_id}: Gap (3 seeds)')
        ax.legend(fontsize=8)

        # (d) Action displacement
        ax = axes[row, 3]
        disp_m = [agg[g]['disp_mean'] for g in gs]
        disp_s = [agg[g]['disp_std'] for g in gs]
        ax.errorbar(dgs, disp_m, yerr=disp_s, fmt='go', ms=8, capsize=4,
                    ecolor='black', elinewidth=1, zorder=3)
        pos_dg = [(d, m) for d, m in zip(dgs, disp_m) if d > 0]
        if len(pos_dg) >= 3:
            c = np.polyfit([x[0] for x in pos_dg], [x[1] for x in pos_dg], 1)
            xf = np.linspace(0, max(dgs)*1.1, 50)
            ax.plot(xf, np.maximum(np.polyval(c, xf), 0), 'r--', lw=2,
                    label=f'Slope $\\approx {c[0]:.2f}$')
        ax.set_xlabel(r'$|\Delta g|$')
        ax.set_ylabel(r'Mean $\|a_s - a_t\|$')
        ax.set_title(f'{env_id}: Displacement (3 seeds)')
        ax.legend()

    plt.suptitle(f'Multi-Seed Transfer ({len(SEEDS)} seeds: {SEEDS})', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'exp10_multiseed.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'exp10_multiseed.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved multiseed figure to {save_dir}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if len(sys.argv) < 2:
        print("Usage: python exp10_multiseed.py [train|eval|plot] [halfcheetah|ant|hopper]")
        sys.exit(1)

    action = sys.argv[1].lower()

    if action == 'plot':
        plot_results()
    elif action in ('train', 'eval'):
        if len(sys.argv) < 3:
            print(f"Usage: python exp10_multiseed.py {action} [halfcheetah|ant|hopper]")
            sys.exit(1)
        env_name = sys.argv[2].lower()
        if env_name not in ENV_CONFIGS:
            print(f"Unknown env: {env_name}. Choose from {list(ENV_CONFIGS.keys())}")
            sys.exit(1)
        if action == 'train':
            run_training(env_name)
        else:
            run_eval(env_name)
    else:
        print(f"Unknown action: {action}")
