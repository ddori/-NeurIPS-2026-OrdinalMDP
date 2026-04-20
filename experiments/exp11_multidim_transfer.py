"""
Experiment 11: Multi-Dimensional Parameter Transfer (d=3)
=========================================================
Tests ordinal transfer when gravity, friction, AND mass change simultaneously.
Validates theory for θ ∈ R^d with d > 1.

Usage:
  python exp11_multidim_transfer.py train     # train all agents (~6 hours)
  python exp11_multidim_transfer.py plot      # plot from cached results
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

SEED = 42
CACHE_DIR = '../cache_exp11'
ENV_ID = 'HalfCheetah-v4'
N_TIMESTEPS = 500_000
N_EVAL_EPISODES = 20
N_TEST_STATES = 300

# ── Source parameters (default) ──
SOURCE_PARAMS = {'gravity': 9.81, 'friction_scale': 1.0, 'mass_scale': 1.0}

# ── Target parameter combinations ──
# Increasing ||Δθ|| with all 3 params changing simultaneously
# θ = (gravity, friction_scale, mass_scale), normalized so each axis has similar scale
# gravity: 5-15 (range 10, center 9.81) → normalize by 5
# friction: 0.3-2.0 (range ~1.7, center 1.0) → normalize by 0.85
# mass: 0.5-2.0 (range 1.5, center 1.0) → normalize by 0.75
TARGETS = [
    # source (for reference)
    {'gravity': 9.81, 'friction_scale': 1.0, 'mass_scale': 1.0, 'label': 'source'},
    # one per distance tier — enough for appendix trend
    # small (||Δθ|| ~ 0.3)
    {'gravity': 9.0,  'friction_scale': 0.9, 'mass_scale': 1.1, 'label': 'small-1'},
    # medium (||Δθ|| ~ 0.8)
    {'gravity': 8.0,  'friction_scale': 0.75, 'mass_scale': 1.25, 'label': 'med-1'},
    # large (||Δθ|| ~ 1.8)
    {'gravity': 6.0,  'friction_scale': 0.5,  'mass_scale': 1.5, 'label': 'large-1'},
    # extreme (||Δθ|| ~ 3.0)
    {'gravity': 5.0,  'friction_scale': 0.4,  'mass_scale': 1.8, 'label': 'extreme-1'},
]


def compute_delta_theta(params):
    """Compute normalized ||Δθ|| from source."""
    dg = (params['gravity'] - SOURCE_PARAMS['gravity']) / 5.0
    df = (params['friction_scale'] - SOURCE_PARAMS['friction_scale']) / 0.85
    dm = (params['mass_scale'] - SOURCE_PARAMS['mass_scale']) / 0.75
    return np.sqrt(dg**2 + df**2 + dm**2)


def make_env(gravity, friction_scale, mass_scale):
    """Create HalfCheetah with modified physics."""
    env = gym.make(ENV_ID)
    env.unwrapped.model.opt.gravity[2] = -gravity
    # Scale all geom friction
    default_friction = np.array([0.4, 0.1, 0.1])
    for i in range(env.unwrapped.model.ngeom):
        env.unwrapped.model.geom_friction[i] = default_friction * friction_scale
    # Scale body masses (skip world body at index 0)
    original_masses = env.unwrapped.model.body_mass.copy()
    for i in range(1, env.unwrapped.model.nbody):
        env.unwrapped.model.body_mass[i] = original_masses[i] * mass_scale
    return env


def get_cache_path(params):
    g = params['gravity']
    f = params['friction_scale']
    m = params['mass_scale']
    return os.path.join(CACHE_DIR, f'sac_halfcheetah_g{g:.2f}_f{f:.2f}_m{m:.2f}_s{SEED}_t{N_TIMESTEPS}.zip')


def train_agent(params):
    """Train SAC agent with given physics parameters."""
    cache_path = get_cache_path(params)
    if os.path.exists(cache_path):
        print(f"  Loading cached: {params['label']}")
        env = make_env(params['gravity'], params['friction_scale'], params['mass_scale'])
        model = SAC.load(cache_path, env=env)
        return model, env

    print(f"  Training: {params['label']} (g={params['gravity']}, "
          f"f={params['friction_scale']}, m={params['mass_scale']})")
    env = make_env(params['gravity'], params['friction_scale'], params['mass_scale'])
    model = SAC(
        'MlpPolicy', env,
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=300_000,
        learning_starts=10000,
        seed=SEED,
        verbose=0,
        device='cpu' if 'extreme' in params['label'] else 'auto',
    )
    t0 = time.time()
    model.learn(total_timesteps=N_TIMESTEPS)
    elapsed = time.time() - t0
    print(f"    Done in {elapsed/60:.1f} min")

    os.makedirs(CACHE_DIR, exist_ok=True)
    model.save(cache_path)
    return model, env


def evaluate(model, params, n_episodes=N_EVAL_EPISODES):
    """Evaluate model in environment with given params."""
    env = make_env(params['gravity'], params['friction_scale'], params['mass_scale'])
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=SEED + ep * 100)
        total_r = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            total_r += r
            done = term or trunc
        returns.append(total_r)
    env.close()
    return np.mean(returns), np.std(returns)


def get_actions(model, states):
    """Get deterministic actions for batch of states."""
    import torch
    actions = []
    for s in states:
        action, _ = model.predict(s, deterministic=True)
        actions.append(action)
    return np.array(actions)


def directional_oc(actions_a, actions_b):
    """Directional OC: fraction of (state, dim) pairs with same sign."""
    signs_a = np.sign(actions_a)
    signs_b = np.sign(actions_b)
    # Ignore near-zero actions
    mask = (np.abs(actions_a) > 0.01) & (np.abs(actions_b) > 0.01)
    if mask.sum() == 0:
        return 0.5
    return np.mean(signs_a[mask] == signs_b[mask])


def run_training():
    """Train all agents."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Multi-Dim Transfer: HalfCheetah (d=3)")
    print(f"  Parameters: gravity × friction × mass")
    print(f"{'='*60}")

    models = {}
    for params in TARGETS:
        model, env = train_agent(params)
        models[params['label']] = model
        env.close()

    print(f"\n[2] Collecting test states...")
    src_model = models['source']
    env = make_env(**{k: v for k, v in SOURCE_PARAMS.items()})
    test_states = []
    s, _ = env.reset(seed=SEED + 999)
    for t in range(5000):
        a, _ = src_model.predict(s, deterministic=True)
        s, r, term, trunc, _ = env.step(a)
        if t >= 500 and t % 10 == 0:
            test_states.append(s.copy())
        if term or trunc:
            s, _ = env.reset(seed=SEED + 999 + t)
    test_states = np.array(test_states[:N_TEST_STATES])
    env.close()
    print(f"  Collected {len(test_states)} test states")

    print(f"\n[3] Evaluating transfer...")
    src_actions = get_actions(src_model, test_states)

    results = []
    for params in TARGETS:
        label = params['label']
        dtheta = compute_delta_theta(params)

        # Source policy deployed at target
        src_ret, src_std = evaluate(src_model, params)
        # Target-trained policy at target
        tgt_ret, tgt_std = evaluate(models[label], params)
        # Transfer gap
        gap = max(tgt_ret - src_ret, 0)

        # OC
        tgt_actions = get_actions(models[label], test_states)
        oc = directional_oc(src_actions, tgt_actions)

        # Action displacement
        disp = np.mean(np.linalg.norm(src_actions - tgt_actions, axis=1))

        results.append({
            'label': label,
            'params': params,
            'dtheta': dtheta,
            'src_return': src_ret,
            'tgt_return': tgt_ret,
            'transfer_gap': gap,
            'oc': oc,
            'displacement': disp,
        })
        print(f"  {label:12s} ||Δθ||={dtheta:.2f}: "
              f"src={src_ret:7.0f}, tgt={tgt_ret:7.0f}, "
              f"gap={gap:7.0f}, OC={oc:.3f}, disp={disp:.3f}")

    # Scale-invariance test
    print(f"\n[4] Scale-invariance...")
    all_actions = [get_actions(models[p['label']], test_states) for p in TARGETS]
    base_signs = np.sign(np.mean([np.sign(a) for a in all_actions], axis=0))
    base_mean = np.mean(all_actions, axis=0)

    scale_ranges = [1, 2, 5, 10, 20, 50, 100]
    mv_inv, ma_inv = [], []
    n_trials = 20
    for sr in scale_ranges:
        mv_t, ma_t = [], []
        for trial in range(n_trials):
            np.random.seed(SEED + trial * 77 + sr)
            scales = np.exp(np.random.uniform(-np.log(max(sr, 1.01)),
                                               np.log(max(sr, 1.01)), len(all_actions)))
            scaled = [a * s for a, s in zip(all_actions, scales)]
            mv_t.append(np.mean(np.sign(np.mean([np.sign(a) for a in scaled], axis=0)) == base_signs))
            scaled_mean = np.mean(scaled, axis=0)
            cos = []
            for i in range(len(test_states)):
                n1, n2 = np.linalg.norm(base_mean[i]), np.linalg.norm(scaled_mean[i])
                if n1 > 1e-6 and n2 > 1e-6:
                    cos.append(np.dot(base_mean[i], scaled_mean[i]) / (n1 * n2))
            ma_t.append(np.mean(cos) if cos else 1.0)
        mv_inv.append(np.mean(mv_t))
        ma_inv.append(np.mean(ma_t))
        print(f"  Scale {sr:3d}x: MV={mv_inv[-1]:.3f}, MA={ma_inv[-1]:.3f}")

    all_results = {
        'targets': results,
        'scale_ranges': scale_ranges,
        'mv_invariance': mv_inv,
        'ma_invariance': ma_inv,
    }

    with open(os.path.join(CACHE_DIR, 'results_multidim.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nSaved results to {CACHE_DIR}/results_multidim.pkl")
    return all_results


def plot_results(save_dir='../figures'):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(CACHE_DIR, 'results_multidim.pkl')
    if not os.path.exists(path):
        print("No results found. Run 'train' first.")
        return
    with open(path, 'rb') as f:
        data = pickle.load(f)

    results = data['targets']
    # Skip source (dtheta=0)
    pts = [r for r in results if r['dtheta'] > 0.01]
    dthetas = [r['dtheta'] for r in pts]
    gaps = [max(r['transfer_gap'], 1.0) for r in pts]
    ocs = [r['oc'] for r in pts]
    disps = [r['displacement'] for r in pts]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    # (a) Source return vs target-trained return
    ax = axes[0]
    labels_a = [r['label'] for r in pts]
    src_rets = [r['src_return'] for r in pts]
    tgt_rets = [r['tgt_return'] for r in pts]
    x_pos = np.arange(len(labels_a))
    w = 0.35
    ax.bar(x_pos - w/2, src_rets, w, color='steelblue', label='Source $\\pi_{\\theta_0}$', edgecolor='black', lw=0.5)
    ax.bar(x_pos + w/2, tgt_rets, w, color='coral', label='Target $\\pi^*_{\\theta}$', edgecolor='black', lw=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{l}\n$\\|\\Delta\\theta\\|$={d:.2f}' for l, d in zip(labels_a, dthetas)], fontsize=7)
    ax.set_ylabel('Return')
    ax.set_title('(a) Transfer Performance ($d=3$)')
    ax.legend(fontsize=8)

    # (b) OC vs ||Δθ||
    ax = axes[1]
    ax.scatter(dthetas, ocs, c='steelblue', s=60, zorder=3, edgecolors='black', lw=0.5)
    if len(dthetas) >= 3:
        c2 = np.polyfit(dthetas, ocs, 2)
        xf = np.linspace(0, max(dthetas) * 1.1, 50)
        ax.plot(xf, np.clip(np.polyval(c2, xf), 0.3, 1.05), 'b-', lw=2, alpha=0.7)
    ax.axhline(0.5, color='red', ls=':', alpha=0.4, label='Random')
    ax.set_xlabel(r'$\|\Delta\theta\|$')
    ax.set_ylabel('Directional OC')
    ax.set_title('(b) OC Decay ($d=3$)')
    ax.set_ylim(0.3, 1.05)
    ax.legend()

    # (c) Scale-invariance
    ax = axes[2]
    ax.semilogx(data['scale_ranges'], data['mv_invariance'], 'bo-', ms=6, lw=2.5,
                label='Ordinal (sign vote)')
    ax.semilogx(data['scale_ranges'], data['ma_invariance'], 's--', color='purple',
                ms=6, lw=2, label='Mean-action')
    ax.axhline(1.0, color='green', ls='--', alpha=0.5)
    ax.set_xlabel('Scale heterogeneity ($r$)')
    ax.set_ylabel('Agreement')
    ax.set_title('(c) Scale-Invariance ($d=3$)')
    ax.set_ylim(0.5, 1.05)
    ax.legend()

    # (d) Action displacement vs ||Δθ|| (check linearity)
    ax = axes[3]
    ax.plot(dthetas, disps, 'go', ms=8, zorder=3)
    if len(dthetas) >= 3:
        c3 = np.polyfit(dthetas, disps, 1)
        xf = np.linspace(0, max(dthetas) * 1.1, 50)
        ax.plot(xf, np.maximum(np.polyval(c3, xf), 0), 'r--', lw=2,
                label=f'Linear fit (slope={c3[0]:.2f})')
    ax.set_xlabel(r'$\|\Delta\theta\|$')
    ax.set_ylabel(r'Mean $\|a_s - a_t\|$')
    ax.set_title('(d) Action Displacement ($d=3$)')
    ax.legend()

    plt.suptitle('HalfCheetah-v4: Multi-Dimensional Transfer ($\\theta \\in \\mathbb{R}^3$: gravity $\\times$ friction $\\times$ mass)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'exp11_multidim_transfer.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'exp11_multidim_transfer.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved exp11 figure to {save_dir}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"  Multi-Dim Transfer Summary (HalfCheetah, d=3)")
    print(f"{'='*70}")
    print(f"  {'Label':>12s}  {'||Δθ||':>7s}  {'Gap':>7s}  {'OC':>6s}  {'Disp':>6s}")
    for r in results:
        print(f"  {r['label']:>12s}  {r['dtheta']:7.3f}  {r['transfer_gap']:7.0f}  "
              f"{r['oc']:6.3f}  {r['displacement']:6.3f}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if len(sys.argv) < 2:
        print("Usage: python exp11_multidim_transfer.py [train|plot]")
        sys.exit(1)

    arg = sys.argv[1].lower()
    if arg == 'train':
        run_training()
    elif arg == 'plot':
        plot_results()
    else:
        print(f"Unknown argument: {arg}")
