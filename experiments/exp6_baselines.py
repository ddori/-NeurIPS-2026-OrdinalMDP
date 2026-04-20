"""
Experiment 6: Comparison with Transfer Baselines
==================================================
Key insight: Ordinal transfer uses only action RANKINGS, not values.
When source environments have heterogeneous reward magnitudes,
Q-value averaging and DR are biased by high-magnitude sources,
while ordinal majority vote gives each source ONE VOTE regardless of scale.

Setup: 10x10 grid, theta = [wind, ice, goal_pref].
Sources have multiplicative reward perturbations that create
scale heterogeneity: some sources have large Q-values, others small.
DR/Q-avg are dominated by high-value sources; Ordinal is not.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'font.size': 13, 'font.family': 'serif',
    'axes.labelsize': 14, 'axes.titlesize': 14,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 11, 'figure.dpi': 150,
})
import os
from collections import defaultdict

SEED = 42
np.random.seed(SEED)


class ChallengeGridMDP:
    """
    10x10 grid with two goals, wind, ice, hazard corridor.
    theta = [wind, ice, goal_pref].
    """
    def __init__(self, grid_size=10, gamma=0.95):
        self.grid_size = grid_size
        self.n_states = grid_size * grid_size
        self.n_actions = 4
        self.gamma = gamma
        self.goal_A = (1, grid_size - 2)
        self.goal_B = (grid_size - 2, 1)
        self.hazard_row = grid_size // 2

    def _to_state(self, r, c):
        return r * self.grid_size + c

    def _to_rc(self, s):
        return s // self.grid_size, s % self.grid_size

    def _move(self, r, c, action):
        dr, dc = [-1, 1, 0, 0], [0, 0, -1, 1]
        nr = np.clip(r + dr[action], 0, self.grid_size - 1)
        nc = np.clip(c + dc[action], 0, self.grid_size - 1)
        return int(nr), int(nc)

    def build(self, theta):
        wind, ice, gpref = theta[0], theta[1], theta[2]
        nS, nA = self.n_states, self.n_actions
        gs = self.grid_size
        P = np.zeros((nS, nA, nS))
        R = np.zeros((nS, nA))

        for s in range(nS):
            r, c = self._to_rc(s)
            R[s, :] = -0.1
            if (r, c) == self.goal_A:
                R[s, :] = 2.0 + 3.0 * gpref
            elif (r, c) == self.goal_B:
                R[s, :] = 2.0 - 3.0 * gpref
            elif r == self.hazard_row:
                R[s, :] = -0.4 - 0.2 * abs(wind)

            if (r, c) in [self.goal_A, self.goal_B]:
                P[s, :, s] = 1.0
                continue

            for a in range(nA):
                nr, nc = self._move(r, c, a)
                intended = self._to_state(nr, nc)
                wind_action = 3 if wind > 0 else 2
                wind_prob = min(abs(wind) * 0.4, 0.45)
                nr_w, nc_w = self._move(r, c, wind_action)
                wind_dest = self._to_state(nr_w, nc_w)
                ice_prob = ice * 0.3 if abs(r - self.hazard_row) <= 1 else 0
                p_intended = max(1.0 - wind_prob - ice_prob, 0.1)
                P[s, a, intended] += p_intended
                if wind_dest != intended:
                    P[s, a, wind_dest] += wind_prob
                else:
                    P[s, a, intended] += wind_prob
                P[s, a, s] += ice_prob
                P[s, a, :] += 1e-6
                P[s, a, :] /= P[s, a, :].sum()

        return P, R


def value_iteration(P, R, gamma, tol=1e-12):
    nS, nA = R.shape
    Q = np.zeros((nS, nA))
    for _ in range(10000):
        V = Q.max(axis=1)
        Q_new = R + gamma * np.einsum('sai,i->sa', P, V)
        if np.max(np.abs(Q_new - Q)) < tol:
            return Q_new, Q_new.max(axis=1), Q_new.argmax(axis=1)
        Q = Q_new
    return Q, Q.max(axis=1), Q.argmax(axis=1)


def policy_eval(P, R, gamma, pi):
    nS = R.shape[0]
    P_pi = np.array([P[s, pi[s], :] for s in range(nS)])
    R_pi = np.array([R[s, pi[s]] for s in range(nS)])
    return np.linalg.solve(np.eye(nS) - gamma * P_pi, R_pi)


def majority_vote_policy(Q_list):
    nS, nA = Q_list[0].shape
    Qs = np.array(Q_list)
    pis = Qs.argmax(axis=2)
    policy = np.zeros(nS, dtype=int)
    for s in range(nS):
        counts = np.bincount(pis[:, s], minlength=nA)
        policy[s] = counts.argmax()
    return policy


def q_averaging_policy(Q_list):
    Q_avg = np.mean(Q_list, axis=0)
    return Q_avg.argmax(axis=1)


def run_experiment():
    print("=== Baseline Comparison ===")
    env = ChallengeGridMDP(grid_size=10, gamma=0.95)
    gamma = env.gamma
    nS = env.n_states
    rho = np.ones(nS) / nS

    # ─── Source generation ───
    # Key: sources with goal_pref > 0 have LARGER Q-values than those with < 0
    # because goal A reward is bigger. This creates natural scale heterogeneity.
    K_source = 40
    theta_sources = np.zeros((K_source, 3))
    theta_sources[:, 0] = np.clip(0.2 + 0.4 * np.random.randn(K_source), -1, 1)
    theta_sources[:, 1] = np.clip(0.3 + 0.25 * np.random.randn(K_source), 0, 1)
    # goal_pref: wide spread centered at 0.15
    theta_sources[:, 2] = np.clip(0.15 + 0.5 * np.random.randn(K_source), -1, 1)

    # Pre-compute
    source_data = []
    Q_sources = []
    for th in theta_sources:
        P, R = env.build(th)
        Q, _, _ = value_iteration(P, R, gamma)
        source_data.append((P, R, Q))
        Q_sources.append(Q)

    # Artificially scale Q-values by source-specific multiplier
    # This simulates sources from environments with different reward magnitudes
    # Ordinal: unaffected. Q-avg: biased by high-scale sources.
    np.random.seed(SEED + 100)
    scales = np.exp(np.random.uniform(np.log(0.2), np.log(5.0), K_source))
    Q_scaled = [Q * s for Q, s in zip(Q_sources, scales)]
    # Also create scaled P, R for DR
    source_data_scaled = []
    for i in range(K_source):
        source_data_scaled.append((source_data[i][0],
                                    source_data[i][1] * scales[i],
                                    Q_sources[i] * scales[i]))

    print(f"  Q-value scales: min={scales.min():.2f}, max={scales.max():.2f}")

    # ─── Policies ───
    # Ordinal: uses RANKINGS only => scale doesn't matter
    pi_ordinal = majority_vote_policy(Q_scaled)
    pi_ordinal_unscaled = majority_vote_policy(Q_sources)
    assert np.all(pi_ordinal == pi_ordinal_unscaled), "Ordinal should be scale-invariant!"
    print("  Confirmed: Ordinal policy is scale-invariant")

    # Q-averaging on scaled Q: biased by high-scale sources
    pi_qavg_scaled = q_averaging_policy(Q_scaled)
    pi_qavg_unscaled = q_averaging_policy(Q_sources)
    qavg_diff = np.mean(pi_qavg_scaled != pi_qavg_unscaled)
    print(f"  Q-Avg policy changes under scaling: {qavg_diff:.1%} of states")

    # DR on scaled R: biased
    P_avg = np.mean([d[0] for d in source_data_scaled], axis=0)
    R_avg = np.mean([d[1] for d in source_data_scaled], axis=0)
    for s in range(nS):
        for a in range(env.n_actions):
            P_avg[s, a, :] = np.maximum(P_avg[s, a, :], 0)
            P_avg[s, a, :] /= P_avg[s, a, :].sum()
    _, _, pi_dr_scaled = value_iteration(P_avg, R_avg, gamma)

    # Unscaled DR (oracle baseline)
    P_avg_u = np.mean([d[0] for d in source_data], axis=0)
    R_avg_u = np.mean([d[1] for d in source_data], axis=0)
    for s in range(nS):
        for a in range(env.n_actions):
            P_avg_u[s, a, :] = np.maximum(P_avg_u[s, a, :], 0)
            P_avg_u[s, a, :] /= P_avg_u[s, a, :].sum()
    _, _, pi_dr_unscaled = value_iteration(P_avg_u, R_avg_u, gamma)

    dr_diff = np.mean(pi_dr_scaled != pi_dr_unscaled)
    print(f"  DR policy changes under scaling: {dr_diff:.1%} of states")

    # Robust: maximin on scaled Q
    pi_robust = np.zeros(nS, dtype=int)
    for s in range(nS):
        best_worst = -np.inf
        for a in range(env.n_actions):
            worst = min(Q[s, a] for Q in Q_scaled)
            if worst > best_worst:
                best_worst = worst
                pi_robust[s] = a

    # Single source: nominal
    P_nom, R_nom = env.build(np.array([0.2, 0.3, 0.3]))
    _, _, pi_single = value_iteration(P_nom, R_nom, gamma)

    # ─── Sweep goal_pref ───
    gprefs = np.linspace(-1.0, 1.0, 40)
    results = defaultdict(list)
    print("Sweeping goal preference...")
    for gp in gprefs:
        theta_t = np.array([0.2, 0.3, gp])
        P_t, R_t = env.build(theta_t)
        Q_t, V_t, pi_t = value_iteration(P_t, R_t, gamma)
        results['gpref'].append(gp)
        for name, pi in [('Ordinal', pi_ordinal),
                         ('DR (scaled)', pi_dr_scaled),
                         ('DR (oracle)', pi_dr_unscaled),
                         ('Q-Avg (scaled)', pi_qavg_scaled),
                         ('Robust', pi_robust),
                         ('Single Source', pi_single),
                         ('Target Opt.', pi_t)]:
            V_pi = policy_eval(P_t, R_t, gamma, pi)
            results[name].append(float(rho @ V_pi))

    # ─── Scale heterogeneity sweep ───
    print("Scale heterogeneity sweep...")
    scale_ranges = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0, 25.0, 50.0]
    n_trials = 30
    K_trial = 30

    eval_thetas = [np.array([0.1, 0.3, 0.4]),
                   np.array([0.3, 0.4, -0.2]),
                   np.array([-0.2, 0.2, 0.5])]
    eval_cache = []
    for th_e in eval_thetas:
        P_e, R_e = env.build(th_e)
        _, V_e, _ = value_iteration(P_e, R_e, gamma)
        eval_cache.append((P_e, R_e, V_e))

    scale_results = defaultdict(lambda: defaultdict(list))

    # Pre-compute a big pool of (theta, P, R, Q)
    n_pool = 150
    pool_thetas = np.zeros((n_pool, 3))
    pool_thetas[:, 0] = np.clip(0.2 + 0.4 * np.random.randn(n_pool), -1, 1)
    pool_thetas[:, 1] = np.clip(0.3 + 0.25 * np.random.randn(n_pool), 0, 1)
    pool_thetas[:, 2] = np.clip(0.15 + 0.5 * np.random.randn(n_pool), -1, 1)

    pool_data = []
    for th in pool_thetas:
        P, R = env.build(th)
        Q, _, _ = value_iteration(P, R, gamma)
        pool_data.append((P, R, Q))

    for sr in scale_ranges:
        for trial in range(n_trials):
            idx = np.random.choice(n_pool, K_trial, replace=False)
            sc = np.exp(np.random.uniform(np.log(1.0 / sr), np.log(sr), K_trial))

            Q_sc = [pool_data[i][2] * sc[i] for i in range(K_trial)]
            R_sc = [pool_data[idx[i]][1] * sc[i] for i in range(K_trial)]
            P_sub = [pool_data[idx[i]][0] for i in range(K_trial)]

            # Ordinal
            pi_ord = majority_vote_policy(Q_sc)
            # Q-Avg
            pi_qa = q_averaging_policy(Q_sc)
            # DR scaled
            P_dr = np.mean(P_sub, axis=0)
            R_dr = np.mean(R_sc, axis=0)
            for s in range(nS):
                for a in range(env.n_actions):
                    P_dr[s, a, :] = np.maximum(P_dr[s, a, :], 0)
                    P_dr[s, a, :] /= P_dr[s, a, :].sum()
            _, _, pi_dr = value_iteration(P_dr, R_dr, gamma)

            for P_e, R_e, V_e in eval_cache:
                V_opt = rho @ V_e
                for name, pi in [('Ordinal', pi_ord), ('DR (scaled)', pi_dr),
                                  ('Q-Avg (scaled)', pi_qa)]:
                    V_pi = policy_eval(P_e, R_e, gamma, pi)
                    gap = max(V_opt - rho @ V_pi, 0)
                    scale_results[name][sr].append(gap)

        print(f"  scale_range={sr:.1f}: "
              f"Ord={np.mean(scale_results['Ordinal'][sr]):.2f}, "
              f"DR={np.mean(scale_results['DR (scaled)'][sr]):.2f}, "
              f"QAvg={np.mean(scale_results['Q-Avg (scaled)'][sr]):.2f}")

    # ─── K sweep (with scale heterogeneity) ───
    K_values = [1, 3, 5, 10, 20, 30, 50]
    n_trials_k = 20
    K_sweep = defaultdict(lambda: defaultdict(list))
    print("K sweep...")
    for K in K_values:
        for trial in range(n_trials_k):
            idx = np.random.choice(n_pool, min(K, n_pool), replace=False)
            sc = np.exp(np.random.uniform(np.log(0.2), np.log(5.0), len(idx)))
            Q_sc = [pool_data[idx[i]][2] * sc[i] for i in range(len(idx))]

            pi_ord = majority_vote_policy(Q_sc)

            R_sc = [pool_data[idx[i]][1] * sc[i] for i in range(len(idx))]
            P_sub = [pool_data[idx[i]][0] for i in range(len(idx))]
            P_dr = np.mean(P_sub, axis=0)
            R_dr = np.mean(R_sc, axis=0)
            for s in range(nS):
                for a in range(env.n_actions):
                    P_dr[s, a, :] = np.maximum(P_dr[s, a, :], 0)
                    P_dr[s, a, :] /= P_dr[s, a, :].sum()
            _, _, pi_dr = value_iteration(P_dr, R_dr, gamma)

            for P_e, R_e, V_e in eval_cache:
                for name, pi in [('Ordinal', pi_ord), ('DR (scaled)', pi_dr)]:
                    V_pi = policy_eval(P_e, R_e, gamma, pi)
                    K_sweep[name][K].append(float(rho @ V_pi))

        print(f"  K={K}: Ord={np.mean(K_sweep['Ordinal'][K]):.3f}, "
              f"DR={np.mean(K_sweep['DR (scaled)'][K]):.3f}")

    # ─── 2D heatmap ───
    print("Computing heatmaps...")
    n_grid = 8
    gp_vals = np.linspace(-0.8, 0.8, n_grid)
    w_vals = np.linspace(-0.8, 0.8, n_grid)
    hm_ordinal = np.zeros((n_grid, n_grid))
    hm_dr = np.zeros((n_grid, n_grid))

    for i, w in enumerate(w_vals):
        for j, gp in enumerate(gp_vals):
            theta_t = np.array([w, 0.3, gp])
            P_t, R_t = env.build(theta_t)
            _, V_t, _ = value_iteration(P_t, R_t, gamma)
            V_opt = rho @ V_t
            V_ord = policy_eval(P_t, R_t, gamma, pi_ordinal)
            V_drv = policy_eval(P_t, R_t, gamma, pi_dr_scaled)
            hm_ordinal[i, j] = V_opt - rho @ V_ord
            hm_dr[i, j] = V_opt - rho @ V_drv

    hm_diff = hm_dr - hm_ordinal

    return {
        'gpref_results': dict(results),
        'scale_results': {k: dict(v) for k, v in scale_results.items()},
        'scale_ranges': scale_ranges,
        'K_sweep': {k: dict(v) for k, v in K_sweep.items()},
        'K_values': K_values,
        'V_opt_fixed': np.mean([rho @ ec[2] for ec in eval_cache]),
        'hm_ordinal': hm_ordinal, 'hm_dr': hm_dr, 'hm_diff': hm_diff,
        'gp_vals': gp_vals, 'w_vals': w_vals,
    }


def plot_results(results, save_dir='../figures'):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))

    # (a) Performance vs goal_pref
    ax = axes[0, 0]
    rr = results['gpref_results']
    styles = [('Ordinal', 'blue', '-', 2.5),
              ('DR (scaled)', 'orange', '--', 2),
              ('DR (oracle)', 'orange', ':', 1.5),
              ('Q-Avg (scaled)', 'purple', '-.', 2),
              ('Robust', 'green', '-.', 1.5),
              ('Target Opt.', 'gray', '-', 1.5)]
    for name, color, ls, lw in styles:
        if name in rr:
            ax.plot(rr['gpref'], rr[name], color=color, ls=ls, lw=lw, label=name)
    ax.set_xlabel(r'Goal preference $\theta_3$')
    ax.set_ylabel(r'$V^{\pi}(\rho)$')
    ax.set_title('(a) Performance vs Goal Preference')
    ax.legend(fontsize=7, loc='lower left')

    # (b) Transfer gap
    ax = axes[0, 1]
    opt = np.array(rr['Target Opt.'])
    for name, color, ls, lw in styles:
        if name != 'Target Opt.' and name in rr:
            gap = np.maximum(opt - np.array(rr[name]), 0)
            ax.plot(rr['gpref'], gap, color=color, ls=ls, lw=lw, label=name)
    ax.set_xlabel(r'Goal preference $\theta_3$')
    ax.set_ylabel('Transfer gap')
    ax.set_title('(b) Transfer Gap')
    ax.legend(fontsize=7)

    # (c) Scale heterogeneity sweep — KEY PLOT
    ax = axes[0, 2]
    sr = results['scale_ranges']
    for name, color, marker in [('Ordinal', 'blue', 'o'),
                                 ('DR (scaled)', 'orange', 's'),
                                 ('Q-Avg (scaled)', 'purple', '^')]:
        means = [np.mean(results['scale_results'][name][s]) for s in sr]
        stds = [np.std(results['scale_results'][name][s]) /
                np.sqrt(len(results['scale_results'][name][s])) for s in sr]
        ax.errorbar(sr, means, yerr=stds, fmt=f'{marker}-', color=color,
                    ms=6, lw=2, capsize=3, label=name)
    ax.set_xlabel('Reward scale range (max/min)')
    ax.set_ylabel('Avg transfer gap')
    ax.set_title('(c) Effect of Scale Heterogeneity')
    ax.legend(fontsize=9)
    ax.set_xscale('log')
    ax.axvline(1.0, color='gray', ls='--', alpha=0.3)

    # (d) K sweep
    ax = axes[1, 0]
    K_vals = results['K_values']
    for name, color in [('Ordinal', 'blue'), ('DR (scaled)', 'orange')]:
        if name in results['K_sweep']:
            means = [np.mean(results['K_sweep'][name][K]) for K in K_vals]
            stds = [np.std(results['K_sweep'][name][K]) for K in K_vals]
            ax.errorbar(K_vals, means, yerr=stds, fmt='o-', color=color,
                        ms=5, lw=2, capsize=3, label=name)
    ax.axhline(results['V_opt_fixed'], color='gray', ls='--', alpha=0.5,
               label='Target optimal')
    ax.set_xlabel('$K$ (source environments)')
    ax.set_ylabel(r'$V^{\pi}(\rho)$')
    ax.set_title('(d) Scaling with $K$ (scale range 25$\\times$)')
    ax.legend(fontsize=9)

    # (e) Ordinal gap heatmap
    ax = axes[1, 1]
    vmax = max(np.max(np.abs(results['hm_ordinal'])),
               np.max(np.abs(results['hm_dr'])))
    im = ax.imshow(results['hm_ordinal'], origin='lower', aspect='auto',
                   extent=[results['gp_vals'][0], results['gp_vals'][-1],
                           results['w_vals'][0], results['w_vals'][-1]],
                   cmap='YlOrRd', vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Gap')
    ax.set_xlabel('Goal preference')
    ax.set_ylabel('Wind')
    ax.set_title('(e) Ordinal: Transfer Gap')

    # (f) Ordinal advantage heatmap
    ax = axes[1, 2]
    vmax_d = max(np.max(np.abs(results['hm_diff'])), 0.1)
    im = ax.imshow(results['hm_diff'], origin='lower', aspect='auto',
                   extent=[results['gp_vals'][0], results['gp_vals'][-1],
                           results['w_vals'][0], results['w_vals'][-1]],
                   cmap='RdBu', vmin=-vmax_d, vmax=vmax_d)
    plt.colorbar(im, ax=ax, label='DR gap $-$ Ordinal gap')
    ax.set_xlabel('Goal preference')
    ax.set_ylabel('Wind')
    ax.set_title('(f) Ordinal Advantage (blue=better)')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'exp6_baselines.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'exp6_baselines.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved exp6 figures to {save_dir}")


if __name__ == '__main__':
    results = run_experiment()
    plot_results(results)

    rr = results['gpref_results']
    opt = np.array(rr['Target Opt.'])
    print("\n=== Avg Transfer Gap (goal pref sweep) ===")
    for m in ['Ordinal', 'DR (scaled)', 'DR (oracle)', 'Q-Avg (scaled)',
              'Robust', 'Single Source']:
        if m in rr:
            gap = np.maximum(opt - np.array(rr[m]), 0)
            print(f"  {m:20s}: mean={np.mean(gap):.2f}, max={np.max(gap):.2f}")
