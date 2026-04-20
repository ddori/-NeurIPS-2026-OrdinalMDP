"""
Experiment 2: Ordinal Stability Radius
=======================================
Validates Theorems 3 & 4 (stability radius) and Proposition 4 (Bellman sensitivity).

Shows:
(1) OC holds within the computed radius, breaks outside
(2) Tight radius (via sensitivity) vs conservative radius (via 2L_Q)
(3) State-level violation characterization (Corollary 1)
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

SEED = 42
np.random.seed(SEED)


class ParameterizedChainMDP:
    """
    Chain MDP with n_states states. Actions: 0=stay, 1=right, 2=left.
    theta controls transition probabilities (slip probability).
    Designed to have clear action gaps that shrink with theta.
    """
    def __init__(self, n_states=8, gamma=0.9):
        self.n_states = n_states
        self.n_actions = 3
        self.gamma = gamma
        self.Rmax = 1.0
        self.Vmax = self.Rmax / (1 - gamma)

    def build(self, theta):
        """theta in R^1, centered at 0."""
        nS, nA = self.n_states, self.n_actions
        P = np.zeros((nS, nA, nS))
        R = np.zeros((nS, nA))

        for s in range(nS):
            # Reward at rightmost state
            R[s, :] = -0.01
            if s == nS - 1:
                R[s, :] = 1.0
            if s == nS // 2:
                R[s, :] = 0.3 + 0.1 * theta  # theta-dependent reward

            slip = 0.1 + 0.15 * theta  # slip increases with theta
            slip = np.clip(slip, 0.01, 0.49)

            for a in range(nA):
                if a == 0:  # stay
                    P[s, a, s] = 1.0 - slip
                    P[s, a, min(s+1, nS-1)] += slip / 2
                    P[s, a, max(s-1, 0)] += slip / 2
                elif a == 1:  # right
                    ns = min(s + 1, nS - 1)
                    P[s, a, ns] = 1.0 - slip
                    P[s, a, s] += slip
                elif a == 2:  # left
                    ns = max(s - 1, 0)
                    P[s, a, ns] = 1.0 - slip
                    P[s, a, s] += slip

            # Normalize
            for a in range(nA):
                P[s, a, :] /= P[s, a, :].sum()

        return P, R


def value_iteration(P, R, gamma, tol=1e-12):
    nS, nA = R.shape
    Q = np.zeros((nS, nA))
    for _ in range(10000):
        V = Q.max(axis=1)
        Q_new = np.zeros_like(Q)
        for s in range(nS):
            for a in range(nA):
                Q_new[s, a] = R[s, a] + gamma * P[s, a, :] @ V
        if np.max(np.abs(Q_new - Q)) < tol:
            return Q_new, Q_new.max(axis=1), Q_new.argmax(axis=1)
        Q = Q_new
    return Q, Q.max(axis=1), Q.argmax(axis=1)


def compute_q_gap(Q, s):
    """Min gap at state s: min_{a != a*} Q(s, a*) - Q(s, a)."""
    a_star = Q[s].argmax()
    gaps = Q[s, a_star] - Q[s, :]
    gaps[a_star] = np.inf
    return gaps.min()


def compute_L_Q(env, theta_range, n_samples=200):
    """Estimate L_Q by finite differences."""
    thetas = np.linspace(theta_range[0], theta_range[1], n_samples)
    max_ratio = 0
    Q_prev = None
    for i, th in enumerate(thetas):
        P, R = env.build(th)
        Q, _, _ = value_iteration(P, R, env.gamma)
        if Q_prev is not None:
            diff = np.max(np.abs(Q - Q_prev))
            dtheta = thetas[1] - thetas[0]
            max_ratio = max(max_ratio, diff / dtheta)
        Q_prev = Q
    return max_ratio


def compute_gap_sensitivity(env, theta0, eps=1e-5):
    """Compute sigma_theta(s,a,a') = dG/dtheta via finite differences."""
    nS, nA = env.n_states, env.n_actions
    P_plus, R_plus = env.build(theta0 + eps)
    P_minus, R_minus = env.build(theta0 - eps)
    Q_plus, _, _ = value_iteration(P_plus, R_plus, env.gamma)
    Q_minus, _, _ = value_iteration(P_minus, R_minus, env.gamma)

    sigma = np.zeros((nS, nA, nA))
    for s in range(nS):
        for a in range(nA):
            for ap in range(nA):
                G_plus = Q_plus[s, a] - Q_plus[s, ap]
                G_minus = Q_minus[s, a] - Q_minus[s, ap]
                sigma[s, a, ap] = (G_plus - G_minus) / (2 * eps)
    return sigma


def run_experiment():
    env = ParameterizedChainMDP(n_states=8, gamma=0.9)
    theta0 = 0.0

    # Compute at theta0
    P0, R0 = env.build(theta0)
    Q0, V0, pi0 = value_iteration(P0, R0, env.gamma)
    nS, nA = env.n_states, env.n_actions

    # ─── Conservative radius (Theorem 3) ───
    L_Q = compute_L_Q(env, [-2, 2])
    L_G_conservative = 2 * L_Q

    conservative_rho_per_state = {}
    for s in range(nS):
        min_gap = np.inf
        for a in range(nA):
            for ap in range(nA):
                if a != ap:
                    gap = abs(Q0[s, a] - Q0[s, ap])
                    if gap > 0:
                        min_gap = min(min_gap, gap)
        conservative_rho_per_state[s] = min_gap / L_G_conservative if min_gap < np.inf else np.inf

    conservative_radius = min(conservative_rho_per_state.values())

    # ─── Tight radius (Theorem 4) via sensitivity ───
    # Compute sigma over a range of theta
    sigma_max = np.zeros((nS, nA, nA))
    for th in np.linspace(-1.5, 1.5, 100):
        sigma = compute_gap_sensitivity(env, th)
        sigma_max = np.maximum(sigma_max, np.abs(sigma))

    tight_rho_per_state = {}
    for s in range(nS):
        min_ratio = np.inf
        for a in range(nA):
            for ap in range(nA):
                if a != ap:
                    gap = abs(Q0[s, a] - Q0[s, ap])
                    if gap > 0 and sigma_max[s, a, ap] > 1e-10:
                        ratio = gap / sigma_max[s, a, ap]
                        min_ratio = min(min_ratio, ratio)
        tight_rho_per_state[s] = min_ratio if min_ratio < np.inf else np.inf

    tight_radius = min(tight_rho_per_state.values())

    # ─── Empirical: sweep theta and check OC ───
    thetas = np.linspace(-2.0, 2.0, 500)
    oc_holds = []
    n_viol_states = []
    viol_per_state = {s: [] for s in range(nS)}

    for th in thetas:
        P_t, R_t = env.build(th)
        Q_t, V_t, pi_t = value_iteration(P_t, R_t, env.gamma)

        # Check OC: does the ranking match theta0?
        violations = 0
        for s in range(nS):
            if pi_t[s] != pi0[s]:
                violations += 1
                viol_per_state[s].append(th)

        oc_holds.append(violations == 0)
        n_viol_states.append(violations)

    # Find empirical radius
    empirical_radius = 0
    for i, th in enumerate(thetas):
        if th >= 0 and oc_holds[i]:
            empirical_radius = max(empirical_radius, abs(th))
    # Also check negative
    for i, th in enumerate(thetas):
        if th <= 0 and oc_holds[i]:
            empirical_radius_neg = abs(th)
    empirical_radius = min(empirical_radius, empirical_radius_neg) if 'empirical_radius_neg' in dir() else empirical_radius

    # Find true empirical radius more carefully
    emp_pos = 0
    for th, oc in zip(thetas, oc_holds):
        if th > 0 and oc:
            emp_pos = th
        elif th > 0 and not oc:
            break
    emp_neg = 0
    for th, oc in zip(reversed(thetas.tolist()), reversed(oc_holds)):
        if th < 0 and oc:
            emp_neg = abs(th)
        elif th < 0 and not oc:
            break
    empirical_radius = min(emp_pos, emp_neg) if emp_neg > 0 else emp_pos

    results = {
        'thetas': thetas.tolist(),
        'oc_holds': oc_holds,
        'n_viol_states': n_viol_states,
        'conservative_radius': conservative_radius,
        'tight_radius': tight_radius,
        'empirical_radius': empirical_radius,
        'conservative_per_state': conservative_rho_per_state,
        'tight_per_state': tight_rho_per_state,
        'viol_per_state': {s: v for s, v in viol_per_state.items()},
        'action_gaps': {s: float(compute_q_gap(Q0, s)) for s in range(nS)},
    }
    return results


def plot_results(results, save_dir='../figures'):
    os.makedirs(save_dir, exist_ok=True)
    thetas = np.array(results['thetas'])
    nS = len(results['action_gaps'])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) OC region with radii
    ax = axes[0]
    oc_arr = np.array(results['oc_holds'], dtype=float)
    n_viol = np.array(results['n_viol_states'])

    ax.fill_between(thetas, 0, 1, where=oc_arr == 1, color='lightgreen', alpha=0.5, label='OC holds')
    ax.fill_between(thetas, 0, 1, where=oc_arr == 0, color='lightcoral', alpha=0.5, label='OC violated')
    ax.axvline(-results['conservative_radius'], color='blue', ls='--', lw=2, label=f"Conservative $\\rho$={results['conservative_radius']:.3f}")
    ax.axvline(results['conservative_radius'], color='blue', ls='--', lw=2)
    ax.axvline(-results['tight_radius'], color='red', ls='-', lw=2, label=f"Tight $\\rho$={results['tight_radius']:.3f}")
    ax.axvline(results['tight_radius'], color='red', ls='-', lw=2)
    ax.axvline(-results['empirical_radius'], color='green', ls=':', lw=2, label=f"Empirical={results['empirical_radius']:.3f}")
    ax.axvline(results['empirical_radius'], color='green', ls=':', lw=2)
    ax.set_xlabel(r'$\theta - \theta_0$')
    ax.set_ylabel('OC status')
    ax.set_title('(a) Ordinal Consistency Region')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_ylim(0, 1.1)

    # (b) Number of violation states
    ax = axes[1]
    ax.plot(thetas, n_viol, 'k-', lw=1.5)
    ax.axvline(-results['conservative_radius'], color='blue', ls='--', lw=1.5)
    ax.axvline(results['conservative_radius'], color='blue', ls='--', lw=1.5)
    ax.axvline(-results['tight_radius'], color='red', ls='-', lw=1.5)
    ax.axvline(results['tight_radius'], color='red', ls='-', lw=1.5)
    ax.set_xlabel(r'$\theta - \theta_0$')
    ax.set_ylabel(r'$|\mathcal{S}_{\mathrm{viol}}|$')
    ax.set_title('(b) Violation States vs $\\theta$')

    # (c) State-level radii
    ax = axes[2]
    states = list(range(nS))
    cons_radii = [min(results['conservative_per_state'][s], 3.0) for s in states]
    tight_radii = [min(results['tight_per_state'][s], 3.0) for s in states]
    gaps = [results['action_gaps'][s] for s in states]

    x = np.arange(nS)
    width = 0.3
    ax.bar(x - width/2, cons_radii, width, color='steelblue', alpha=0.7, label='Conservative $\\rho_s$')
    ax.bar(x + width/2, tight_radii, width, color='coral', alpha=0.7, label='Tight $\\rho_s$')
    ax2 = ax.twinx()
    ax2.plot(x, gaps, 'g^-', markersize=8, lw=1.5, label='Action gap $\\kappa$')
    ax.set_xlabel('State $s$')
    ax.set_ylabel('Stability radius $\\rho_s$')
    ax2.set_ylabel('Action gap $\\kappa(s)$', color='green')
    ax.set_title('(c) Per-State Stability Radius')
    ax.set_xticks(x)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'exp2_stability_radius.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'exp2_stability_radius.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved exp2 figures to {save_dir}")


if __name__ == '__main__':
    results = run_experiment()
    plot_results(results)

    print(f"\nConservative radius: {results['conservative_radius']:.4f}")
    print(f"Tight radius: {results['tight_radius']:.4f}")
    print(f"Empirical radius: {results['empirical_radius']:.4f}")
    print(f"Tightening factor: {results['tight_radius']/results['conservative_radius']:.2f}x")
