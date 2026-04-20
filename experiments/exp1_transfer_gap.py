"""
Experiment 1: Transfer Gap Decomposition vs Simulation Lemma
============================================================
Validates Theorem 1 (exact transfer gap) and Remark 2 (dominance over Sim Lemma).

Setup: Parameterized tabular MDP family with physics parameter theta controlling transitions.
- Grid-world with stochastic wind (theta controls wind strength)
- Discrete actions: {up, down, left, right}
- Shows: (1) our decomposition is exact, (2) Simulation Lemma is loose
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
import json

SEED = 42
np.random.seed(SEED)

# ─── Parameterized Grid-World MDP ───
class ParameterizedGridMDP:
    """
    5x5 grid. Actions: 0=up, 1=down, 2=left, 3=right.
    theta in [0,1] controls wind strength pushing agent rightward.
    Reward: +1 at goal (4,4), -0.1 step cost, +0.5 at subgoal (2,2).
    """
    def __init__(self, grid_size=5, gamma=0.95):
        self.grid_size = grid_size
        self.n_states = grid_size * grid_size
        self.n_actions = 4
        self.gamma = gamma
        self.Rmax = 1.0
        self.Vmax = self.Rmax / (1 - gamma)
        self.goal = (grid_size - 1, grid_size - 1)
        self.subgoal = (grid_size // 2, grid_size // 2)

    def _to_state(self, r, c):
        return r * self.grid_size + c

    def _to_rc(self, s):
        return s // self.grid_size, s % self.grid_size

    def _move(self, r, c, action):
        dr = [-1, 1, 0, 0]
        dc = [0, 0, -1, 1]
        nr = np.clip(r + dr[action], 0, self.grid_size - 1)
        nc = np.clip(c + dc[action], 0, self.grid_size - 1)
        return int(nr), int(nc)

    def build_transition(self, theta):
        """P[s, a, s'] with wind parameter theta."""
        nS, nA = self.n_states, self.n_actions
        P = np.zeros((nS, nA, nS))
        for s in range(nS):
            r, c = self._to_rc(s)
            if (r, c) == self.goal:
                P[s, :, s] = 1.0  # absorbing
                continue
            for a in range(nA):
                nr, nc = self._move(r, c, a)
                # Wind pushes right with prob theta * 0.3
                wind_prob = theta * 0.3
                nr_wind, nc_wind = self._move(r, c, 3)  # wind = right
                intended = self._to_state(nr, nc)
                wind_dest = self._to_state(nr_wind, nc_wind)
                if intended == wind_dest:
                    P[s, a, intended] = 1.0
                else:
                    P[s, a, intended] += (1.0 - wind_prob)
                    P[s, a, wind_dest] += wind_prob
                # Small random noise
                noise = 0.05
                P[s, a, :] = (1 - noise) * P[s, a, :] + noise / nS
                P[s, a, :] /= P[s, a, :].sum()
        return P

    def build_reward(self, theta):
        """R[s, a] - slight theta dependence in step cost."""
        nS, nA = self.n_states, self.n_actions
        R = np.full((nS, nA), -0.1 - 0.02 * theta)
        for a in range(nA):
            R[self._to_state(*self.goal), a] = 1.0
            R[self._to_state(*self.subgoal), a] = 0.5
        return R


def value_iteration(P, R, gamma, tol=1e-12):
    nS, nA = R.shape
    Q = np.zeros((nS, nA))
    for _ in range(10000):
        V = Q.max(axis=1)
        Q_new = R + gamma * (P @ V)  # P is (nS, nA, nS), V is (nS,)
        # Q_new[s,a] = R[s,a] + gamma * sum_s' P[s,a,s'] V[s']
        Q_new2 = np.zeros_like(Q)
        for s in range(nS):
            for a in range(nA):
                Q_new2[s, a] = R[s, a] + gamma * P[s, a, :] @ V
        if np.max(np.abs(Q_new2 - Q)) < tol:
            return Q_new2, Q_new2.max(axis=1), Q_new2.argmax(axis=1)
        Q = Q_new2
    return Q, Q.max(axis=1), Q.argmax(axis=1)


def policy_evaluation(P, R, gamma, pi, tol=1e-12):
    """Evaluate deterministic policy pi in MDP (P, R, gamma)."""
    nS = R.shape[0]
    # Build P_pi and R_pi
    P_pi = np.zeros((nS, nS))
    R_pi = np.zeros(nS)
    for s in range(nS):
        a = pi[s]
        P_pi[s, :] = P[s, a, :]
        R_pi[s] = R[s, a]
    # V = R_pi + gamma * P_pi @ V => V = (I - gamma P_pi)^{-1} R_pi
    V = np.linalg.solve(np.eye(nS) - gamma * P_pi, R_pi)
    Q = np.zeros((nS, R.shape[1]))
    for s in range(nS):
        for a in range(R.shape[1]):
            Q[s, a] = R[s, a] + gamma * P[s, a, :] @ V
    return V, Q


def discounted_visitation(P, gamma, pi, rho, nS):
    """Compute d^pi_{theta,rho}(s) = (1-gamma) sum_t gamma^t Pr(s_t=s)."""
    P_pi = np.zeros((nS, nS))
    for s in range(nS):
        P_pi[s, :] = P[s, pi[s], :]
    # d = (1 - gamma) * rho @ (I - gamma * P_pi)^{-1}
    d = (1 - gamma) * rho @ np.linalg.solve((np.eye(nS) - gamma * P_pi).T, np.eye(nS)).T
    return d


def simulation_lemma_bound(P_s, P_t, Rmax, gamma):
    """Standard Simulation Lemma upper bound."""
    max_tv = 0
    nS, nA, _ = P_s.shape
    for s in range(nS):
        for a in range(nA):
            tv = np.sum(np.abs(P_s[s, a, :] - P_t[s, a, :])) / 2
            max_tv = max(max_tv, tv)
    return 2 * gamma * Rmax / (1 - gamma)**2 * max_tv


def run_experiment():
    env = ParameterizedGridMDP(grid_size=5, gamma=0.95)
    nS, nA, gamma = env.n_states, env.n_actions, env.gamma

    # Source parameter
    theta_s = 0.0
    P_s = env.build_transition(theta_s)
    R_s = env.build_reward(theta_s)
    Q_s, V_s, pi_s = value_iteration(P_s, R_s, gamma)

    # Sweep target parameters
    thetas = np.linspace(0.0, 1.0, 50)
    rho = np.ones(nS) / nS  # uniform initial

    exact_gaps = []
    our_bound = []
    sim_lemma = []
    n_violations = []

    for theta_t in thetas:
        P_t = env.build_transition(theta_t)
        R_t = env.build_reward(theta_t)
        Q_t, V_t, pi_t = value_iteration(P_t, R_t, gamma)

        # Deploy pi_s in M_{theta_t}
        V_pi_s, Q_pi_s = policy_evaluation(P_t, R_t, gamma, pi_s)

        # True transfer gap
        true_gap = rho @ V_t - rho @ V_pi_s
        exact_gaps.append(true_gap)

        # Our exact decomposition (Theorem 1)
        d_pi_s = discounted_visitation(P_t, gamma, pi_s, rho, nS)
        Sviol = set()
        our_decomp = 0.0
        for s in range(nS):
            delta_s = V_t[s] - Q_t[s, pi_s[s]]
            if pi_s[s] != pi_t[s]:
                Sviol.add(s)
            our_decomp += d_pi_s[s] * delta_s
        our_decomp /= (1 - gamma)
        our_bound.append(our_decomp)
        n_violations.append(len(Sviol))

        # Simulation Lemma bound
        sl = simulation_lemma_bound(P_s, P_t, env.Rmax, gamma)
        sim_lemma.append(sl)

    # ─── Violation Growth (Theorem: Violation Growth Rate) ───
    # Compute action gaps kappa(s, theta_0)
    action_gaps = np.zeros(nS)
    for s in range(nS):
        best_a = pi_s[s]
        gaps = [Q_s[s, best_a] - Q_s[s, a] for a in range(nA) if a != best_a]
        action_gaps[s] = min(gaps) if gaps else float('inf')

    # Compute L_Q empirically
    L_Q = 0.0
    for theta_t in thetas[1:]:
        P_t = env.build_transition(theta_t)
        R_t = env.build_reward(theta_t)
        Q_t, _, _ = value_iteration(P_t, R_t, gamma)
        dtheta = abs(theta_t - theta_s)
        if dtheta > 0:
            L_Q = max(L_Q, np.max(np.abs(Q_t - Q_s)) / dtheta)

    # Build F_kappa CDF
    sorted_gaps = np.sort(action_gaps)
    def F_kappa(x):
        return np.sum(action_gaps <= x) / nS

    # Predicted vs actual violation fraction
    predicted_viol_frac = []
    actual_viol_frac = []
    graceful_bound = []
    for i, theta_t in enumerate(thetas):
        dtheta = abs(theta_t - theta_s)
        pred = F_kappa(2 * L_Q * dtheta)
        predicted_viol_frac.append(pred)
        actual_viol_frac.append(n_violations[i] / nS)
        graceful_bound.append(env.Vmax / (1 - gamma) * pred)

    results = {
        'thetas': thetas.tolist(),
        'exact_gaps': exact_gaps,
        'our_bound': our_bound,
        'sim_lemma': sim_lemma,
        'n_violations': n_violations,
        'action_gaps': action_gaps.tolist(),
        'L_Q': float(L_Q),
        'predicted_viol_frac': predicted_viol_frac,
        'actual_viol_frac': actual_viol_frac,
        'graceful_bound': graceful_bound,
    }
    return results


def plot_results(results, save_dir='../figures'):
    os.makedirs(save_dir, exist_ok=True)
    thetas = results['thetas']

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))

    # (a) Transfer gap comparison
    ax = axes[0]
    ax.plot(thetas, results['exact_gaps'], 'b-', lw=2, label='True gap')
    ax.plot(thetas, results['our_bound'], 'r--', lw=2, label='Our decomp. (Thm 1)')
    ax.fill_between(thetas, 0, results['sim_lemma'], alpha=0.15, color='gray')
    ax.plot(thetas, results['sim_lemma'], 'k:', lw=1.5, label='Simulation Lemma')
    ax.set_xlabel(r'Target parameter $\theta_t$')
    ax.set_ylabel('Transfer gap')
    ax.set_title('(a) Transfer Gap: Exact vs Bounds')
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=-0.01)

    # (b) Ratio: Sim Lemma / True gap
    ax = axes[1]
    ratios = []
    for eg, sl in zip(results['exact_gaps'], results['sim_lemma']):
        if eg > 1e-8:
            ratios.append(sl / eg)
        else:
            ratios.append(np.nan)
    ax.plot(thetas, ratios, 'g-', lw=2)
    ax.set_xlabel(r'Target parameter $\theta_t$')
    ax.set_ylabel('Sim Lemma / True gap')
    ax.set_title('(b) Looseness of Simulation Lemma')
    ax.set_yscale('log')

    # (c) Number of violation states
    ax = axes[2]
    ax.bar(thetas, results['n_violations'], width=0.018, color='coral', alpha=0.8)
    ax.set_xlabel(r'Target parameter $\theta_t$')
    ax.set_ylabel(r'$|\mathcal{S}_{\mathrm{viol}}|$')
    ax.set_title('(c) Violation Set Size')

    # (d) Violation Growth: F_kappa bound vs actual
    ax = axes[3]
    ax.plot(thetas, results['actual_viol_frac'], 'b-', lw=2, label=r'Actual $|\mathcal{S}_{\mathrm{viol}}|/|\mathcal{S}|$')
    ax.plot(thetas, results['predicted_viol_frac'], 'r--', lw=2, label=r'$F_\kappa(2L_Q\|\Delta\theta\|)$ bound')
    ax.set_xlabel(r'Target parameter $\theta_t$')
    ax.set_ylabel('Violation fraction')
    ax.set_title('(d) Violation Growth Rate')
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=-0.01)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'exp1_transfer_gap.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'exp1_transfer_gap.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved exp1 figures to {save_dir}")


if __name__ == '__main__':
    results = run_experiment()
    plot_results(results)

    # Print key stats
    print(f"\nMax Simulation Lemma bound: {max(results['sim_lemma']):.4f}")
    print(f"Max true gap: {max(results['exact_gaps']):.6f}")
    print(f"Our decomposition matches true gap: "
          f"{np.allclose(results['exact_gaps'], results['our_bound'], atol=1e-8)}")
    print(f"Max |Sviol|: {max(results['n_violations'])}")
    print(f"Looseness ratio range: {min(r for r in [s/e if e > 1e-8 else float('inf') for s, e in zip(results['sim_lemma'], results['exact_gaps'])] if np.isfinite(r)):.1f}x "
          f"to {max(r for r in [s/e if e > 1e-8 else 0 for s, e in zip(results['sim_lemma'], results['exact_gaps'])]):.1f}x")
    # Violation growth stats
    print(f"\nViolation Growth (Theorem violation-growth):")
    print(f"  L_Q = {results['L_Q']:.4f}")
    print(f"  Action gaps: min={min(results['action_gaps']):.4f}, "
          f"median={np.median(results['action_gaps']):.4f}")
    bound_valid = all(p >= a - 1e-10 for p, a in
                      zip(results['predicted_viol_frac'], results['actual_viol_frac']))
    print(f"  F_kappa bound valid (pred >= actual everywhere): {bound_valid}")
