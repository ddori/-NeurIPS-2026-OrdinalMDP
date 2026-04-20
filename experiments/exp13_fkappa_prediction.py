"""
Experiment 13: F_kappa as Structural Fingerprint
=================================================
Two sub-experiments validating that F_kappa is the "right quantity":

(a) Per-state violation order: states with smaller action gaps violate first
    as ||Δθ|| increases, exactly as Theorem 2 predicts.

(b) Cross-environment F_kappa shape predicts transfer degradation pattern:
    flat F_kappa near origin → graceful degradation; steep → catastrophic.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'font.size': 13, 'font.family': 'serif',
    'axes.labelsize': 14, 'axes.titlesize': 14,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 10, 'figure.dpi': 150,
})
import os

SEED = 42
np.random.seed(SEED)


# ── MDP Builders ──

class GridMDP:
    """5x5 grid with wind (from Exp 1)."""
    def __init__(self, grid_size=5, gamma=0.95):
        self.grid_size = grid_size
        self.n_states = grid_size * grid_size
        self.n_actions = 4
        self.gamma = gamma
        self.Rmax = 1.0
        self.goal = (grid_size-1, grid_size-1)
        self.subgoal = (grid_size//2, grid_size//2)
        self.name = f'Grid {grid_size}×{grid_size}'

    def _to_state(self, r, c): return r * self.grid_size + c
    def _to_rc(self, s): return s // self.grid_size, s % self.grid_size
    def _move(self, r, c, a):
        dr, dc = [-1,1,0,0], [0,0,-1,1]
        return int(np.clip(r+dr[a], 0, self.grid_size-1)), int(np.clip(c+dc[a], 0, self.grid_size-1))

    def build(self, theta):
        nS, nA = self.n_states, self.n_actions
        P = np.zeros((nS, nA, nS))
        R = np.full((nS, nA), -0.1 - 0.02*theta)
        for s in range(nS):
            r, c = self._to_rc(s)
            if (r,c) == self.goal:
                P[s, :, s] = 1.0
                for a in range(nA): R[s,a] = 1.0
                continue
            if (r,c) == self.subgoal:
                for a in range(nA): R[s,a] = 0.5
            for a in range(nA):
                nr, nc = self._move(r, c, a)
                wp = theta * 0.3
                wr, wc = self._move(r, c, 3)
                intended = self._to_state(nr, nc)
                wind_dest = self._to_state(wr, wc)
                if intended == wind_dest:
                    P[s,a,intended] = 1.0
                else:
                    P[s,a,intended] += 1.0 - wp
                    P[s,a,wind_dest] += wp
                P[s,a,:] = (1-0.05)*P[s,a,:] + 0.05/nS
                P[s,a,:] /= P[s,a,:].sum()
        return P, R


class ChainMDP:
    """Chain with diverse gaps (from Exp 12)."""
    def __init__(self, n_states=20, gamma=0.95):
        self.n_states = n_states
        self.n_actions = 3
        self.gamma = gamma
        self.Rmax = 1.0
        self.name = f'Chain {n_states}'

    def build(self, theta):
        nS, nA = self.n_states, self.n_actions
        P = np.zeros((nS, nA, nS))
        for s in range(nS):
            P[s,0,s] = 0.8; P[s,0,min(s+1,nS-1)] = 0.1; P[s,0,max(s-1,0)] = 0.1
            P[s,1,min(s+1,nS-1)] = 0.8; P[s,1,s] = 0.15; P[s,1,max(s-1,0)] = 0.05
            P[s,2,max(s-1,0)] = 0.8; P[s,2,s] = 0.15; P[s,2,min(s+1,nS-1)] = 0.05
        R = np.zeros((nS, nA))
        gap_base = np.linspace(0.05, 1.0, nS)
        for s in range(nS):
            R[s,0] = gap_base[s]/2 - theta*1.0
            R[s,1] = -gap_base[s]/2 + theta*1.0
            R[s,2] = -gap_base[s]
        return P, R


class CliffMDP:
    """6x6 grid with cliff and two goals. Many near-tie states near cliff edge."""
    def __init__(self, gamma=0.95):
        self.grid_size = 6
        self.n_states = 36
        self.n_actions = 4
        self.gamma = gamma
        self.Rmax = 1.0
        self.name = 'Cliff 6×6'

    def _to_state(self, r, c): return r * self.grid_size + c
    def _to_rc(self, s): return s // self.grid_size, s % self.grid_size
    def _move(self, r, c, a):
        dr, dc = [-1,1,0,0], [0,0,-1,1]
        return int(np.clip(r+dr[a], 0, self.grid_size-1)), int(np.clip(c+dc[a], 0, self.grid_size-1))

    def build(self, theta):
        nS, nA, gs = self.n_states, self.n_actions, self.grid_size
        P = np.zeros((nS, nA, nS))
        R = np.full((nS, nA), -0.05)
        cliff = [(5,c) for c in range(1,5)]  # bottom row middle
        goal1 = (5,5)  # bottom right
        goal2 = (0,5)  # top right

        for s in range(nS):
            r, c = self._to_rc(s)
            if (r,c) == goal1 or (r,c) == goal2:
                P[s,:,s] = 1.0
                R[s,:] = 1.0
                continue
            if (r,c) in cliff:
                P[s,:,self._to_state(0,0)] = 1.0  # reset to start
                R[s,:] = -1.0 - theta*0.5
                continue
            for a in range(nA):
                nr, nc = self._move(r, c, a)
                slip_prob = theta * 0.2
                P[s,a,self._to_state(nr,nc)] += 1.0 - slip_prob
                # Slip: random adjacent
                for a2 in range(nA):
                    nr2, nc2 = self._move(r, c, a2)
                    P[s,a,self._to_state(nr2,nc2)] += slip_prob / nA
                P[s,a,:] /= P[s,a,:].sum()
            # Near cliff edges: very close Q-values → small gaps
            if r == 4 and 0 < c < 5:
                R[s,:] += theta * 0.1  # θ-dependent reward near cliff
        return P, R


def value_iteration(P, R, gamma, tol=1e-12):
    nS, nA = R.shape
    Q = np.zeros((nS, nA))
    for _ in range(10000):
        V = Q.max(axis=1)
        Q_new = np.zeros_like(Q)
        for s in range(nS):
            for a in range(nA):
                Q_new[s,a] = R[s,a] + gamma * P[s,a,:] @ V
        if np.max(np.abs(Q_new - Q)) < tol:
            return Q_new, Q_new.max(axis=1), Q_new.argmax(axis=1)
        Q = Q_new
    return Q, Q.max(axis=1), Q.argmax(axis=1)


def get_action_gaps(Q, pi, nS, nA):
    gaps = np.zeros(nS)
    for s in range(nS):
        g = [Q[s, pi[s]] - Q[s, a] for a in range(nA) if a != pi[s]]
        gaps[s] = min(g) if g else float('inf')
    return gaps


def run_experiment():
    envs = [GridMDP(5, 0.95), ChainMDP(20, 0.95), CliffMDP(0.95)]
    thetas_range = np.linspace(0, 1, 100)
    theta_s = 0.0

    save_dir = '../figures'
    os.makedirs(save_dir, exist_ok=True)

    # ════════════════════════════════════════════
    # Part (a): Per-state violation order
    # ════════════════════════════════════════════
    print("=" * 60)
    print("  Part (a): Per-state violation order")
    print("=" * 60)

    fig_a, axes_a = plt.subplots(1, len(envs), figsize=(6*len(envs), 5))

    for idx, env in enumerate(envs):
        ax = axes_a[idx]
        nS, nA, gamma = env.n_states, env.n_actions, env.gamma

        P_s, R_s = env.build(theta_s)
        Q_s, V_s, pi_s = value_iteration(P_s, R_s, gamma)
        gaps = get_action_gaps(Q_s, pi_s, nS, nA)

        # For each state, find the smallest ||Δθ|| at which it violates
        violation_thresholds = np.full(nS, np.inf)
        for theta_t in thetas_range:
            P_t, R_t = env.build(theta_t)
            Q_t, V_t, pi_t = value_iteration(P_t, R_t, gamma)
            for s in range(nS):
                if pi_s[s] != pi_t[s] and violation_thresholds[s] == np.inf:
                    violation_thresholds[s] = abs(theta_t - theta_s)

        # Plot: action gap vs violation threshold
        finite_mask = violation_thresholds < np.inf
        if finite_mask.sum() > 0:
            ax.scatter(gaps[finite_mask], violation_thresholds[finite_mask],
                      c='steelblue', s=40, alpha=0.7, edgecolors='black', lw=0.5, zorder=3)
            # Fit (only if gaps have some spread)
            if finite_mask.sum() >= 3 and np.std(gaps[finite_mask]) > 1e-8:
                try:
                    c = np.polyfit(gaps[finite_mask], violation_thresholds[finite_mask], 1)
                    xf = np.linspace(0, max(gaps[finite_mask])*1.1, 50)
                    r2 = np.corrcoef(gaps[finite_mask], violation_thresholds[finite_mask])[0,1]**2
                    ax.plot(xf, np.maximum(np.polyval(c, xf), 0), 'r--', lw=2,
                            label=f'Linear fit (R²={r2:.2f})')
                except Exception:
                    pass

        # States that never violate
        never_mask = violation_thresholds == np.inf
        if never_mask.sum() > 0:
            ax.scatter(gaps[never_mask], [max(thetas_range)*1.05]*never_mask.sum(),
                      c='green', s=40, marker='^', alpha=0.7, label=f'Never violated ({never_mask.sum()})')

        ax.set_xlabel('Action gap $\\kappa(s)$ at source')
        ax.set_ylabel('$\\|\\Delta\\theta\\|$ at first violation')
        ax.set_title(f'{env.name}')
        ax.legend(fontsize=8)

        # Correlation
        if finite_mask.sum() >= 3:
            corr = np.corrcoef(gaps[finite_mask], violation_thresholds[finite_mask])[0,1]
            print(f"  {env.name}: {finite_mask.sum()} states violated, "
                  f"corr(κ, threshold) = {corr:.3f}")

    plt.suptitle('Per-State: Small Action Gaps Violate First\n'
                 '(Theorem 2 predicts states violate in order of increasing $\\kappa(s)$)',
                 fontsize=13, y=1.04)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'exp13_perstate_violation.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'exp13_perstate_violation.png'), bbox_inches='tight')
    plt.close()

    # ════════════════════════════════════════════
    # Part (b): F_kappa shape predicts degradation
    # ════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  Part (b): F_kappa shape predicts degradation pattern")
    print("=" * 60)

    fig_b, axes_b = plt.subplots(1, 2, figsize=(14, 5))

    # Left: F_kappa CDFs for all envs
    ax = axes_b[0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    env_results = []

    for idx, env in enumerate(envs):
        nS, nA, gamma = env.n_states, env.n_actions, env.gamma
        P_s, R_s = env.build(theta_s)
        Q_s, V_s, pi_s = value_iteration(P_s, R_s, gamma)
        gaps = get_action_gaps(Q_s, pi_s, nS, nA)

        # F_kappa CDF
        sorted_g = np.sort(gaps[gaps < np.inf])
        x_vals = np.linspace(0, max(sorted_g)*1.2, 200)
        cdf_vals = [np.mean(gaps <= x) for x in x_vals]
        ax.plot(x_vals, cdf_vals, lw=2.5, color=colors[idx], label=env.name)

        # Steepness near origin: F_kappa(median_gap/2)
        median_gap = np.median(gaps[gaps < np.inf])
        steepness = np.mean(gaps <= median_gap/4)  # fraction with very small gaps

        # Transfer degradation curve
        transfer_gaps = []
        for theta_t in thetas_range:
            P_t, R_t = env.build(theta_t)
            Q_t, V_t, pi_t = value_iteration(P_t, R_t, gamma)
            V_pi_s = np.zeros(nS)
            P_pi = np.zeros((nS, nS))
            R_pi = np.zeros(nS)
            for s in range(nS):
                P_pi[s,:] = P_t[s, pi_s[s], :]
                R_pi[s] = R_t[s, pi_s[s]]
            V_pi_s = np.linalg.solve(np.eye(nS) - gamma*P_pi, R_pi)
            rho = np.ones(nS) / nS
            tg = rho @ V_t - rho @ V_pi_s
            transfer_gaps.append(max(tg, 0))

        env_results.append({
            'name': env.name,
            'steepness': steepness,
            'transfer_gaps': transfer_gaps,
            'color': colors[idx],
        })
        print(f"  {env.name}: steepness (F_κ at median/4) = {steepness:.3f}, "
              f"max transfer gap = {max(transfer_gaps):.4f}")

    ax.set_xlabel('$x$')
    ax.set_ylabel('$F_\\kappa(x)$')
    ax.set_title('(a) Action Gap CDF $F_\\kappa$')
    ax.legend()
    ax.set_ylim(-0.02, 1.05)

    # Right: Transfer gap curves
    ax = axes_b[1]
    for r in env_results:
        ax.plot(thetas_range, r['transfer_gaps'], lw=2.5, color=r['color'],
                label=f"{r['name']} (steep={r['steepness']:.2f})")
    ax.set_xlabel(r'$\|\Delta\theta\|$')
    ax.set_ylabel('Transfer gap')
    ax.set_title('(b) Transfer Gap Scaling')
    ax.legend(fontsize=9)

    plt.suptitle('$F_\\kappa$ Shape Predicts Transfer Degradation Pattern',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'exp13_fkappa_prediction.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'exp13_fkappa_prediction.png'), bbox_inches='tight')
    plt.close()
    print(f"\nSaved figures to {save_dir}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_experiment()
