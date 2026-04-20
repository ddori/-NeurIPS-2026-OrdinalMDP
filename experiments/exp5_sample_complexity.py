"""
Experiment 5: Sample Complexity
================================
Validates Theorems 8 (discrete) and 9 (continuous).

(a) Discrete: Majority-vote ordinal policy converges with K ~ 1/(2*kappa_min^2) * log(...)
(b) Discrete: Transfer gap decreases with K
(c) Robustness to noisy Q-estimates (Corollary 2)
(d)-(f) Continuous: Mean-action converges at rate O(1/K), matches theory bound
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
from scipy import linalg

SEED = 42
np.random.seed(SEED)


# ─── Discrete MDP with controlled ranking disagreements ───

class DisagreementMDP:
    """
    6-state, 3-action MDP parameterized by theta in [-1, 1].
    States 0, 2, 4 have theta-dependent rankings (action preference flips).
    States 1, 3, 5 have stable rankings.
    All actions lead to the SAME transitions (differ only in reward),
    so Q-value rankings are determined purely by immediate rewards.
    """
    def __init__(self, gamma=0.9):
        self.n_states = 6
        self.n_actions = 3
        self.gamma = gamma

    def build(self, theta):
        nS, nA = self.n_states, self.n_actions
        P = np.zeros((nS, nA, nS))
        R = np.zeros((nS, nA))

        # SHARED transitions for all actions (rewards-only differentiation)
        # State 0 -> {1, 2} uniformly
        for a in range(nA):
            P[0, a, 1] = 0.5; P[0, a, 2] = 0.5
        # State 1 -> {3, 4}
        for a in range(nA):
            P[1, a, 3] = 0.6; P[1, a, 4] = 0.4
        # State 2 -> {4, 5}
        for a in range(nA):
            P[2, a, 4] = 0.5; P[2, a, 5] = 0.5
        # State 3 -> {5}
        for a in range(nA):
            P[3, a, 5] = 1.0
        # State 4 -> {5}
        for a in range(nA):
            P[4, a, 5] = 1.0
        # State 5: terminal
        for a in range(nA):
            P[5, a, 5] = 1.0

        # REWARDS: theta controls which action is best
        # State 0: action 0 vs 1 flip at theta=0 (gap = 1.0*|theta|)
        R[0, 0] = 1.0 + 1.0 * theta
        R[0, 1] = 1.0 - 1.0 * theta
        R[0, 2] = 0.2

        # State 1: action 0 always best (large gap, stable)
        R[1, 0] = 1.5
        R[1, 1] = 0.5
        R[1, 2] = 0.3

        # State 2: action 1 vs 0 flip at theta=0 (gap = 0.8*|theta|)
        R[2, 0] = 0.8 - 0.8 * theta
        R[2, 1] = 0.8 + 0.8 * theta
        R[2, 2] = 0.1

        # State 3: action 2 always best (stable)
        R[3, 0] = 0.3
        R[3, 1] = 0.4
        R[3, 2] = 1.2

        # State 4: action 0 vs 1 flip at theta=0 (gap = 0.6*|theta|)
        R[4, 0] = 0.6 + 0.6 * theta
        R[4, 1] = 0.6 - 0.6 * theta
        R[4, 2] = 0.05

        # State 5: terminal (clear ranking to avoid noise sensitivity)
        R[5, :] = [0.5, 0.2, 0.05]

        # Light transition noise for realism
        for s in range(nS):
            for a in range(nA):
                P[s, a, :] = 0.97 * P[s, a, :] + 0.03 / nS
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
    V = np.linalg.solve(np.eye(nS) - gamma * P_pi, R_pi)
    return V


def majority_vote_policy(Q_list):
    """Plurality majority vote: at each state, pick action most often ranked first."""
    K = len(Q_list)
    nS, nA = Q_list[0].shape
    Qs = np.array(Q_list)  # (K, nS, nA)
    policy = np.zeros(nS, dtype=int)
    for s in range(nS):
        # Count how many sources select each action as best
        best_actions = Qs[:, s, :].argmax(axis=1)  # (K,)
        votes = np.bincount(best_actions, minlength=nA)
        policy[s] = votes.argmax()
    return policy


def run_discrete_experiment():
    print("=== Discrete Sample Complexity ===")
    env = DisagreementMDP(gamma=0.9)
    nS, nA, gamma = env.n_states, env.n_actions, env.gamma

    # Source distribution: theta ~ Uniform(-0.3, 0.7)
    # ~30% of sources have theta < 0 (minority), ~70% have theta > 0 (majority)
    # This creates genuine ranking disagreement at flip states
    n_pool = 800
    thetas_pool = np.random.uniform(-0.3, 0.7, n_pool)

    Q_pool = []
    pi_pool = []
    for th in thetas_pool:
        P, R = env.build(th)
        Q, V, pi = value_iteration(P, R, gamma)
        Q_pool.append(Q)
        pi_pool.append(pi)

    # ── Population ordinal policy (ground truth with all sources) ──
    pi_pop = majority_vote_policy(Q_pool)
    print(f"  Population ordinal policy: {pi_pop}")

    # ── Compute ordinal margin kappa_min ──
    Qs = np.array(Q_pool)
    kappa_min = np.inf
    for s in range(nS):
        for a in range(nA):
            for ap in range(a+1, nA):
                prob = np.mean(Qs[:, s, a] > Qs[:, s, ap])
                kappa = abs(prob - 0.5)
                if kappa > 0.005:  # skip exact ties
                    kappa_min = min(kappa_min, kappa)
    print(f"  kappa_min = {kappa_min:.4f}")

    # ── Fixed evaluation targets ──
    n_targets = 50
    thetas_tgt = np.random.uniform(-0.3, 0.7, n_targets)
    target_data = []
    for th in thetas_tgt:
        P, R = env.build(th)
        Q, V, pi = value_iteration(P, R, gamma)
        target_data.append((P, R, Q, V, pi))

    # ── Sweep K ──
    K_values = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 350, 500]
    n_trials = 200
    rho = np.ones(nS) / nS

    accuracy_stats = []
    gap_stats = []

    for K in K_values:
        accs = []
        gaps = []
        for trial in range(n_trials):
            indices = np.random.choice(n_pool, K, replace=True)
            Q_sampled = [Q_pool[i] for i in indices]
            pi_mv = majority_vote_policy(Q_sampled)

            # Accuracy: compare to population ordinal policy
            acc = np.mean(pi_mv == pi_pop)
            accs.append(acc)

            # Transfer gap: average over fixed targets
            trial_gap = 0
            for P_t, R_t, Q_t, V_t, pi_t in target_data[:10]:
                V_mv = policy_eval(P_t, R_t, gamma, pi_mv)
                trial_gap += max(rho @ V_t - rho @ V_mv, 0)
            gaps.append(trial_gap / 10)

        accuracy_stats.append((np.mean(accs), np.std(accs) / np.sqrt(n_trials)))
        gap_stats.append((np.mean(gaps), np.std(gaps) / np.sqrt(n_trials)))
        print(f"  K={K:4d}: accuracy={np.mean(accs):.4f} ± {np.std(accs):.4f}, "
              f"gap={np.mean(gaps):.6f}")

    # Theoretical K*
    delta = 0.05
    K_theory = max(1, 1 / (2 * kappa_min**2) * np.log(nS * nA * (nA - 1) / 2 / delta))
    print(f"  K* (theory) = {K_theory:.0f}")

    # ── Noisy Q robustness (Corollary 2) ──
    # Use sources with |theta| >= 0.15 so Q-gaps are well-defined
    clear_indices = [i for i, th in enumerate(thetas_pool) if abs(th) >= 0.15]
    noise_levels = np.array([0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8])
    K_fixed = 80
    n_noise_trials = 500
    noisy_results = []
    for eps in noise_levels:
        accs = []
        for trial in range(n_noise_trials):
            indices = np.random.choice(clear_indices, K_fixed, replace=True)
            Q_noisy = [Q_pool[i] + eps * np.random.randn(nS, nA) for i in indices]
            pi_mv = majority_vote_policy(Q_noisy)
            acc = np.mean(pi_mv == pi_pop)
            accs.append(acc)
        noisy_results.append((np.mean(accs), np.std(accs) / np.sqrt(n_noise_trials)))

    # Compute min Q-gap among clear sources (|theta| >= 0.15)
    min_gap = np.inf
    for i in clear_indices[:200]:
        Q = Q_pool[i]
        for s in range(nS):
            a_star = Q[s].argmax()
            for a in range(nA):
                if a != a_star and Q[s, a_star] - Q[s, a] > 0:
                    min_gap = min(min_gap, Q[s, a_star] - Q[s, a])
    print(f"  min Q-gap (|θ|≥0.15) = {min_gap:.4f}, g_min/2 = {min_gap/2:.4f}")

    # ── Model-Free kappa_hat estimation (Proposition: Model-Free Certificate) ──
    print("\n  --- Model-Free kappa estimation ---")
    delta_mf = 0.05
    K_mf_values = [3, 5, 10, 20, 50, 100, 200, 500]
    n_mf_trials = 300
    kappa_hat_stats = []
    for K_mf in K_mf_values:
        kappas = []
        for trial in range(n_mf_trials):
            indices = np.random.choice(n_pool, K_mf, replace=True)
            # Compute empirical pairwise agreement
            eta_min = np.inf
            for s in range(nS):
                for a in range(nA):
                    for ap in range(a+1, nA):
                        p_hat = np.mean([Q_pool[i][s, a] > Q_pool[i][s, ap]
                                         for i in indices])
                        eta = abs(p_hat - 0.5)
                        if eta < eta_min:
                            eta_min = eta
            # kappa_hat = eta_min - sqrt(log(2*|S|*C(|A|,2)/delta) / (2K))
            n_pairs = nS * nA * (nA - 1) // 2
            correction = np.sqrt(np.log(2 * n_pairs / delta_mf) / (2 * K_mf))
            kappa_hat = eta_min - correction
            kappas.append(kappa_hat)
        kappa_hat_stats.append((np.mean(kappas), np.std(kappas),
                                np.mean([k > 0 for k in kappas])))
        print(f"    K={K_mf:4d}: kappa_hat={np.mean(kappas):.4f} ± {np.std(kappas):.4f}, "
              f"P(kappa_hat>0)={np.mean([k > 0 for k in kappas]):.2f}")
    print(f"    True kappa_min = {kappa_min:.4f}")

    return {
        'K_values': K_values,
        'accuracy_stats': accuracy_stats,
        'gap_stats': gap_stats,
        'kappa_min': kappa_min,
        'K_theory': K_theory,
        'noise_levels': noise_levels,
        'noisy_results': noisy_results,
        'min_q_gap': min_gap,
        'K_mf_values': K_mf_values,
        'kappa_hat_stats': kappa_hat_stats,
    }


# ─── Continuous: Analytical LQR Mean-Action ───

class SimpleLQR:
    def __init__(self, gamma=0.95):
        self.n_x = 2
        self.n_u = 1
        self.gamma = gamma
        self.Q_cost = np.eye(2)
        self.R_cost = 0.1 * np.eye(1)
        self.dt = 0.1

    def get_dynamics(self, theta):
        k = 2.0 + theta[0]
        c = 0.5 + theta[1]
        Ac = np.array([[0, 1], [-k, -c]])
        Bc = np.array([[0], [1]])
        A = np.eye(2) + self.dt * Ac
        B = self.dt * Bc
        return A, B

    def optimal_gain(self, theta):
        A, B = self.get_dynamics(theta)
        gamma = self.gamma
        Ag = np.sqrt(gamma) * A
        Bg = np.sqrt(gamma) * B
        try:
            P = linalg.solve_discrete_are(Ag, Bg, self.Q_cost, self.R_cost)
        except Exception:
            P = np.eye(2) * 10
            for _ in range(5000):
                P_new = self.Q_cost + gamma * A.T @ P @ A - \
                    gamma**2 * A.T @ P @ B @ np.linalg.solve(
                        self.R_cost + gamma * B.T @ P @ B, B.T @ P @ A)
                if np.max(np.abs(P_new - P)) < 1e-12:
                    break
                P = P_new
        K = gamma * np.linalg.solve(self.R_cost + gamma * B.T @ P @ B, B.T @ P @ A)
        return K, P

    def optimal_action(self, theta, x):
        K, _ = self.optimal_gain(theta)
        return (-K @ x).flatten()


def run_continuous_experiment():
    print("\n=== Continuous Sample Complexity ===")
    lqr = SimpleLQR(gamma=0.95)

    mu_mean = np.array([0.0, 0.0])
    mu_std = 0.3

    # Test states
    test_states = [np.array([1.0, 0.5]), np.array([0.5, -1.0]),
                   np.array([2.0, 0.0]), np.array([-1.0, 1.0]),
                   np.array([1.5, 0.5]), np.array([-0.5, 1.5])]

    # Large source pool
    n_pool = 1500
    thetas_pool = mu_mean + mu_std * np.random.randn(n_pool, 2)

    # Compute optimal actions for all sources at all test states
    actions_pool = {}
    for xi, x in enumerate(test_states):
        actions_pool[xi] = np.array([lqr.optimal_action(th, x) for th in thetas_pool])

    # Population mean action (ground truth for convergence)
    pop_mean_actions = {}
    for xi in range(len(test_states)):
        pop_mean_actions[xi] = actions_pool[xi].mean(axis=0)

    # Hessian at population center
    _, P_nom = lqr.optimal_gain(mu_mean)
    _, B = lqr.get_dynamics(mu_mean)
    H = lqr.R_cost + lqr.gamma * B.T @ P_nom @ B
    L_aa = float(np.max(np.linalg.eigvalsh(H)))
    mu_sc = float(np.min(np.linalg.eigvalsh(H)))
    print(f"  L_aa = {L_aa:.4f}, mu = {mu_sc:.4f}")

    K_values = [1, 2, 3, 5, 8, 15, 30, 50, 100, 200, 500, 1000]
    n_trials = 300

    mse_stats = []
    gap_stats = []

    for K in K_values:
        mses = []
        gaps = []
        for trial in range(n_trials):
            src_idx = np.random.choice(n_pool, K, replace=True)

            total_mse = 0
            total_gap = 0
            for xi, x in enumerate(test_states):
                # Mean action from K sources
                a_bar = actions_pool[xi][src_idx].mean(axis=0)
                # Compare to population mean
                da_pop = (a_bar - pop_mean_actions[xi]).reshape(-1, 1)
                total_mse += (da_pop.T @ da_pop).item()
                total_gap += (0.5 * da_pop.T @ H @ da_pop).item()

            mses.append(total_mse / len(test_states))
            gaps.append(total_gap / len(test_states))

        mse_stats.append((np.mean(mses), np.std(mses) / np.sqrt(n_trials)))
        gap_stats.append((np.mean(gaps), np.std(gaps) / np.sqrt(n_trials)))
        print(f"  K={K:4d}: MSE={np.mean(mses):.6f}, gap={np.mean(gaps):.6f}")

    # Action variance and diameter for theory bound
    all_actions = actions_pool[0]
    pop_mean = all_actions.mean(axis=0)
    action_var = np.mean(np.linalg.norm(all_actions - pop_mean, axis=1)**2)
    D = np.max(np.linalg.norm(all_actions - pop_mean, axis=1))
    print(f"  Action variance = {action_var:.6f}, D = {D:.4f}")

    # Theory: MSE ~ sigma^2/K, gap ~ L_aa/2 * sigma^2/K
    # where sigma^2 is the per-source action variance
    theory_mse = [action_var / K for K in K_values]
    theory_gap = [L_aa / 2 * action_var / K for K in K_values]

    return {
        'K_values': K_values,
        'mse_stats': mse_stats,
        'gap_stats': gap_stats,
        'theory_mse': theory_mse,
        'theory_gap': theory_gap,
        'L_aa': L_aa,
        'mu_sc': mu_sc,
        'action_var': action_var,
        'D': D,
    }


def plot_results(disc, cont, save_dir='../figures'):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # ─── Discrete (a): Accuracy vs K ───
    ax = axes[0, 0]
    K = disc['K_values']
    acc_m = [s[0] for s in disc['accuracy_stats']]
    acc_se = [s[1] for s in disc['accuracy_stats']]
    ax.errorbar(K, acc_m, yerr=acc_se, fmt='bo-', ms=5, lw=2, capsize=3,
                label='Majority vote')
    ax.axhline(1.0, color='green', ls='--', alpha=0.5)
    ax.axvline(disc['K_theory'], color='red', ls=':', lw=2,
               label=f'Theory $K^* \\approx {disc["K_theory"]:.0f}$')
    ax.set_xlabel('$K$ (source environments)')
    ax.set_ylabel('Policy accuracy')
    ax.set_title('(a) Discrete: Majority Vote Accuracy')
    ax.legend(fontsize=9)
    ax.set_xscale('log')

    # ─── Discrete (b): Model-Free kappa estimation ───
    ax = axes[0, 1]
    K_mf = disc['K_mf_values']
    khat_m = [s[0] for s in disc['kappa_hat_stats']]
    khat_std = [s[1] for s in disc['kappa_hat_stats']]
    ax.errorbar(K_mf, khat_m, yerr=khat_std, fmt='mo-', ms=5, lw=2, capsize=3,
                label=r'$\hat{\kappa}$ (Prop. model-free)')
    ax.axhline(disc['kappa_min'], color='green', ls='--', lw=2,
               label=f'True $\\kappa_{{\\min}} = {disc["kappa_min"]:.3f}$')
    ax.axhline(0, color='gray', ls=':', lw=1)
    ax.set_xlabel('$K$ (source environments)')
    ax.set_ylabel(r'Estimated $\hat{\kappa}$')
    ax.set_title('(b) Model-Free Margin Estimate')
    ax.legend(fontsize=9)
    ax.set_xscale('log')

    # ─── Discrete (c): Noisy Q robustness ───
    ax = axes[0, 2]
    noise = disc['noise_levels']
    noisy_m = [r[0] for r in disc['noisy_results']]
    noisy_se = [r[1] for r in disc['noisy_results']]
    ax.errorbar(noise, noisy_m, yerr=noisy_se, fmt='gs-', ms=5, lw=2, capsize=3)
    g_half = disc['min_q_gap'] / 2
    ax.axvline(g_half, color='red', ls=':', lw=2,
               label=f'$g_{{\\min}}/2 = {g_half:.3f}$')
    ax.set_xlabel(r'Q-estimate noise $\epsilon_Q$')
    ax.set_ylabel('Policy accuracy')
    ax.set_title('(c) Robustness to Noisy Q (Cor. 2)')
    ax.legend(fontsize=9)

    # ─── Continuous (d): MSE vs K ───
    ax = axes[1, 0]
    K_c = cont['K_values']
    mse_m = [s[0] for s in cont['mse_stats']]
    ax.loglog(K_c, mse_m, 'bo-', ms=5, lw=2, label='Empirical MSE')
    ax.loglog(K_c, cont['theory_mse'], 'r--', lw=2, label='$\\sigma^2/K$ (theory)')
    ax.set_xlabel('$K$')
    ax.set_ylabel('MSE: $\\|\\bar{a} - \\mu_a\\|^2$')
    ax.set_title('(d) Continuous: Action MSE vs $K$')
    ax.legend(fontsize=9)

    # ─── Continuous (e): Gap vs K ───
    ax = axes[1, 1]
    gap_m_c = [s[0] for s in cont['gap_stats']]
    gap_se_c = [s[1] for s in cont['gap_stats']]
    ax.errorbar(K_c, gap_m_c, yerr=gap_se_c, fmt='bo-', ms=5, lw=2, capsize=3,
                label='Empirical gap')
    ax.plot(K_c, cont['theory_gap'], 'r--', lw=2, label='Thm 9 bound')
    ax.set_xlabel('$K$')
    ax.set_ylabel('Transfer gap')
    ax.set_title('(e) Continuous: Gap vs $K$ (Thm 9)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=9)

    # ─── Continuous (f): Decomposition ───
    ax = axes[1, 2]
    # At finite K: gap ≈ L_aa/2 * sigma^2/K
    # Show empirical gap vs theory, and highlight 1/K scaling
    ax.loglog(K_c, gap_m_c, 'bo-', ms=5, lw=2, label='Total gap')
    ax.loglog(K_c, cont['theory_gap'], 'r--', lw=2, label='$\\frac{L_{aa}}{2} \\cdot \\frac{\\sigma^2}{K}$')
    # Show slope reference
    c_ref = gap_m_c[3] * K_c[3]
    ax.loglog(K_c, [c_ref / k for k in K_c], 'k:', lw=1.5, alpha=0.5, label='$O(1/K)$ slope')
    ax.set_xlabel('$K$')
    ax.set_ylabel('Transfer gap')
    ax.set_title('(f) Gap Scaling: $O(1/K)$')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'exp5_sample_complexity.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'exp5_sample_complexity.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved exp5 figures to {save_dir}")


if __name__ == '__main__':
    disc = run_discrete_experiment()
    cont = run_continuous_experiment()
    plot_results(disc, cont)
