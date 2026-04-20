"""
Experiment 12: Tightness of F_kappa bound across discount factors
=================================================================
Tests how tight the violation growth bound F_kappa(2L_Q||Δθ||) is
for different values of γ ∈ {0, 0.5, 0.9, 0.95, 0.99}.

Uses a chain MDP designed to produce diverse action gaps (like Prop 1 construction),
so violations accumulate gradually and tightness is meaningful.

Answers reviewer Q2: "Can you show empirical tightness for γ > 0?"
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


def build_chain_mdp(n_states, gamma, theta):
    """
    Chain MDP with diverse action gaps.
    n_states states, 3 actions. Action gaps at θ=0 are linearly spaced.
    θ shifts the reward structure, causing orderings to flip progressively.
    """
    nS, nA = n_states, 3
    # Transition: action 0 = stay, action 1 = right, action 2 = left
    P = np.zeros((nS, nA, nS))
    for s in range(nS):
        # Stay
        P[s, 0, s] = 0.8
        P[s, 0, min(s+1, nS-1)] = 0.1
        P[s, 0, max(s-1, 0)] = 0.1
        # Right
        P[s, 1, min(s+1, nS-1)] = 0.8
        P[s, 1, s] = 0.15
        P[s, 1, max(s-1, 0)] = 0.05
        # Left
        P[s, 2, max(s-1, 0)] = 0.8
        P[s, 2, s] = 0.15
        P[s, 2, min(s+1, nS-1)] = 0.05

    # Reward: designed so action gaps at θ=0 are linearly spaced
    # kappa(s) = gap_base[s], and θ shifts rewards to flip orderings
    R = np.zeros((nS, nA))
    gap_base = np.linspace(0.05, 1.0, nS)  # smallest to largest gap

    for s in range(nS):
        # At θ=0: action 0 is best with gap gap_base[s]
        R[s, 0] = gap_base[s] / 2 - theta * 1.0  # shrinks with θ
        R[s, 1] = -gap_base[s] / 2 + theta * 1.0  # grows with θ
        R[s, 2] = -gap_base[s]  # always worst

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


def compute_fkappa(kappas, x):
    return np.mean(np.array(kappas) <= x)


def run_experiment():
    gammas = [0.0, 0.5, 0.9, 0.95, 0.99]
    n_states = 20
    theta_s = 0.0
    thetas = np.linspace(0.0, 1.0, 100)

    results = {}

    for gamma in gammas:
        print(f"\n--- γ = {gamma} ---")

        # Source
        P, R_s = build_chain_mdp(n_states, gamma, theta_s)
        Q_s, V_s, pi_s = value_iteration(P, R_s, gamma)

        # Action gaps at source
        kappas = np.zeros(n_states)
        for s in range(n_states):
            q_best = Q_s[s, pi_s[s]]
            gaps = [q_best - Q_s[s, a] for a in range(3) if a != pi_s[s]]
            kappas[s] = min(gaps) if gaps else 0.0

        # Estimate L_Q
        L_Q = 0
        dt = 0.001
        for theta_t in np.linspace(0, 1, 200):
            _, R_t = build_chain_mdp(n_states, gamma, theta_t)
            _, R_t2 = build_chain_mdp(n_states, gamma, theta_t + dt)
            Q_t, _, _ = value_iteration(P, R_t, gamma)
            Q_t2, _, _ = value_iteration(P, R_t2, gamma)
            lip = np.max(np.abs(Q_t2 - Q_t)) / dt
            L_Q = max(L_Q, lip)

        print(f"  L_Q = {L_Q:.4f}")
        print(f"  kappa range: [{min(kappas):.4f}, {max(kappas):.4f}]")

        # Sweep targets
        actual_viols = []
        predicted_viols = []
        dthetas_list = []

        for theta_t in thetas:
            _, R_t = build_chain_mdp(n_states, gamma, theta_t)
            Q_t, V_t, pi_t = value_iteration(P, R_t, gamma)

            n_viol = np.sum(pi_s != pi_t)
            actual_frac = n_viol / n_states
            actual_viols.append(actual_frac)

            dtheta = abs(theta_t - theta_s)
            x = 2 * L_Q * dtheta
            pred_frac = compute_fkappa(kappas, x)
            predicted_viols.append(pred_frac)
            dthetas_list.append(dtheta)

        # Tightness ratio (only where both > 0)
        ratios = []
        for av, pv in zip(actual_viols, predicted_viols):
            if av > 0.01:
                ratios.append(pv / av)

        avg_ratio = np.mean(ratios) if ratios else float('inf')
        max_actual = max(actual_viols)
        max_pred = max(predicted_viols)
        print(f"  Avg tightness ratio: {avg_ratio:.2f}×")
        print(f"  Max violation: actual={max_actual:.3f}, bound={max_pred:.3f}")

        results[gamma] = {
            'dthetas': dthetas_list,
            'actual': actual_viols,
            'predicted': predicted_viols,
            'L_Q': L_Q,
            'kappas': kappas.tolist(),
            'avg_ratio': avg_ratio,
        }

    # ── Plot ──
    save_dir = '../figures'
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, len(gammas), figsize=(4.2 * len(gammas), 4))

    for i, gamma in enumerate(gammas):
        ax = axes[i]
        r = results[gamma]
        dt = r['dthetas']
        ax.step(dt, r['predicted'], where='post', lw=2.5, color='red',
                label='$F_\\kappa(2L_Q\\|\\Delta\\theta\\|)$', zorder=2)
        ax.plot(dt, r['actual'], 'o', ms=3, color='#1f77b4', alpha=0.7,
                label='Actual $|S_{\\mathrm{viol}}|/|S|$', zorder=3)
        ax.fill_between(dt, r['actual'], r['predicted'], alpha=0.12, color='red',
                        step='post')
        ax.set_xlabel(r'$\|\Delta\theta\|$')
        if i == 0:
            ax.set_ylabel('Violation fraction')
        ratio_str = f'{r["avg_ratio"]:.1f}' if r["avg_ratio"] < 100 else '∞'
        ax.set_title(f'$\\gamma = {gamma}$  (ratio: {ratio_str}×)')
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=7, loc='upper left')

    plt.suptitle('Violation Growth Bound Tightness across $\\gamma$\n(20-state chain MDP, 3 actions)',
                 fontsize=13, y=1.04)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'exp12_gamma_tightness.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'exp12_gamma_tightness.png'), bbox_inches='tight')
    plt.close()
    print(f"\nSaved figure to {save_dir}/exp12_gamma_tightness.pdf")

    # Summary
    print(f"\n{'='*50}")
    print(f"  Tightness Summary (20-state chain)")
    print(f"{'='*50}")
    print(f"  {'γ':>6}  {'L_Q':>8}  {'Ratio':>8}  {'MaxViol':>8}  {'MaxBound':>9}")
    for gamma in gammas:
        r = results[gamma]
        ratio_str = f'{r["avg_ratio"]:.2f}' if r["avg_ratio"] < 100 else '∞'
        print(f"  {gamma:6.2f}  {r['L_Q']:8.3f}  {ratio_str:>8}×  "
              f"{max(r['actual']):8.3f}  {max(r['predicted']):9.3f}")

    return results


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_experiment()
