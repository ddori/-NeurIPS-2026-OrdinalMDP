"""
Experiment 4: Control Environment Transfer
============================================
Top row: Discrete Wind Grid (6x6, 4 actions) — gradual OC decline
Bottom row: Pendulum — analytical LQR transfer (quadratic + linear)
"""

import numpy as np
from scipy import linalg
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


# ============================================================
# Discrete: Wind Grid MDP
# ============================================================

class WindGridMDP:
    """
    6x6 grid, 4 actions (up/down/left/right), goal at top-right.
    theta in [0,1] controls wind strength (pushes down and left).
    As theta increases, optimal actions change state-by-state.
    """
    def __init__(self, size=6, gamma=0.95):
        self.size = size
        self.nS = size * size
        self.nA = 4  # 0=up, 1=down, 2=left, 3=right
        self.gamma = gamma
        self.goal = (0, size - 1)  # top-right corner

    def build(self, theta):
        size = self.size
        nS, nA = self.nS, self.nA
        P = np.zeros((nS, nA, nS))
        R = np.zeros((nS, nA))
        dr = [-1, 1, 0, 0]
        dc = [0, 0, -1, 1]

        for s in range(nS):
            r, c = divmod(s, size)
            R[s, :] = -0.05
            if (r, c) == self.goal:
                R[s, :] = 1.0
                P[s, :, s] = 1.0
                continue

            for a in range(nA):
                nr = max(0, min(size - 1, r + dr[a]))
                nc = max(0, min(size - 1, c + dc[a]))
                intended = nr * size + nc

                # Wind pushes down AND left
                wr = max(0, min(size - 1, r + 1))
                wc = max(0, min(size - 1, c - 1))
                wind_dest = wr * size + wc

                p_wind = theta * 0.55
                p_slip = theta * 0.10  # slip = stay in place
                p_intended = max(1.0 - p_wind - p_slip, 0.05)

                P[s, a, intended] += p_intended
                if wind_dest != intended:
                    P[s, a, wind_dest] += p_wind
                else:
                    P[s, a, intended] += p_wind
                P[s, a, s] += p_slip

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


def run_discrete():
    print("=== Discrete: Wind Grid MDP ===")
    env = WindGridMDP(size=6, gamma=0.95)
    gamma = env.gamma
    nS = env.nS

    # Source: no wind
    theta_src = 0.0
    P_src, R_src = env.build(theta_src)
    Q_src, V_src, pi_src = value_iteration(P_src, R_src, gamma)
    rho = np.ones(nS) / nS

    # Sweep wind strength
    thetas = np.linspace(0, 1.0, 50)
    perf_src = []
    perf_opt = []
    oc_list = []
    sviol_list = []

    for th in thetas:
        P_t, R_t = env.build(th)
        Q_t, V_t, pi_t = value_iteration(P_t, R_t, gamma)

        # Source policy performance at target
        V_pi = policy_eval(P_t, R_t, gamma, pi_src)
        perf_src.append(float(rho @ V_pi))
        perf_opt.append(float(rho @ V_t))

        # Ordinal consistency: fraction of states with same optimal action
        agree = np.mean(pi_src == pi_t)
        # Exclude goal state (always agrees)
        non_goal = [s for s in range(nS) if (s // env.size, s % env.size) != env.goal]
        agree_ng = np.mean([pi_src[s] == pi_t[s] for s in non_goal])
        oc_list.append(agree_ng)

        # Violation states
        sviol = np.sum(pi_src != pi_t)
        sviol_list.append(sviol)

    print(f"  Source perf at theta=0: {perf_src[0]:.4f}")
    print(f"  OC range: {min(oc_list):.3f} to {max(oc_list):.3f}")
    print(f"  S_viol range: {min(sviol_list)} to {max(sviol_list)}")

    return {
        'thetas': thetas, 'perf_src': perf_src, 'perf_opt': perf_opt,
        'oc': oc_list, 'sviol': sviol_list,
        'nS': nS, 'theta_src': theta_src,
    }


# ============================================================
# Continuous: Analytical Pendulum LQR
# ============================================================

class AnalyticalPendulum:
    def __init__(self, dt=0.05, gamma=0.99):
        self.dt = dt
        self.gamma = gamma
        self.Q_cost = np.diag([10.0, 1.0])
        self.R_cost = np.array([[0.1]])

    def get_AB(self, gravity=10.0, mass=1.0, length=1.0, damping=0.1):
        g, m, l, c = gravity, mass, length, damping
        Ac = np.array([[0, 1], [-g / l, -c / (m * l**2)]])
        Bc = np.array([[0], [1 / (m * l**2)]])
        A = np.eye(2) + self.dt * Ac
        B = self.dt * Bc
        return A, B

    def solve_lqr(self, gravity=10.0, mass=1.0, length=1.0, damping=0.1):
        A, B = self.get_AB(gravity, mass, length, damping)
        gamma = self.gamma
        Ag = np.sqrt(gamma) * A
        Bg = np.sqrt(gamma) * B
        try:
            P = linalg.solve_discrete_are(Ag, Bg, self.Q_cost, self.R_cost)
        except Exception:
            P = 100 * np.eye(2)
            for _ in range(10000):
                P_new = self.Q_cost + gamma * A.T @ P @ A - \
                    gamma**2 * A.T @ P @ B @ np.linalg.solve(
                        self.R_cost + gamma * B.T @ P @ B, B.T @ P @ A)
                if np.max(np.abs(P_new - P)) < 1e-12:
                    break
                P = P_new
        K = gamma * np.linalg.solve(self.R_cost + gamma * B.T @ P @ B, B.T @ P @ A)
        return K, P

    def policy_cost_matrix(self, K, gravity=10.0, mass=1.0, length=1.0, damping=0.1):
        A, B = self.get_AB(gravity, mass, length, damping)
        A_cl = A - B @ K
        C = self.Q_cost + K.T @ self.R_cost @ K
        n = 2
        kron = self.gamma * np.kron(A_cl.T, A_cl.T)
        vec_P = np.linalg.solve(np.eye(n * n) - kron, C.flatten())
        return vec_P.reshape(n, n)

    def transfer_gap_analytical(self, K_src, gravity_t, mass_t=1.0,
                                 length_t=1.0, damping_t=0.1, x0_list=None):
        K_opt, _ = self.solve_lqr(gravity_t, mass_t, length_t, damping_t)
        P_src = self.policy_cost_matrix(K_src, gravity_t, mass_t, length_t, damping_t)
        P_opt = self.policy_cost_matrix(K_opt, gravity_t, mass_t, length_t, damping_t)
        dP = P_src - P_opt
        if x0_list is None:
            x0_list = [np.array([np.cos(a), np.sin(a)])
                       for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
        gaps = [float(x0 @ dP @ x0) for x0 in x0_list]
        return np.mean(gaps), K_opt


def run_pendulum():
    print("\n=== Pendulum: Analytical LQR Transfer ===")
    pend = AnalyticalPendulum()
    g_src = 10.0
    K_src, P_src = pend.solve_lqr(gravity=g_src)
    print(f"  Source gain: K = [{K_src[0, 0]:.4f}, {K_src[0, 1]:.4f}]")

    x0_list = [np.array([np.cos(a), np.sin(a)])
               for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]

    gravities = np.linspace(3.0, 20.0, 60)
    gaps = []
    disps = []
    K_opts = []
    src_returns = []
    opt_returns = []

    for g in gravities:
        gap, K_opt = pend.transfer_gap_analytical(K_src, gravity_t=g, x0_list=x0_list)
        gaps.append(max(gap, 0))
        disps.append(np.linalg.norm(K_opt - K_src))
        K_opts.append(K_opt)
        P_s = pend.policy_cost_matrix(K_src, gravity=g)
        P_o = pend.policy_cost_matrix(K_opt, gravity=g)
        src_returns.append(-np.mean([float(x0 @ P_s @ x0) for x0 in x0_list]))
        opt_returns.append(-np.mean([float(x0 @ P_o @ x0) for x0 in x0_list]))

    delta_g = np.abs(gravities - g_src)
    mask = delta_g > 0.5
    gaps_arr = np.array(gaps)
    valid = mask & (gaps_arr > 1e-8)
    if valid.sum() > 3:
        coeffs = np.polyfit(np.log10(delta_g[valid]), np.log10(gaps_arr[valid]), 1)
        print(f"  Log-log slope (gap vs |Δg|): {coeffs[0]:.3f}")
    disps_arr = np.array(disps)
    if valid.sum() > 3:
        coeffs_d = np.polyfit(np.log10(delta_g[valid]),
                              np.log10(disps_arr[valid] + 1e-15), 1)
        print(f"  Log-log slope (displacement vs |Δg|): {coeffs_d[0]:.3f}")

    return {
        'gravities': gravities, 'g_src': g_src,
        'gaps': gaps, 'disps': disps,
        'src_returns': src_returns, 'opt_returns': opt_returns,
        'delta_g': delta_g,
    }


# ============================================================
# Plotting
# ============================================================

def plot_results(disc, pend, save_dir='../figures'):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # ─── Discrete (a): Performance ───
    ax = axes[0, 0]
    ax.plot(disc['thetas'], disc['perf_src'], 'b-', lw=2, label='Source policy')
    ax.plot(disc['thetas'], disc['perf_opt'], 'r--', lw=2, label='Target-optimal')
    ax.axvline(disc['theta_src'], color='gray', ls='--', alpha=0.5)
    ax.fill_between(disc['thetas'],
                     disc['perf_src'], disc['perf_opt'],
                     alpha=0.15, color='red', label='Transfer gap')
    ax.set_xlabel(r'Wind strength $\theta$')
    ax.set_ylabel(r'$V^{\pi}(\rho)$')
    ax.set_title('(a) Wind Grid: Performance')
    ax.legend(fontsize=9)

    # ─── Discrete (b): Ordinal Consistency ───
    ax = axes[0, 1]
    ax.plot(disc['thetas'], disc['oc'], 'g-', lw=2.5)
    ax.axhline(0.7, color='orange', ls=':', alpha=0.5, label='70% threshold')
    ax.axhline(1.0, color='gray', ls='--', alpha=0.3)
    ax.fill_between(disc['thetas'], 0.7, disc['oc'],
                     where=[o >= 0.7 for o in disc['oc']],
                     alpha=0.1, color='green')
    ax.set_xlabel(r'Wind strength $\theta$')
    ax.set_ylabel('Ordinal Consistency (OC)')
    ax.set_title('(b) Wind Grid: Ordinal Consistency')
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=9)

    # ─── Discrete (c): Violation States ───
    ax = axes[0, 2]
    ax.plot(disc['thetas'], disc['sviol'], 'darkorange', lw=2.5)
    ax.axhline(disc['nS'], color='gray', ls='--', alpha=0.3,
               label=f'$|\\mathcal{{S}}| = {disc["nS"]}$')
    ax.set_xlabel(r'Wind strength $\theta$')
    ax.set_ylabel(r'$|\mathcal{S}_{\mathrm{viol}}|$')
    ax.set_title(r'(c) Wind Grid: Violation States $|\mathcal{S}_{\mathrm{viol}}|$')
    ax.legend(fontsize=9)

    # ─── Pendulum (d): Gravity Transfer ───
    ax = axes[1, 0]
    ax.plot(pend['gravities'], pend['src_returns'], 'b-', lw=2, label='Source policy')
    ax.plot(pend['gravities'], pend['opt_returns'], 'r--', lw=2, label='Target-optimal')
    ax.axvline(pend['g_src'], color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Target gravity (m/s$^2$)')
    ax.set_ylabel('Return ($-$cost)')
    ax.set_title('(d) Pendulum: Gravity Transfer')
    ax.legend(fontsize=9)

    # ─── Pendulum (e): Quadratic Degradation ───
    ax = axes[1, 1]
    dg = np.array(pend['delta_g'])
    gps = np.array(pend['gaps'])
    mask = dg > 0.3
    valid = mask & (gps > 1e-8)
    ax.scatter(dg[mask], gps[mask], c='steelblue', s=20, alpha=0.7, zorder=3,
               label='Transfer gap')
    if valid.sum() > 3:
        alpha_fit = np.polyfit(dg[valid]**2, gps[valid], 1)[0]
        xf = np.linspace(0, dg.max(), 100)
        ax.plot(xf, alpha_fit * xf**2, 'r--', lw=2,
                label=f'$\\alpha|\\Delta g|^2$ ($\\alpha$={alpha_fit:.4f})')
    ax.set_xlabel(r'$|\Delta g|$ (m/s$^2$)')
    ax.set_ylabel('Transfer gap')
    ax.set_title('(e) Pendulum: Quadratic Degradation')
    ax.legend(fontsize=9)

    # ─── Pendulum (f): Action Displacement ───
    ax = axes[1, 2]
    disps = np.array(pend['disps'])
    ax.scatter(dg, disps, c='coral', s=20, alpha=0.7, zorder=3,
               label=r'$\|K_t - K_s\|$')
    if mask.sum() > 3:
        sl = np.polyfit(dg[mask], disps[mask], 1)[0]
        xf = np.linspace(0, dg.max(), 100)
        ax.plot(xf, sl * xf, 'g--', lw=2, label=f'Linear (slope={sl:.4f})')
    ax.set_xlabel(r'$|\Delta g|$ (m/s$^2$)')
    ax.set_ylabel('Gain displacement $\\|K_t - K_s\\|$')
    ax.set_title('(f) Pendulum: Action Sensitivity')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'exp4_gymnasium.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'exp4_gymnasium.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved exp4 figures to {save_dir}")


if __name__ == '__main__':
    disc = run_discrete()
    pend = run_pendulum()
    plot_results(disc, pend)
