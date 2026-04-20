"""
Experiment 7: Higher-Dimensional Continuous Control Transfer
=============================================================
4-joint robot (8D state, 4D action) modeled as linearized multi-body system.
All computations are ANALYTICAL (no simulation noise).

Validates:
- Quadratic degradation (Theorem 5) in high dimensions
- Action displacement linearity (Theorem 7)
- Multi-source mean-gain scaling (Theorem 9)
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


class MultiJointLQR:
    """
    n-joint robot as LQR: x = [q, q_dot], u = joint torques.
    M(theta) q_ddot + C(theta) q_dot + G(theta) q = u
    All transfer gaps computed analytically via Riccati equations.
    """
    def __init__(self, n_joints=4, gamma=0.99):
        self.n = n_joints
        self.obs_dim = 2 * n_joints
        self.act_dim = n_joints
        self.gamma = gamma
        self.dt = 0.02
        self.Q_cost = np.eye(self.obs_dim)
        self.R_cost = 0.05 * np.eye(self.act_dim)

    def get_dynamics(self, theta):
        """theta = [gravity_scale, friction_scale, mass_scale]."""
        g_scale, f_scale, m_scale = theta
        n = self.n

        # Mass matrix
        M_diag = m_scale * (1.0 + 0.5 * np.arange(n)[::-1] / n)
        M = np.diag(M_diag)
        # Coupling between adjacent joints
        for i in range(n - 1):
            coupling = m_scale * 0.15 * (1 - i / n)
            M[i, i+1] = coupling
            M[i+1, i] = coupling
        M_inv = np.linalg.inv(M)

        # Damping
        C = f_scale * np.diag(0.3 + 0.2 * np.arange(n) / n)

        # Gravity stiffness (linearized about upright)
        G = g_scale * np.diag(9.8 * (1.0 - 0.15 * np.arange(n)))

        # Continuous A, B
        nd = 2 * n
        A_c = np.zeros((nd, nd))
        A_c[:n, n:] = np.eye(n)
        A_c[n:, :n] = -M_inv @ G
        A_c[n:, n:] = -M_inv @ C

        B_c = np.zeros((nd, n))
        B_c[n:, :] = M_inv

        # Euler discretization
        A = np.eye(nd) + self.dt * A_c
        B = self.dt * B_c
        return A, B

    def solve_dare(self, theta):
        """Solve DARE for discounted LQR. Returns P, K."""
        A, B = self.get_dynamics(theta)
        gamma = self.gamma
        A_g = np.sqrt(gamma) * A
        B_g = np.sqrt(gamma) * B

        try:
            P = linalg.solve_discrete_are(A_g, B_g, self.Q_cost, self.R_cost)
        except Exception:
            # Fallback: iterate
            P = 10 * np.eye(self.obs_dim)
            for _ in range(10000):
                S = self.R_cost + gamma * B.T @ P @ B
                P_new = self.Q_cost + gamma * A.T @ P @ A - \
                    gamma**2 * A.T @ P @ B @ np.linalg.solve(S, B.T @ P @ A)
                if np.max(np.abs(P_new - P)) < 1e-12:
                    break
                P = P_new
            P = P_new

        K = gamma * np.linalg.solve(self.R_cost + gamma * B.T @ P @ B, B.T @ P @ A)
        return P, K

    def analytical_transfer_gap(self, theta_s, theta_t, x):
        """
        EXACT transfer gap: V*(x;theta_t) - Q*(x, u_s; theta_t)
        where u_s = -K_s x is the source policy action.

        V*(x) = -x' P_t x
        Q*(x, u) = -(x'Qx + u'Ru + gamma * (Ax+Bu)' P_t (Ax+Bu))

        Gap = x' P_t x - (x'Qx + u_s'Ru_s + gamma*(A_t x + B_t u_s)'P_t(A_t x + B_t u_s))
            where u_s = -K_s x, and A_t, B_t are target dynamics
        """
        P_t, K_t = self.solve_dare(theta_t)
        _, K_s = self.solve_dare(theta_s)
        A_t, B_t = self.get_dynamics(theta_t)

        # Value under target-optimal: V* = -x'P_t x (cost-to-go)
        v_opt = x.T @ P_t @ x

        # Value under source policy: u = -K_s x
        u_s = -K_s @ x
        immediate_cost = x.T @ self.Q_cost @ x + u_s.T @ self.R_cost @ u_s
        x_next = A_t @ x + B_t @ u_s
        future_cost = self.gamma * x_next.T @ P_t @ x_next

        v_source_onestep = immediate_cost + future_cost

        # Gap = V^{pi_s} - V* in cost sense = v_source_onestep - v_opt
        # But we need to be careful: the one-step cost + future uses target P_t for future
        # This is NOT the full policy evaluation. For exact gap, use the matrix formula:

        # Under policy u=-Kx, the closed-loop cost is x' P_K x where
        # P_K solves P_K = Q + K'RK + gamma*(A-BK)' P_K (A-BK)
        A_cl_s = A_t - B_t @ K_s
        # Solve Lyapunov for P_K_s
        P_Ks = self._solve_policy_cost(A_cl_s, K_s)

        gap = x.T @ (P_Ks - P_t) @ x
        return max(float(gap), 0)

    def _solve_policy_cost(self, A_cl, K):
        """Solve P = Q + K'RK + gamma * A_cl' P A_cl."""
        C_mat = self.Q_cost + K.T @ self.R_cost @ K
        # P = C + gamma * A_cl' P A_cl
        # vec(P) = vec(C) + gamma * (A_cl kron A_cl)' vec(P)
        # (I - gamma * A_cl' kron A_cl') vec(P) = vec(C)
        n = self.obs_dim
        I_nn = np.eye(n * n)
        A_kron = np.kron(A_cl.T, A_cl.T)
        vec_P = np.linalg.solve(I_nn - self.gamma * A_kron, C_mat.flatten())
        return vec_P.reshape(n, n)

    def expected_transfer_gap(self, theta_s, theta_t, n_states=100):
        """Average transfer gap over random initial states."""
        gaps = []
        for _ in range(n_states):
            x = 0.3 * np.random.randn(self.obs_dim)
            gaps.append(self.analytical_transfer_gap(theta_s, theta_t, x))
        return np.mean(gaps), np.std(gaps)

    def gain_displacement(self, theta_s, theta_t):
        """||K_t - K_s||_F."""
        _, K_s = self.solve_dare(theta_s)
        _, K_t = self.solve_dare(theta_t)
        return np.linalg.norm(K_t - K_s, 'fro')


def run_experiment():
    print("=== Multi-Joint Robot Transfer (Analytical) ===")
    sys = MultiJointLQR(n_joints=4, gamma=0.99)
    theta0 = np.array([1.0, 1.0, 1.0])

    # ─── (a) Gravity sweep: returns ───
    g_scales = np.linspace(0.5, 2.0, 30)
    gravity_results = {'scale': [], 'gap_mean': [], 'gap_std': [],
                       'disp': [], 'opt_cost': [], 'transfer_cost': []}

    np.random.seed(SEED)
    test_states = [0.3 * np.random.randn(sys.obs_dim) for _ in range(50)]

    _, K_src = sys.solve_dare(theta0)

    for gs in g_scales:
        theta_t = np.array([gs, 1.0, 1.0])
        P_t, K_t = sys.solve_dare(theta_t)
        A_t, B_t = sys.get_dynamics(theta_t)

        # Analytical costs
        A_cl_s = A_t - B_t @ K_src
        P_Ks = sys._solve_policy_cost(A_cl_s, K_src)

        opt_costs = [x.T @ P_t @ x for x in test_states]
        transfer_costs = [x.T @ P_Ks @ x for x in test_states]
        gaps = [max(tc - oc, 0) for tc, oc in zip(transfer_costs, opt_costs)]

        gravity_results['scale'].append(gs)
        gravity_results['gap_mean'].append(np.mean(gaps))
        gravity_results['gap_std'].append(np.std(gaps))
        gravity_results['disp'].append(sys.gain_displacement(theta0, theta_t))
        gravity_results['opt_cost'].append(-np.mean(opt_costs))  # negative = return
        gravity_results['transfer_cost'].append(-np.mean(transfer_costs))

    # ─── (b,c) Directional sweep for quadratic verification ───
    directions = [
        (np.array([1, 0, 0]), 'Gravity'),
        (np.array([0, 1, 0]), 'Friction'),
        (np.array([0, 0, 1]), 'Mass'),
        (np.array([1, 1, 1]) / np.sqrt(3), 'Combined'),
    ]
    magnitudes = np.linspace(0, 0.5, 50)

    quad_data = {}
    for dirn, name in directions:
        gaps = []
        disps = []
        for mag in magnitudes:
            theta_t = theta0 + mag * dirn
            # Ensure positive
            theta_t = np.maximum(theta_t, 0.2)

            P_t, K_t = sys.solve_dare(theta_t)
            A_t, B_t = sys.get_dynamics(theta_t)
            A_cl_s = A_t - B_t @ K_src
            P_Ks = sys._solve_policy_cost(A_cl_s, K_src)

            # Average gap
            state_gaps = [max(float(x.T @ (P_Ks - P_t) @ x), 0) for x in test_states]
            gaps.append(np.mean(state_gaps))
            disps.append(sys.gain_displacement(theta0, theta_t))

        quad_data[name] = {'gaps': gaps, 'disps': disps}

    # ─── (d) 2D heatmap: gravity x friction ───
    n_grid = 20
    g_vals = np.linspace(0.6, 1.5, n_grid)
    f_vals = np.linspace(0.6, 1.5, n_grid)
    gap_heatmap = np.zeros((n_grid, n_grid))

    for i, gv in enumerate(g_vals):
        for j, fv in enumerate(f_vals):
            theta_t = np.array([gv, fv, 1.0])
            P_t, _ = sys.solve_dare(theta_t)
            A_t, B_t = sys.get_dynamics(theta_t)
            A_cl_s = A_t - B_t @ K_src
            P_Ks = sys._solve_policy_cost(A_cl_s, K_src)

            state_gaps = [max(float(x.T @ (P_Ks - P_t) @ x), 0) for x in test_states[:20]]
            gap_heatmap[j, i] = np.mean(state_gaps)

    # ─── (f) Multi-source mean-gain ───
    K_vals = [1, 2, 3, 5, 10, 20, 50, 100]
    n_trials = 80
    ms_results = {}

    for K in K_vals:
        trial_gaps = []
        for trial in range(n_trials):
            # Sample K source params
            thetas_src = theta0 + 0.15 * np.random.randn(K, 3)
            thetas_src = np.maximum(thetas_src, 0.3)

            # Mean gain
            gains = [sys.solve_dare(th)[1] for th in thetas_src]
            K_mean = np.mean(gains, axis=0)

            # Random target
            theta_t = theta0 + 0.2 * np.random.randn(3)
            theta_t = np.maximum(theta_t, 0.3)
            P_t, _ = sys.solve_dare(theta_t)
            A_t, B_t = sys.get_dynamics(theta_t)
            A_cl_mean = A_t - B_t @ K_mean
            P_Km = sys._solve_policy_cost(A_cl_mean, K_mean)

            # Average gap
            sg = [max(float(x.T @ (P_Km - P_t) @ x), 0) for x in test_states[:20]]
            trial_gaps.append(np.mean(sg))

        ms_results[K] = (np.mean(trial_gaps), np.std(trial_gaps) / np.sqrt(n_trials))
        print(f"  K={K:3d}: mean gap = {np.mean(trial_gaps):.6f} +/- {np.std(trial_gaps):.6f}")

    return {
        'gravity_results': gravity_results,
        'quad_data': {k: v for k, v in quad_data.items()},
        'magnitudes': magnitudes.tolist(),
        'gap_heatmap': gap_heatmap.tolist(),
        'g_vals': g_vals.tolist(),
        'f_vals': f_vals.tolist(),
        'ms_results': ms_results,
        'K_vals': K_vals,
        'n_joints': sys.n,
        'directions': [d[1] for d in directions],
    }


def plot_results(results, save_dir='../figures'):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    mags = np.array(results['magnitudes'])

    # (a) Transfer returns vs gravity
    ax = axes[0, 0]
    gr = results['gravity_results']
    ax.plot(gr['scale'], gr['transfer_cost'], 'b-o', ms=3, lw=2, label='Source policy')
    ax.plot(gr['scale'], gr['opt_cost'], 'r--s', ms=3, lw=2, label='Target optimal')
    ax.axvline(1.0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Gravity scale')
    ax.set_ylabel('Expected return (neg. cost)')
    ax.set_title(f'(a) {results["n_joints"]}-Joint Robot: Gravity Transfer')
    ax.legend(fontsize=9)

    # (b) Transfer gap by direction
    ax = axes[0, 1]
    for i, name in enumerate(results['directions']):
        ax.plot(mags, results['quad_data'][name]['gaps'], color=colors[i], lw=2, label=name)
    ax.set_xlabel(r'$\|\Delta\theta\|$')
    ax.set_ylabel('Transfer gap')
    ax.set_title('(b) Transfer Gap by Direction')
    ax.legend(fontsize=9)

    # (c) Log-log quadratic verification
    ax = axes[0, 2]
    for i, name in enumerate(results['directions']):
        gaps = np.array(results['quad_data'][name]['gaps'])
        mask = (mags > 0.02) & (gaps > 1e-10)
        if mask.sum() > 5:
            ax.loglog(mags[mask], gaps[mask], 'o', color=colors[i], ms=3, alpha=0.6)
            # Fit quadratic in small perturbation region
            fit_mask = mask & (mags < 0.3)
            if fit_mask.sum() > 3:
                log_m = np.log(mags[fit_mask])
                log_g = np.log(np.maximum(gaps[fit_mask], 1e-20))
                slope = np.polyfit(log_m, log_g, 1)[0]
                c_fit = np.exp(np.polyfit(log_m, log_g, 1)[1])
                ax.loglog(mags[mask], c_fit * mags[mask]**slope, '--', color=colors[i], lw=2,
                         label=f'{name}: slope={slope:.2f}')
    ax.set_xlabel(r'$\|\Delta\theta\|$ (log)')
    ax.set_ylabel('Transfer gap (log)')
    ax.set_title('(c) Log-Log: Quadratic Verification')
    ax.legend(fontsize=8)

    # (d) 2D heatmap
    ax = axes[1, 0]
    hm = np.array(results['gap_heatmap'])
    im = ax.imshow(hm, origin='lower', aspect='auto',
                   extent=[results['g_vals'][0], results['g_vals'][-1],
                           results['f_vals'][0], results['f_vals'][-1]],
                   cmap='YlOrRd')
    plt.colorbar(im, ax=ax, label='Transfer gap')
    ax.plot(1.0, 1.0, 'w*', ms=15)
    ax.set_xlabel('Gravity scale')
    ax.set_ylabel('Friction scale')
    ax.set_title('(d) 2D Parameter Heatmap')

    # (e) Gain displacement
    ax = axes[1, 1]
    for i, name in enumerate(results['directions']):
        disps = np.array(results['quad_data'][name]['disps'])
        ax.plot(mags, disps, color=colors[i], lw=2, label=name)
    ax.set_xlabel(r'$\|\Delta\theta\|$')
    ax.set_ylabel(r'$\|K_{\theta_t} - K_{\theta_s}\|_F$')
    ax.set_title('(e) Gain Displacement (Linear)')
    ax.legend(fontsize=9)

    # (f) Multi-source scaling
    ax = axes[1, 2]
    K_vals = results['K_vals']
    means = [results['ms_results'][K][0] for K in K_vals]
    stds = [results['ms_results'][K][1] for K in K_vals]
    ax.errorbar(K_vals, means, yerr=stds, fmt='bo-', ms=6, lw=2, capsize=3, label='Empirical')
    # 1/K fit
    # gap = a/K + b (irreducible)
    if len(means) > 3:
        from scipy.optimize import curve_fit
        try:
            def model(K, a, b):
                return a / np.array(K) + b
            popt, _ = curve_fit(model, K_vals, means, p0=[means[0], means[-1]], maxfev=5000)
            K_fit = np.linspace(1, max(K_vals), 200)
            ax.plot(K_fit, model(K_fit, *popt), 'r--', lw=2,
                    label=f'Fit: ${popt[0]:.4f}/K + {popt[1]:.4f}$')
        except:
            c_ref = means[2] * K_vals[2]
            ax.plot(K_vals, [c_ref/K for K in K_vals], 'r--', lw=1.5, label='$O(1/K)$')
    ax.set_xlabel('$K$ (source environments)')
    ax.set_ylabel('Expected transfer gap')
    ax.set_title('(f) Multi-Source Scaling')
    ax.set_xscale('log')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'exp7_mujoco.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'exp7_mujoco.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved exp7 figures to {save_dir}")


if __name__ == '__main__':
    results = run_experiment()
    plot_results(results)
