"""
Experiment 8: Model-Agnostic Transfer (DQN)
=============================================
Shows that Ordinal MDP theory applies to NEURAL policies.

4-action CartPole (forces: -10, -3, +3, +10) to give richer rankings.
OC measured as directional agreement (force sign) for smooth curves.
Scale-invariance benefits from 4 actions (richer ranking = more Q-Avg sensitivity).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
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
import warnings
warnings.filterwarnings('ignore')

SEED = 42
N_ACTIONS = 4
FORCES = [-10.0, -3.0, 3.0, 10.0]


# ============================================================
# CartPole Environment (4 actions)
# ============================================================

class CartPoleEnv:
    def __init__(self, gravity=9.8, masscart=1.0, masspole=0.1, length=0.5):
        self.gravity = gravity
        self.masscart = masscart
        self.masspole = masspole
        self.total_mass = masscart + masspole
        self.length = length
        self.polemass_length = masspole * length
        self.tau = 0.02
        self.theta_threshold = 12 * np.pi / 180
        self.x_threshold = 2.4
        self.state = None

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.state = np.random.uniform(-0.05, 0.05, 4)
        return self.state.copy()

    def step(self, action):
        x, xd, th, thd = self.state
        force = FORCES[action]
        costh = np.cos(th)
        sinth = np.sin(th)
        temp = (force + self.polemass_length * thd**2 * sinth) / self.total_mass
        thacc = (self.gravity * sinth - costh * temp) / \
                (self.length * (4.0/3.0 - self.masspole * costh**2 / self.total_mass))
        xacc = temp - self.polemass_length * thacc * costh / self.total_mass
        self.state = np.array([x + self.tau*xd, xd + self.tau*xacc,
                               th + self.tau*thd, thd + self.tau*thacc])
        done = abs(self.state[0]) > self.x_threshold or abs(self.state[2]) > self.theta_threshold
        return self.state.copy(), 1.0 if not done else 0.0, done


# ============================================================
# LQR Optimal (analytical ground truth)
# ============================================================

def lqr_gain(gravity):
    """Return LQR gain K for given gravity."""
    mc, mp, l = 1.0, 0.1, 0.5
    mt = mc + mp
    l_eff = l * (4.0/3.0 - mp / mt)
    Ac = np.array([[0,1,0,0], [0,0,-mp*gravity/mt,0], [0,0,0,1], [0,0,gravity/l_eff,0]])
    Bc = np.array([[0], [1/mt], [0], [-1/(mt*l_eff)]])
    A = np.eye(4) + 0.02 * Ac
    B = 0.02 * Bc
    Q = np.diag([1.0, 0.1, 10.0, 0.1])
    R = np.array([[0.01]])
    try:
        P = linalg.solve_discrete_are(np.sqrt(0.99)*A, np.sqrt(0.99)*B, Q, R)
    except:
        P = 100 * np.eye(4)
        for _ in range(5000):
            P_new = Q + 0.99*A.T@P@A - 0.99**2*A.T@P@B@np.linalg.solve(R+0.99*B.T@P@B, B.T@P@A)
            if np.max(np.abs(P_new - P)) < 1e-12: break
            P = P_new
    return 0.99 * np.linalg.solve(R + 0.99*B.T@P@B, B.T@P@A)


def lqr_optimal_force_sign(state, gravity):
    """Return force direction: 0=left, 1=right."""
    K = lqr_gain(gravity)
    u = (-K @ state).flatten()[0]
    return 1 if u > 0 else 0


def lqr_optimal_action(state, gravity):
    """Return closest 4-action to LQR optimal."""
    K = lqr_gain(gravity)
    u = (-K @ state).flatten()[0]
    return int(np.argmin([abs(u - f) for f in FORCES]))


# ============================================================
# DQN
# ============================================================

class QNetwork(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS))

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, lr=1e-3, gamma=0.99, eps_start=1.0, eps_end=0.01,
                 eps_decay=0.995, buf_size=10000, batch_size=64, hidden=64):
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.q_net = QNetwork(hidden)
        self.target_net = QNetwork(hidden)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = deque(maxlen=buf_size)

    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        with torch.no_grad():
            return self.q_net(torch.FloatTensor(state).unsqueeze(0)).argmax(1).item()

    def store(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return
        idx = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        s, a, r, s2, d = zip(*batch)
        s = torch.FloatTensor(np.array(s))
        a = torch.LongTensor(a).unsqueeze(1)
        r = torch.FloatTensor(r)
        s2 = torch.FloatTensor(np.array(s2))
        d = torch.FloatTensor(d)
        q = self.q_net(s).gather(1, a).squeeze()
        with torch.no_grad():
            tgt = r + self.gamma * self.target_net(s2).max(1)[0] * (1 - d)
        loss = nn.MSELoss()(q, tgt)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def get_q(self, states):
        with torch.no_grad():
            return self.q_net(torch.FloatTensor(states)).numpy()

    def get_action(self, states):
        return self.get_q(states).argmax(axis=1)

    def get_force_sign(self, states):
        """0=left (actions 0,1), 1=right (actions 2,3)."""
        actions = self.get_action(states)
        return (actions >= 2).astype(int)


def train_dqn(gravity, n_episodes=250, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = CartPoleEnv(gravity=gravity)
    agent = DQNAgent(lr=1e-3, eps_decay=0.995, hidden=64, buf_size=10000)
    rewards = []
    for ep in range(n_episodes):
        s = env.reset(seed=seed*1000+ep)
        total_r = 0
        for t in range(300):
            a = agent.act(s)
            s2, r, done = env.step(a)
            agent.store(s, a, r, s2, float(done))
            if t % 4 == 0:
                agent.train_step()
            s = s2
            total_r += r
            if done:
                break
        rewards.append(total_r)
        if ep % 8 == 0:
            agent.update_target()
    return agent, rewards


def evaluate(agent, gravity, n_ep=10):
    env = CartPoleEnv(gravity=gravity)
    rets = []
    for ep in range(n_ep):
        s = env.reset(seed=9000+ep)
        ret = 0
        for t in range(300):
            with torch.no_grad():
                a = agent.q_net(torch.FloatTensor(s).unsqueeze(0)).argmax(1).item()
            s, r, done = env.step(a)
            ret += r
            if done:
                break
        rets.append(ret)
    return np.mean(rets)


# ============================================================
# Main Experiment
# ============================================================

def run_experiment():
    print("=== Exp 8: DQN Model-Agnostic Transfer (4-action) ===")

    # ─── Train DQN agents ───
    train_gravities = [6.0, 8.0, 9.8, 12.0, 15.0, 19.0]
    agents = {}
    print("Training DQN agents (best of 2 seeds)...")
    for g in train_gravities:
        best_agent, best_r = None, -1
        for si in range(2):
            agent, rews = train_dqn(g, n_episodes=250, seed=SEED + si*100 + int(g*10))
            avg = np.mean(rews[-40:])
            if avg > best_r:
                best_agent, best_r = agent, avg
        agents[g] = best_agent
        print(f"  g={g:5.1f}: best avg = {best_r:.1f}")

    src_g = 9.8
    src_agent = agents[src_g]

    # ─── Test states ───
    np.random.seed(SEED + 999)
    test_states = np.random.uniform(
        [-0.3, -0.3, -0.10, -0.3],
        [0.3, 0.3, 0.10, 0.3], size=(200, 4))

    # ─── (a) Directional OC: DQN vs LQR across gravities ───
    # Use force-sign agreement (left/right) for smooth curve
    print("\nDirectional OC: DQN(g=9.8) vs LQR-optimal...")
    oc_gravities = np.linspace(4.0, 22.0, 25)
    src_signs = src_agent.get_force_sign(test_states)

    oc_directional = []
    for g in oc_gravities:
        lqr_signs = np.array([lqr_optimal_force_sign(s, g) for s in test_states])
        oc = np.mean(src_signs == lqr_signs)
        oc_directional.append(oc)

    # DQN-DQN directional OC at trained gravities
    oc_dqn_dqn = []
    for g in train_gravities:
        tgt_signs = agents[g].get_force_sign(test_states)
        oc = np.mean(src_signs == tgt_signs)
        oc_dqn_dqn.append(oc)
        print(f"  DirOC(DQN-DQN) g={g:.1f}: {oc:.3f}")

    # ─── (b) Transfer performance ───
    print("\nTransfer performance...")
    eval_gravities = np.linspace(5.0, 20.0, 12)
    src_perf = []
    for g in eval_gravities:
        sp = evaluate(src_agent, g, n_ep=10)
        src_perf.append(sp)

    # ─── (c) Directional vote convergence: MV vs Q-Avg (force sign) ───
    print("\nDirectional vote convergence over K agents...")
    all_agents = list(agents.values())
    K_values = list(range(1, len(all_agents)+1))
    n_trials = 30
    target_g = 11.0
    n_test = 100
    # Target: LQR force direction
    target_dirs = np.array([lqr_optimal_force_sign(s, target_g) for s in test_states[:n_test]])

    mv_acc = []
    qa_acc = []
    for K in K_values:
        mv_a, qa_a = [], []
        for trial in range(n_trials):
            idx = np.random.choice(len(all_agents), K, replace=True)
            sub = [all_agents[i] for i in idx]
            # MV: vote on force direction
            dir_votes = np.zeros((n_test, 2), dtype=int)  # 0=left, 1=right
            q_sum = np.zeros((n_test, N_ACTIONS))
            for ag in sub:
                q = ag.get_q(test_states[:n_test])
                dirs = (q.argmax(axis=1) >= 2).astype(int)  # 0,1=left; 2,3=right
                q_sum += q
                for i in range(n_test):
                    dir_votes[i, dirs[i]] += 1
            mv_dirs = dir_votes.argmax(axis=1)
            mv_a.append(np.mean(mv_dirs == target_dirs))
            # QAvg: direction of argmax(sum Q)
            qa_dirs = (q_sum.argmax(axis=1) >= 2).astype(int)
            qa_a.append(np.mean(qa_dirs == target_dirs))
        mv_acc.append((np.mean(mv_a), np.std(mv_a)/np.sqrt(n_trials)))
        qa_acc.append((np.mean(qa_a), np.std(qa_a)/np.sqrt(n_trials)))
        print(f"  K={K}: MV={np.mean(mv_a):.3f}, QAvg={np.mean(qa_a):.3f}")

    # ─── (d) Q-value scale analysis ───
    print("\nQ-value scales...")
    q_scales = []
    for g in train_gravities:
        q = agents[g].get_q(test_states[:100])
        q_scales.append((g, np.mean(np.abs(q)), np.max(q) - np.min(q)))

    # ─── (e) Scale-invariance — KEY PLOT ───
    print("\nScale-invariance test (random per-agent scales)...")

    class ScaledAgent:
        def __init__(self, agent, scale):
            self.agent = agent; self.scale = scale
        def get_q(self, s):
            return self.agent.get_q(s) * self.scale
        def get_action(self, s):
            return self.get_q(s).argmax(axis=1)

    n_sc = 100
    base_votes = np.zeros((n_sc, N_ACTIONS), dtype=int)
    for ag in all_agents:
        acts = ag.get_action(test_states[:n_sc])
        for i in range(n_sc):
            base_votes[i, acts[i]] += 1
    base_mv = base_votes.argmax(axis=1)

    base_qsum = sum(ag.get_q(test_states[:n_sc]) for ag in all_agents)
    base_qa = base_qsum.argmax(axis=1)

    scale_ranges = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0]
    mv_inv = []
    qa_inv = []
    n_scale_trials = 20
    for sr in scale_ranges:
        mv_ag, qa_ag = [], []
        for trial in range(n_scale_trials):
            np.random.seed(SEED + trial*77 + int(sr*10))
            log_s = np.random.uniform(-np.log(max(sr, 1.01)), np.log(max(sr, 1.01)),
                                      len(all_agents))
            scales = np.exp(log_s)
            scaled = [ScaledAgent(a, s) for a, s in zip(all_agents, scales)]
            # MV
            votes = np.zeros((n_sc, N_ACTIONS), dtype=int)
            for ag in scaled:
                acts = ag.get_action(test_states[:n_sc])
                for i in range(n_sc):
                    votes[i, acts[i]] += 1
            mv_ag.append(np.mean(votes.argmax(axis=1) == base_mv))
            # QAvg
            qsum = sum(ag.get_q(test_states[:n_sc]) for ag in scaled)
            qa_ag.append(np.mean(qsum.argmax(axis=1) == base_qa))
        mv_inv.append((np.mean(mv_ag), np.std(mv_ag)/np.sqrt(n_scale_trials)))
        qa_inv.append((np.mean(qa_ag), np.std(qa_ag)/np.sqrt(n_scale_trials)))
        print(f"  range={sr:.0f}x: MV={np.mean(mv_ag):.3f}, QAvg={np.mean(qa_ag):.3f}")

    return {
        'oc_gravities': oc_gravities.tolist(), 'oc_directional': oc_directional,
        'train_gravities': train_gravities, 'oc_dqn_dqn': oc_dqn_dqn,
        'eval_gravities': eval_gravities.tolist(), 'src_perf': src_perf,
        'K_values': K_values, 'mv_acc': mv_acc, 'qa_acc': qa_acc,
        'q_scales': q_scales, 'src_g': src_g,
        'scale_ranges': scale_ranges, 'mv_inv': mv_inv, 'qa_inv': qa_inv,
    }


def plot_results(results, save_dir='../figures'):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # (a) Directional OC
    ax = axes[0, 0]
    ax.plot(results['oc_gravities'], results['oc_directional'], 'g-', lw=2.5,
            label='DQN vs LQR (direction)')
    ax.scatter(results['train_gravities'], results['oc_dqn_dqn'],
               c='blue', s=50, zorder=3, label='DQN vs DQN', edgecolors='black', lw=0.5)
    ax.axvline(results['src_g'], color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Target gravity (m/s$^2$)')
    ax.set_ylabel('Ordinal Consistency')
    ax.set_title('(a) DQN: Ordinal Consistency')
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=8)

    # (b) Transfer performance
    ax = axes[0, 1]
    ax.plot(results['eval_gravities'], results['src_perf'], 'b-o', ms=4, lw=2,
            label=f'Source DQN ($g={results["src_g"]}$)')
    ax.axvline(results['src_g'], color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Target gravity (m/s$^2$)')
    ax.set_ylabel('Episode reward')
    ax.set_title('(b) DQN: Transfer Performance')
    ax.legend(fontsize=9)

    # (c) Majority vote vs K
    ax = axes[0, 2]
    K = results['K_values']
    mv_m = [s[0] for s in results['mv_acc']]
    mv_se = [s[1] for s in results['mv_acc']]
    qa_m = [s[0] for s in results['qa_acc']]
    qa_se = [s[1] for s in results['qa_acc']]
    ax.errorbar(K, mv_m, yerr=mv_se, fmt='bo-', ms=5, lw=2, capsize=3,
                label='Majority Vote (ordinal)')
    ax.errorbar(K, qa_m, yerr=qa_se, fmt='s--', color='purple', ms=5, lw=2,
                capsize=3, label='Q-Averaging (cardinal)')
    ax.set_xlabel('$K$ (source DQN agents)')
    ax.set_ylabel('Directional accuracy vs LQR')
    ax.set_title('(c) DQN: Directional Vote vs $K$')
    ax.legend(fontsize=9)

    # (d) Q-value scales
    ax = axes[1, 0]
    gs = [s[0] for s in results['q_scales']]
    means = [s[1] for s in results['q_scales']]
    ranges = [s[2] for s in results['q_scales']]
    x = range(len(gs))
    ax.bar(x, means, color='steelblue', alpha=0.7, label='Mean $|Q|$')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{g:.0f}' for g in gs], fontsize=9)
    ax.set_xlabel('Training gravity')
    ax.set_ylabel('Q-value magnitude')
    ax.set_title('(d) DQN Q-Value Scales Vary')
    ax2 = ax.twinx()
    ax2.plot(x, ranges, 'ro-', ms=5, lw=2, label='Q range')
    ax2.set_ylabel('Q range', color='red')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    # (e) Scale-invariance — KEY PLOT
    ax = axes[1, 1]
    sr = results['scale_ranges']
    mv_m = [s[0] for s in results['mv_inv']]
    mv_se = [s[1] for s in results['mv_inv']]
    qa_m = [s[0] for s in results['qa_inv']]
    qa_se = [s[1] for s in results['qa_inv']]
    ax.errorbar(sr, mv_m, yerr=mv_se, fmt='bo-', ms=6, lw=2.5, capsize=3,
                label='Majority Vote')
    ax.errorbar(sr, qa_m, yerr=qa_se, fmt='s--', color='purple', ms=6, lw=2,
                capsize=3, label='Q-Averaging')
    ax.set_xscale('log')
    ax.axhline(1.0, color='green', ls='--', alpha=0.5)
    ax.set_xlabel('Scale heterogeneity range ($r$)')
    ax.set_ylabel('Agreement with unscaled')
    ax.set_title('(e) Scale-Invariance: MV vs Q-Avg')
    ax.set_ylim(0.5, 1.05)
    ax.legend(fontsize=9)

    # (f) OC decay with |Δg|
    ax = axes[1, 2]
    dg = np.abs(np.array(results['oc_gravities']) - results['src_g'])
    oc = np.array(results['oc_directional'])
    ax.scatter(dg, oc, c='steelblue', s=20, alpha=0.7, zorder=3)
    mask = dg > 0.5
    if mask.sum() > 3:
        coeffs = np.polyfit(dg[mask], oc[mask], 2)
        xf = np.linspace(0, dg.max(), 50)
        ax.plot(xf, np.polyval(coeffs, xf), 'r--', lw=2, label='Quadratic fit')
    ax.set_xlabel(r'$|\Delta g|$ from source')
    ax.set_ylabel('Ordinal Consistency')
    ax.set_title('(f) OC Decay with Parameter Distance')
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'exp8_dqn_transfer.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'exp8_dqn_transfer.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved exp8 figures to {save_dir}")


if __name__ == '__main__':
    results = run_experiment()
    plot_results(results)
