"""
Experiment 19: Validate the F_kappa violation-growth bound directly.

Theorem (violation growth): |S_viol|/|S| <= F_kappa(2 * L_Q * ||Delta theta||).

We test this on SAC/MuJoCo by combining two cached sources:
  (a) exp14: per-state action-gap samples kappa(s) at source gravity.
  (b) exp18: L_Q^cert from source-only finite differences (Proposition 2).

For each env and each target gravity g_t, we compare
  bound    = F_kappa^hat( 2 * L_Q^cert * |g_t - g_src| )
  measured = 1 - OC_directional(g_src, g_t)            [cached in exp14]

If the bound holds, we have direct empirical evidence that the shape of F_kappa
(not just its median) governs transfer fragility, which is the reviewer's ask.
"""

import os, pickle, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.size': 12, 'font.family': 'serif',
    'axes.labelsize': 13, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'figure.dpi': 150,
})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXP14_PKL = os.path.join(ROOT, 'figures', 'exp14_results.pkl')
EXP18_PKL = os.path.join(ROOT, 'cache_exp10', 'results_lq_estimator.pkl')
SRC_GRAVITY = 9.81
GRAVITIES = [5.0, 7.0, 9.81, 12.0, 15.0]

DEFAULT_EPS_Q = 1.0
DEFAULT_H_Q = None  # None means use the data lower bound from fd2


def compute_lq_cert(record, eps_q=DEFAULT_EPS_Q, H_Q=DEFAULT_H_Q):
    """Proposition 2 source-side cert (mirrors exp18.compute_lq_stats)."""
    gravities = record['gravities']
    Q = record['q_values']  # (K, N)
    K, N = Q.shape
    dQ = np.diff(Q, axis=0)
    h = np.diff(gravities)
    fd1 = np.abs(dQ) / h[:, None]

    fd2 = []
    for k in range(1, K - 1):
        hL = gravities[k] - gravities[k - 1]
        hR = gravities[k + 1] - gravities[k]
        num = 2.0 * (hL * (Q[k + 1] - Q[k]) - hR * (Q[k] - Q[k - 1]))
        den = hL * hR * (hL + hR)
        fd2.append(np.abs(num / den))
    fd2 = np.array(fd2)

    H_Q_data = float(fd2.max()) if fd2.size else 0.0
    H_Q_used = H_Q if H_Q is not None else H_Q_data

    cert_per_pair = (fd1.max(axis=1) + 2 * eps_q / h) + H_Q_used * h
    return {
        'L_Q_cert': float(cert_per_pair.max()),
        'H_Q_used': H_Q_used,
    }


def empirical_fkappa(gaps, x):
    """F_kappa(x) = fraction of states with kappa(s) <= x."""
    gaps = np.asarray(gaps)
    if np.isscalar(x):
        return float(np.mean(gaps <= x))
    return np.array([np.mean(gaps <= xi) for xi in x])


def main():
    with open(EXP14_PKL, 'rb') as f:
        exp14 = pickle.load(f)
    with open(EXP18_PKL, 'rb') as f:
        exp18 = pickle.load(f)

    env_ids = list(exp14.keys())
    summary = {}

    for env_id in env_ids:
        gaps = exp14[env_id]['gaps']
        oc_curve = exp14[env_id]['oc_values']  # over GRAVITIES
        label = exp14[env_id]['label']
        color = exp14[env_id]['color']

        stats = compute_lq_cert(exp18[env_id])
        L_Q = stats['L_Q_cert']

        rows = []
        for g, oc in zip(GRAVITIES, oc_curve):
            dg = abs(g - SRC_GRAVITY)
            if dg == 0:
                continue
            measured_viol = 1.0 - float(oc)
            threshold = 2.0 * L_Q * dg
            bound = empirical_fkappa(gaps, threshold)
            # Inverse: smallest kappa* such that F_kappa(kappa*) >= measured.
            sorted_g = np.sort(gaps)
            idx = int(np.ceil(measured_viol * len(sorted_g))) - 1
            idx = max(0, min(idx, len(sorted_g) - 1))
            kappa_star = float(sorted_g[idx])
            # Effective L_Q that would make bound exact: kappa* = 2 L_eff * dg.
            L_eff = kappa_star / (2.0 * dg) if dg > 0 else np.nan
            rows.append({
                'g': g, 'dg': dg,
                'measured': measured_viol,
                'threshold': threshold,
                'bound': bound,
                'kappa_star': kappa_star,
                'L_eff': L_eff,
            })

        summary[env_id] = {
            'label': label, 'color': color,
            'L_Q_cert': L_Q,
            'H_Q': stats['H_Q_used'],
            'median_gap': float(np.median(gaps)),
            'rows': rows,
            'gaps': gaps,
        }

    # ── Print table ──
    print()
    print(f"{'env':<14}{'L_Q^cert':>10}{'median_k':>10}  {'|dg|':>6}{'thresh=2LDg':>14}{'F_k(bound)':>12}{'measured':>10}{'L_eff':>10}{'holds?':>8}")
    print('-' * 100)
    all_holds = True
    n_tests = 0
    for env_id, s in summary.items():
        for r in s['rows']:
            holds = r['bound'] + 1e-6 >= r['measured']
            all_holds = all_holds and holds
            n_tests += 1
            print(f"{s['label']:<14}{s['L_Q_cert']:>10.2f}{s['median_gap']:>10.3f}  "
                  f"{r['dg']:>6.2f}{r['threshold']:>14.2f}{r['bound']:>12.3f}"
                  f"{r['measured']:>10.3f}{r['L_eff']:>10.3f}{'YES' if holds else 'NO':>8}")
    print()
    print(f"Bound validation: {'ALL HOLD' if all_holds else 'SOME VIOLATIONS'} "
          f"({n_tests} tests)")
    # Ratio L_cert / L_eff per env: how loose is the certified Lipschitz in practice?
    print()
    print(f"{'env':<14}{'L_Q^cert':>10}{'median L_eff':>14}{'ratio':>8}")
    for env_id, s in summary.items():
        L_effs = [r['L_eff'] for r in s['rows']]
        m_eff = float(np.median(L_effs))
        ratio = s['L_Q_cert'] / m_eff if m_eff > 0 else np.nan
        print(f"{s['label']:<14}{s['L_Q_cert']:>10.2f}{m_eff:>14.3f}{ratio:>8.0f}x")
        s['median_L_eff'] = m_eff
        s['L_ratio'] = ratio

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) F_kappa CDFs with vertical lines at 2 L_Q |Δg|
    ax = axes[0]
    for env_id, s in summary.items():
        gaps = np.sort(s['gaps'])
        gaps = gaps[gaps > 0]
        cdf = np.arange(1, len(gaps) + 1) / len(s['gaps'])
        ax.plot(gaps, cdf, lw=2.2, color=s['color'], label=s['label'])
    ax.set_xscale('log')
    ax.set_xlabel(r'Action gap $\kappa(s)$ (log)')
    ax.set_ylabel(r'$\hat F_\kappa(x)$')
    ax.set_title(r'(a) Empirical $\hat F_\kappa$ per env')
    ax.legend(fontsize=9)
    ax.set_ylim(-0.02, 1.05)

    # (b) Shape test: F_kappa(c * |Dg|) at fitted c vs measured violation,
    # with a median-only baseline predictor (step function at the median gap)
    # for a head-to-head between full-CDF shape and a single-summary statistic.
    ax = axes[1]
    for env_id, s in summary.items():
        dgs = np.array([r['dg'] for r in s['rows']])
        m   = np.array([r['measured'] for r in s['rows']])
        gaps = s['gaps']
        med_kappa = float(np.median(gaps))

        # Shape predictor: F_kappa(c * |Dg|), best c by LS.
        cs = np.logspace(-2, 3, 400)
        errs_shape = [np.mean((empirical_fkappa(gaps, c * dgs) - m) ** 2) for c in cs]
        c_star = float(cs[int(np.argmin(errs_shape))])
        mse_shape = float(np.min(errs_shape))

        # Median baseline: 1[c * |Dg| >= median_kappa], best c by LS over same grid.
        # This is what a "median gap" summary predicts: zero violations until the
        # threshold, then 100% flips. It is a step function at the median.
        errs_med = [np.mean(((c * dgs >= med_kappa).astype(float) - m) ** 2) for c in cs]
        c_med = float(cs[int(np.argmin(errs_med))])
        mse_med = float(np.min(errs_med))

        s['c_star'] = c_star
        s['c_median'] = c_med
        s['mse_shape'] = mse_shape
        s['mse_median'] = mse_med

        xs = np.linspace(0, dgs.max() * 1.1, 200)
        ax.plot(xs, empirical_fkappa(gaps, c_star * xs),
                '-', color=s['color'], lw=2.2,
                label=f"{s['label']}: shape (MSE={mse_shape:.3f})")
        ax.plot(xs, (c_med * xs >= med_kappa).astype(float),
                '--', color=s['color'], lw=1.4, alpha=0.7,
                label=f"{s['label']}: median-only (MSE={mse_med:.3f})")
        ax.plot(dgs, m, 'o', color=s['color'], ms=7, mec='black', mew=0.8)
    ax.set_xlabel(r'$|\Delta g|$ (m/s$^2$)')
    ax.set_ylabel('violation fraction')
    ax.set_title(r'(b) shape (solid) vs median-only (dashed) vs measured ($\circ$)')
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=7.5, loc='lower right', ncol=1)

    # (c) bound validity: x = F_kappa(2 L_cert dg) (always 1.0), y = measured;
    # show the certified bound is valid but loose, and the fitted c* reveals
    # the shape of F_kappa is what predicts the trend.
    ax = axes[2]
    labels_done = set()
    for env_id, s in summary.items():
        for r in s['rows']:
            label = s['label'] if s['label'] not in labels_done else None
            labels_done.add(s['label'])
            ax.scatter(r['bound'], r['measured'], s=100, color=s['color'],
                       edgecolor='black', lw=1, label=label, zorder=4)
    ax.plot([0, 1.05], [0, 1.05], 'k--', lw=1, alpha=0.5)
    ax.fill_between([0, 1.05], [0, 1.05], 1.05, color='green', alpha=0.08)
    ax.text(0.5, 0.1, r'certified bound $\geq$ measured',
            transform=ax.transAxes, fontsize=10, ha='center',
            color='darkgreen', style='italic')
    ax.set_xlabel(r'$\hat F_\kappa(2\hat L_Q^{\mathrm{cert}}|\Delta g|)$')
    ax.set_ylabel(r'$1 - \mathrm{OC}$')
    ax.set_title('(c) certified bound valid (loose)')
    ax.set_xlim(-0.02, 1.05); ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    out_pdf = os.path.join(ROOT, 'figures', 'exp19_fkappa_bound.pdf')
    out_png = os.path.join(ROOT, 'figures', 'exp19_fkappa_bound.png')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()

    out_pkl = os.path.join(ROOT, 'cache_exp10', 'exp19_bound_validation.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump(summary, f)
    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_pkl}")

    # ── Shape vs median head-to-head ──
    print()
    print(f"{'env':<14}{'MSE shape':>12}{'MSE median':>14}{'ratio':>10}")
    print('-' * 55)
    for env_id, s in summary.items():
        ratio = s['mse_median'] / s['mse_shape'] if s['mse_shape'] > 0 else float('inf')
        print(f"{s['label']:<14}{s['mse_shape']:>12.4f}{s['mse_median']:>14.4f}{ratio:>9.1f}x")


if __name__ == '__main__':
    main()
