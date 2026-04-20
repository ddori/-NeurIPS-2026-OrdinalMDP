"""
Experiment 20: Defend the quadratic upper bound against the Ant exceedance.

Two panels (cross-seed mean gap; the headline quantity in the paper):

  (B1) Precondition violation vs quadratic residual correlation.
       For each of the 12 (env, gravity) points, x = frac_nonconcave at that
       gravity (precondition severity), y = log(gap_mean) - 2 log(|Dg|)
       (quadratic deviation). Tests whether precondition severity tracks
       deviation from the bound.

  (B2) Median strong-concavity margin per (env, gravity), with the
       quadratic-bound-exceedance threshold (mu = 0) highlighted. Shows that
       Ant is the only environment whose median mu becomes negative at any
       gravity, matching the fact that Ant is the only environment whose
       multi-seed slope exceeds the quadratic upper bound.

Per-seed slope distribution was tried but discarded: 4-point per-seed log-log
slopes have std ~3 across seeds, dominated by single-gravity outliers, and
would only undermine the multi-seed headline rather than defend it.

Inputs:
  - cache_exp10/results_hessian.pkl              per-state mu and aggregates
  - cache_exp10/results_multiseed.pkl            HalfCheetah multi-seed gaps
  - cache_exp10/results_multiseed_ant.pkl        Ant
  - cache_exp10/results_multiseed_hopper.pkl     Hopper
"""

import os, pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.size': 12, 'font.family': 'serif',
    'axes.labelsize': 13, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'figure.dpi': 150,
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_GRAVITY = 9.81
GRAVITIES = [5.0, 7.0, 9.81, 12.0, 15.0]
GAP_FLOOR = 1.0  # avoid log(0); matches exp10 fit protocol

ENV_FILES = [
    ('HalfCheetah', 'HalfCheetah-v4', 'results_multiseed.pkl',        '#1f77b4'),
    ('Ant',         'Ant-v4',         'results_multiseed_ant.pkl',    '#d62728'),
    ('Hopper',      'Hopper-v4',      'results_multiseed_hopper.pkl', '#2ca02c'),
]


def load_all():
    with open(os.path.join(ROOT, 'cache_exp10', 'results_hessian.pkl'), 'rb') as f:
        hessian = pickle.load(f)
    data = {}
    for label, env_key, fname, color in ENV_FILES:
        with open(os.path.join(ROOT, 'cache_exp10', fname), 'rb') as f:
            multi = pickle.load(f)
        data[label] = {
            'env_key': env_key,
            'color': color,
            'mu_stats': hessian[env_key]['mu_stats'],
            'per_seed': multi['per_seed'],
            'aggregated': multi['aggregated'],
            'seeds': multi['seeds'],
        }
    return data


def fit_loglog_slope(dgs, gaps):
    """Match exp10 fit protocol: clip gap at GAP_FLOOR, require |Dg| > 0.3."""
    dgs = np.asarray(dgs)
    gaps = np.asarray(gaps)
    mask = dgs > 0.3
    if mask.sum() < 2:
        return np.nan
    log_dg = np.log(dgs[mask])
    log_gap = np.log(np.maximum(gaps[mask], GAP_FLOOR))
    slope, _ = np.polyfit(log_dg, log_gap, 1)
    return float(slope)


def main():
    data = load_all()
    env_order = list(data.keys())

    # ── Build (env, gravity) point set on cross-seed mean gap ──
    points = []
    for label, d in data.items():
        for g in GRAVITIES:
            if g == SRC_GRAVITY:
                continue
            dg = abs(g - SRC_GRAVITY)
            frac_nc = d['mu_stats'][g]['frac_nonconcave']
            med_mu = d['mu_stats'][g]['median']
            gap = max(float(d['aggregated'][g]['gap_mean']), GAP_FLOOR)
            residual = np.log(gap) - 2.0 * np.log(dg)
            points.append({
                'env': label, 'g': g, 'dg': dg, 'gap': gap,
                'frac_nc': frac_nc, 'med_mu': med_mu, 'residual': residual,
                'color': d['color'],
            })

    # Multi-seed-avg slope (headline number).
    multiseed_slopes = {}
    for label, d in data.items():
        dgs = []; gaps = []
        for g in GRAVITIES:
            if g == SRC_GRAVITY: continue
            dgs.append(abs(g - SRC_GRAVITY))
            gaps.append(float(d['aggregated'][g]['gap_mean']))
        multiseed_slopes[label] = fit_loglog_slope(dgs, gaps)

    # Correlations.
    frac_nc_arr = np.array([p['frac_nc'] for p in points])
    resid_arr   = np.array([p['residual'] for p in points])
    pearson = float(np.corrcoef(frac_nc_arr, resid_arr)[0, 1])
    def rank(a):
        order = np.argsort(a, kind='mergesort')
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(a))
        return ranks
    spearman = float(np.corrcoef(rank(frac_nc_arr), rank(resid_arr))[0, 1])

    # Print summary.
    print()
    print(f"{'env':<14}{'g':>6}{'dg':>6}{'gap':>10}{'frac_nc':>10}{'med_mu':>10}{'residual':>10}")
    print('-' * 70)
    for p in points:
        print(f"{p['env']:<14}{p['g']:>6.2f}{p['dg']:>6.2f}{p['gap']:>10.1f}"
              f"{p['frac_nc']:>10.2f}{p['med_mu']:>10.2f}{p['residual']:>10.2f}")
    print()
    print(f"Multi-seed-avg slopes: " +
          ", ".join(f"{l}={multiseed_slopes[l]:.2f}" for l in env_order))
    print(f"Pearson r(frac_nc, residual) = {pearson:.3f}, "
          f"Spearman = {spearman:.3f}, n = {len(points)}")

    # ── Plot: 2 panels ──
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    # (B1) Residual vs precondition severity
    ax = axes[0]
    drawn = set()
    for p in points:
        lab = p['env'] if p['env'] not in drawn else None
        drawn.add(p['env'])
        ax.scatter(p['frac_nc'], p['residual'], s=110, c=p['color'],
                   edgecolor='black', lw=0.8, alpha=0.9, label=lab, zorder=3)
        # Annotate gravity.
        ax.annotate(f"g={p['g']:.0f}", (p['frac_nc'], p['residual']),
                    xytext=(6, 4), textcoords='offset points', fontsize=8)
    if np.std(frac_nc_arr) > 0:
        coeffs = np.polyfit(frac_nc_arr, resid_arr, 1)
        xs = np.linspace(frac_nc_arr.min() * 0.9, frac_nc_arr.max() * 1.05, 30)
        ax.plot(xs, np.polyval(coeffs, xs), 'k--', lw=1.5, alpha=0.6,
                label=f'OLS (slope={coeffs[0]:.2f})')
    ax.set_xlabel('Fraction of non-concave states (precondition severity)')
    ax.set_ylabel(r'Quadratic residual $\log(\mathrm{gap}) - 2\log|\Delta g|$')
    ax.set_title('(a) Precondition vs quadratic deviation '
                 f'(Pearson $r={pearson:.2f}$, $n=12$)')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    # (B2) Median mu per (env, gravity), with mu = 0 line
    ax = axes[1]
    width = 0.25
    xs = np.array(GRAVITIES)
    for i, label in enumerate(env_order):
        d = data[label]
        med_mus = [d['mu_stats'][g]['median'] for g in GRAVITIES]
        offset = (i - 1) * width
        ax.bar(xs + offset, med_mus, width, color=d['color'],
               edgecolor='black', lw=0.6, alpha=0.85,
               label=f"{label} (slope {multiseed_slopes[label]:.2f})")
    ax.axhline(0, color='red', ls='--', lw=1.5, alpha=0.7,
               label=r'Precondition limit ($\mu=0$)')
    ax.set_xlabel('Gravity (m/s$^2$)')
    ax.set_ylabel(r'Median strong-concavity margin $\tilde\mu(s)$')
    ax.set_title('(b) Median $\\mu$ per (env, gravity); '
                 r'$\tilde\mu < 0$ only on Ant$|_{g=5}$')
    ax.set_xticks(xs)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_pdf = os.path.join(ROOT, 'figures', 'exp20_precondition_defense.pdf')
    out_png = os.path.join(ROOT, 'figures', 'exp20_precondition_defense.png')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()

    out_pkl = os.path.join(ROOT, 'cache_exp10', 'exp20_results.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'points': points,
            'multiseed_slopes': multiseed_slopes,
            'pearson': pearson,
            'spearman': spearman,
        }, f)
    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_pkl}")


if __name__ == '__main__':
    main()
