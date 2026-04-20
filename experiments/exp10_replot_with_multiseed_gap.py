"""
Regenerate figures/exp10_mujoco_transfer.{pdf,png} with the gap-column (col 3)
replaced by cross-seed-mean cell-level gaps over seeds {42, 123, 7}, and the
slope label set to the cross-seed-mean fit value (HC 1.20, Ant 2.52, Hop 1.17).

Cols 0-2 (transfer return, OC decay, scale-invariance) are kept from the
existing seed-42 cache so the figure layout and the rest of the paper's
narrative are unchanged. Only the headline slope panel is updated to match
the cross-seed numbers cited in the body and Table 1.

Inputs:
  - cache_exp10/results_{halfcheetah,ant,hopper}.pkl       (seed 42 single-seed)
  - cache_exp10/results_multiseed{,_ant,_hopper}.pkl       (seeds 42,123,7)

Outputs:
  - figures/exp10_mujoco_transfer.{pdf,png}                (overwritten)
"""

import os, pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, 'cache_exp10')
FIGS  = os.path.join(ROOT, 'figures')

SRC_GRAVITY = 9.81
GAP_FLOOR = 1.0

ENVS = [
    ('halfcheetah', 'HalfCheetah', 'results_multiseed.pkl'),
    ('ant',         'Ant',         'results_multiseed_ant.pkl'),
    ('hopper',      'Hopper',      'results_multiseed_hopper.pkl'),
]


def fit_loglog_slope(dgs, gaps):
    dgs = np.asarray(dgs); gaps = np.asarray(gaps)
    m = dgs > 0.3
    if m.sum() < 2:
        return np.nan, None
    log_dg = np.log(dgs[m])
    log_gap = np.log(np.maximum(gaps[m], GAP_FLOOR))
    coeffs = np.polyfit(log_dg, log_gap, 1)
    return float(coeffs[0]), coeffs


def main():
    matplotlib.rcParams.update({'figure.dpi': 150})

    # Load single-seed (cols 0-2) and multi-seed (col 3) caches.
    single = {}
    multi  = {}
    for short, label, mfn in ENVS:
        with open(os.path.join(CACHE, f'results_{short}.pkl'), 'rb') as f:
            single[short] = pickle.load(f)
        with open(os.path.join(CACHE, mfn), 'rb') as f:
            multi[short] = pickle.load(f)

    n_envs = len(ENVS)
    fig, axes = plt.subplots(n_envs, 4, figsize=(20, 4.2 * n_envs))
    panels = 'abcdefghijklmnopqrst'

    for row, (short, label, _) in enumerate(ENVS):
        r = single[short]
        eg = r['eval_gravities']
        sg = r['src_gravity']
        gs = r['gravities']
        pi = row * 4

        # Col 0: Transfer return + DR (unchanged from seed-42 cache).
        ax = axes[row, 0]
        ax.plot(eg, r['opt_returns'], 'go-', ms=4, lw=2, label='Target-trained')
        ax.plot(eg, r['src_returns'], 'b^--', ms=4, lw=2, label=f'Source ($g={sg}$)')
        ax.plot(eg, r['dr_returns'], 'rs-.', ms=4, lw=1.5, label='Domain Rand.')
        ax.fill_between(eg, r['src_returns'], r['opt_returns'], alpha=0.1, color='red')
        ax.axvline(sg, color='gray', ls='--', alpha=0.5)
        ax.set_xlabel('Gravity (m/s$^2$)')
        ax.set_ylabel('Episode return')
        ax.set_title(f'({panels[pi]}) {label}: Transfer Return')
        ax.legend(fontsize=7, loc='best')

        # Col 1: OC decay (unchanged).
        ax = axes[row, 1]
        dg = [abs(g - sg) for g in gs]
        oc = r['oc_values']
        ax.scatter(dg, oc, c='steelblue', s=60, zorder=3, edgecolors='black', lw=0.5)
        if len(dg) >= 3:
            coeffs = np.polyfit(dg, oc, 2)
            xfit = np.linspace(0, max(dg), 50)
            ax.plot(xfit, np.clip(np.polyval(coeffs, xfit), 0.35, 1.05),
                    'b-', lw=2.5, alpha=0.7, label='Trend')
        ax.axhline(0.5, color='red', ls=':', alpha=0.4, label='Random')
        ax.set_xlabel(r'$|\Delta g|$ from source')
        ax.set_ylabel('Directional OC')
        ax.set_title(f'({panels[pi+1]}) {label}: OC Decay')
        ax.set_ylim(0.35, 1.05)
        ax.legend(fontsize=9)

        # Col 2: Scale-invariance (unchanged).
        ax = axes[row, 2]
        ax.semilogx(r['scale_ranges'], r['mv_invariance'], 'bo-', ms=6, lw=2.5,
                    label='Ordinal (sign vote)')
        ax.semilogx(r['scale_ranges'], r['qa_invariance'], 's--', color='purple',
                    ms=6, lw=2, label='Mean-action')
        ax.axhline(1.0, color='green', ls='--', alpha=0.5)
        ax.set_xlabel('Scale heterogeneity ($r$)')
        ax.set_ylabel('Agreement with unscaled')
        ax.set_title(f'({panels[pi+2]}) {label}: Scale-Invariance')
        ax.set_ylim(0.5, 1.05)
        ax.legend(fontsize=9)

        # Col 3: Transfer Gap (log-log) — REPLACED with cross-seed-mean.
        ax = axes[row, 3]
        agg = multi[short]['aggregated']
        gravities_ms = sorted(agg.keys())
        dg_ms = np.array([abs(g - SRC_GRAVITY) for g in gravities_ms])
        gap_mean = np.array([float(agg[g]['gap_mean']) for g in gravities_ms])
        gap_std  = np.array([float(agg[g]['gap_std'])  for g in gravities_ms])

        pos = dg_ms > 0.3
        dg_pos = dg_ms[pos]
        gap_pos = np.maximum(gap_mean[pos], GAP_FLOOR)
        gap_pos_std = gap_std[pos]

        order = np.argsort(dg_pos)
        dg_pos = dg_pos[order]; gap_pos = gap_pos[order]; gap_pos_std = gap_pos_std[order]

        # Asymmetric error bars in log space; clip lower to GAP_FLOOR.
        lower = gap_pos - np.minimum(gap_pos - GAP_FLOOR, gap_pos_std)
        upper = gap_pos + gap_pos_std
        yerr = np.vstack([gap_pos - lower, upper - gap_pos])

        ax.errorbar(dg_pos, gap_pos, yerr=yerr, fmt='b^-', ms=7, lw=2,
                    elinewidth=1.2, capsize=3,
                    label='Transfer gap (mean $\\pm$ std, 3 seeds)')

        slope, coeffs = fit_loglog_slope(dg_pos, gap_pos)
        if coeffs is not None:
            xfit = np.logspace(np.log10(dg_pos.min() * 0.85),
                               np.log10(dg_pos.max() * 1.18), 30)
            ax.loglog(xfit, np.exp(np.polyval(coeffs, np.log(xfit))),
                      'r--', lw=2,
                      label=f'Cross-seed slope $\\approx {slope:.2f}$')

        # Quadratic reference (slope 2) anchored at the leftmost point.
        x_ref = np.array([dg_pos.min() * 0.85, dg_pos.max() * 1.18])
        y_ref = gap_pos[0] * (x_ref / dg_pos[0]) ** 2
        ax.loglog(x_ref, y_ref, color='gray', lw=1, ls=':',
                  label='Quadratic reference (slope $2$)')

        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r'$|\Delta g|$ from source')
        ax.set_ylabel('Transfer gap (cross-seed mean)')
        ax.set_title(f'({panels[pi+3]}) {label}: Gap Scaling (multi-seed)')
        ax.legend(fontsize=8, loc='best')

    plt.tight_layout()
    out_pdf = os.path.join(FIGS, 'exp10_mujoco_transfer.pdf')
    out_png = os.path.join(FIGS, 'exp10_mujoco_transfer.png')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()
    print(f'[saved] {out_pdf}')
    print(f'[saved] {out_png}')


if __name__ == '__main__':
    main()
