"""
Experiment 22: Per-seed slope dispersion and direction asymmetry.

Two diagnostics built from the existing multiseed cache, no new training:

  (a) Per-seed log-log slope of cell-level gap vs |Delta g|.
      For each env and seed s in {42, 123, 7}, fit slope on the four shifted
      gravities {5, 7, 12, 15}. Reports {s42, s123, s7, mean, std} alongside
      the cross-seed-aggregate-gap slope (the headline number).

      Finding: per-seed slopes have std 2.4-3.4 across all three envs,
      driven by single (seed, gravity) cells where src > tgt and gap is
      clipped to GAP_FLOOR. The cross-seed-mean of gaps cancels these
      cell-level zero events; the aggregate slope is the only stable
      summary. Ant slope > 2 is driven by seed 123 alone; seeds 42 and 7
      give 1.60 and 1.72 (both below 2).

  (b) Direction asymmetry: gap at heavier gravity vs lighter gravity for
      matched |Delta g|. Ratio gap_heavy / gap_light at |dg| in {2.5, 5}.

      Finding: Ant's gap concentrates 16x more on the lighter side at
      |dg|=2.5 (ratio 0.06). HalfCheetah is mildly asymmetric (0.39).
      Hopper is reversed (1.52: heavier is harder). Combined with
      median mu(s) flipping negative only at Ant g=5, the precondition
      failure that drives Ant's slope > 2 is a *light-gravity* boundary
      case, not a general framework failure.

Inputs:
  - cache_exp10/results_multiseed{,_ant,_hopper}.pkl

Outputs:
  - cache_exp10/exp22_seed_and_direction.pkl
"""

import os, pickle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, 'cache_exp10')

SRC = 9.81
GS = [5.0, 7.0, 12.0, 15.0]
GAP_FLOOR = 1.0

ENVS = [
    ('HalfCheetah', 'results_multiseed.pkl'),
    ('Ant',         'results_multiseed_ant.pkl'),
    ('Hopper',      'results_multiseed_hopper.pkl'),
]


def slope(dgs, gaps):
    dgs = np.asarray(dgs); gaps = np.asarray(gaps)
    m = dgs > 0.3
    if m.sum() < 2: return np.nan
    log_dg = np.log(dgs[m])
    log_gap = np.log(np.maximum(gaps[m], GAP_FLOOR))
    return float(np.polyfit(log_dg, log_gap, 1)[0])


def main():
    out = {'per_env': {}}
    print('=== (a) Per-seed slope variance ===')
    print(f'{"env":<13}{"s42":>8}{"s123":>8}{"s7":>8}{"mean":>8}{"std":>8}{"agg":>8}')
    for label, fn in ENVS:
        d = pickle.load(open(os.path.join(CACHE, fn), 'rb'))
        seeds = d['seeds']
        per_seed_slopes = []
        per_seed_gaps = {}
        for s in seeds:
            r = d['per_seed'][s]['returns']
            dgs = [abs(g - SRC) for g in GS]
            gaps = [float(r[g]['gap']) for g in GS]
            per_seed_slopes.append(slope(dgs, gaps))
            per_seed_gaps[s] = dict(zip(GS, gaps))
        agg = d['aggregated']
        agg_gaps = [float(agg[g]['gap_mean']) for g in GS]
        agg_slope = slope([abs(g - SRC) for g in GS], agg_gaps)
        arr = np.array(per_seed_slopes)
        print(f'{label:<13}{arr[0]:>8.2f}{arr[1]:>8.2f}{arr[2]:>8.2f}'
              f'{arr.mean():>8.2f}{arr.std():>8.2f}{agg_slope:>8.2f}')
        out['per_env'][label] = {
            'seeds': list(seeds),
            'per_seed_slopes': arr.tolist(),
            'per_seed_mean': float(arr.mean()),
            'per_seed_std': float(arr.std()),
            'agg_slope': agg_slope,
            'per_seed_gaps': per_seed_gaps,
            'agg_gaps': dict(zip(GS, agg_gaps)),
        }

    print()
    print('=== (b) Direction asymmetry (cross-seed-mean gaps) ===')
    print(f'{"env":<13}{"|dg|":>6}{"gL (g=)":>14}{"gH (g=)":>14}{"ratio H/L":>12}')
    asym_pairs = [(2.5, 7.0, 12.0), (5.0, 5.0, 15.0)]
    for label, fn in ENVS:
        d = pickle.load(open(os.path.join(CACHE, fn), 'rb'))
        agg = d['aggregated']
        env_asym = {}
        for dg, gL, gH in asym_pairs:
            gapL = float(agg[gL]['gap_mean'])
            gapH = float(agg[gH]['gap_mean'])
            ratio = gapH / max(gapL, 1e-6)
            print(f'{label:<13}{dg:>6.1f}{gapL:>10.1f} (g={gL})'
                  f'{gapH:>10.1f} (g={gH}){ratio:>12.2f}')
            env_asym[(gL, gH)] = {'gap_light': gapL, 'gap_heavy': gapH, 'ratio': ratio}
        out['per_env'][label]['direction_asymmetry'] = env_asym

    out_path = os.path.join(CACHE, 'exp22_seed_and_direction.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    print(f'\n[saved] {out_path}')


if __name__ == '__main__':
    main()
