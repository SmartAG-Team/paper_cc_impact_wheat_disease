"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/winter_wheat_population_20261010')
import numpy as np

def allocation_bounds(frame):
    """Joint fraction extrema for arbitrary crop allocations inside each country.

    Countries have exposed capacity E, other domain capacity N and outside
    capacity O. Their winter production is constrained to [L,U]. Minimum share
    pairs minimum feasible exposed mass with maximum feasible other-domain mass;
    the reverse pairing maximizes the share. Each pairing is jointly feasible.
    """
    if frame.country.duplicated().any():
        raise ValueError('Duplicate country margin')
    columns = ['total_tonnes', 'domain_tonnes', 'exposed_tonnes', 'winter_lower_tonnes', 'winter_upper_tonnes']
    a = frame[columns].to_numpy(float)
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError('Capacities and margins must be finite and nonnegative')
    t, d, e, l, u = a.T
    eps = 1e-06
    if np.any(d > t + eps) or np.any(e > d + eps) or np.any(l > u + eps) or np.any(u > t + eps):
        raise ValueError('Infeasible capacities or winter margins')
    d = np.minimum(d, t)
    e = np.minimum(e, d)
    u = np.minimum(u, t)
    l = np.minimum(l, u)
    n = d - e
    o = t - d
    xmin = np.maximum(0, l - (t - e))
    xmax = np.minimum(u, e)
    ymin = np.maximum(0, l - (t - n))
    ymax = np.minimum(u, n)
    domain_lower = float(np.maximum(0, l - o).sum())
    domain_upper = float(np.minimum(u, d).sum())
    result = dict(domain_winter_lower_tonnes=domain_lower, domain_winter_upper_tonnes=domain_upper, exposed_winter_lower_tonnes=float(xmin.sum()), exposed_winter_upper_tonnes=float(xmax.sum()), possible_domain_population=domain_upper > 0)
    for kind, x, y in [('lower', xmin.copy(), ymax.copy()), ('upper', xmax.copy(), ymin.copy())]:
        if x.sum() + y.sum() == 0 and domain_upper > 0:
            if kind == 'lower':
                x[np.argmax(xmax)] = xmax.max()
            else:
                y[np.argmax(ymax)] = ymax.max()
        z = np.maximum(0, l - x - y)
        if np.any(z > o + eps) or np.any(x + y + z > u + eps):
            raise ValueError('Internal witness infeasibility')
        denominator = float(x.sum() + y.sum())
        result['share_' + kind + '_pct'] = 100 * float(x.sum()) / denominator if denominator > 0 else np.nan
        result['witness_' + kind] = [dict(country=country, exposed_winter_tonnes=float(xx), other_domain_winter_tonnes=float(yy), outside_winter_tonnes=float(zz)) for country, xx, yy, zz in zip(frame.country, x, y, z)]
    return result
