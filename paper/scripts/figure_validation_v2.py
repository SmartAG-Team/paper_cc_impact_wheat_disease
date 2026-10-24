"""Auditable displays of existing assessment-level monitoring forecasts.

No model is fitted here. Predictive R² is 1 − weighted SSE / weighted SST,
with the weighted target mean recomputed in every paired site resample.
Resampling intervals describe fixed, repeatedly inspected development forecasts;
they are not external-validation coverage statements.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D

from .figure_validation import diagnostics


EVALUATIONS = ('temporal', 'spatial_50km')
COLORS = {'temporal': '#0072B2', 'spatial_50km': '#D55E00'}
LABELS = {'temporal': 'Temporal', 'spatial_50km': 'Spatial, 50 km'}
MARKERS = {'temporal': 'o', 'spatial_50km': 's'}
BOOTSTRAP_SEED = 20261024
BOOTSTRAP_REPLICATES = 2000
EDGES = np.linspace(0., 100., 41)
STYLE = {'font.family': 'DejaVu Sans', 'font.size': 12,
         'axes.labelsize': 12, 'axes.titlesize': 12, 'legend.fontsize': 12,
         'xtick.labelsize': 12, 'ytick.labelsize': 12,
         'axes.spines.top': True, 'axes.spines.right': True,
         'axes.linewidth': .8, 'axes.edgecolor': '#353B40',
         'xtick.direction': 'out', 'ytick.direction': 'out',
         'svg.fonttype': 'none', 'pdf.fonttype': 42}


def _canonical(frame):
    aliases = {'globaltarget': 'global_target_index', 'global_target': 'global_target_index',
               'field': 'field_id', 'site': 'site_id',
               'observed': 'observed_percent', 'prediction': 'prediction_percent'}
    return frame.rename(columns={k: v for k, v in aliases.items()
                                 if k in frame and v not in frame}).copy()


def _scores(frame):
    w = frame.scientific_weight.to_numpy(float)
    y = frame.observed_percent.to_numpy(float)
    p = frame.prediction_percent.to_numpy(float)
    weight_sum = w.sum()
    mean = np.average(y, weights=w)
    sse = np.sum(w * (p-y)**2)
    sst = np.sum(w * (y-mean)**2)
    return dict(n=len(frame), weight_sum=weight_sum,
                rmse=float(np.sqrt(sse/weight_sum)),
                mae=float(np.average(np.abs(p-y), weights=w)),
                bias=float(np.average(p-y, weights=w)),
                r2=float(1-sse/sst) if sst > 0 else np.nan,
                observed_mean=float(mean), observed_sd=float(np.sqrt(sst/weight_sum)))


def _coordinates(points, observations):
    required = ['global_target_index', 'field_id', 'site_id', 'date',
                'latitude', 'longitude']
    missing = sorted(set(required)-set(observations.columns))
    if missing:
        raise ValueError(f'Observation ledger is missing columns: {missing}')
    if observations.global_target_index.duplicated().any():
        raise ValueError('Observation ledger target identifiers must be unique')
    if observations[['site_id', 'latitude', 'longitude']].isna().any().any():
        raise ValueError('Missing site coordinates in observation ledger')
    coords = observations.groupby('site_id')[['latitude', 'longitude']].nunique()
    if (coords > 1).any().any():
        raise ValueError('Site coordinates must be unique; no coordinate averaging is permitted')
    for identity in ['field_id', 'site_id', 'date', 'observed_percent']:
        if identity not in observations:
            continue
        reference = points[['global_target_index', identity]].merge(
            observations[['global_target_index', identity]], on='global_target_index',
            how='left', validate='many_to_one', suffixes=('', '_ledger'))
        if identity == 'date':
            same = pd.to_datetime(reference[identity]).eq(
                pd.to_datetime(reference[identity+'_ledger']))
        else:
            same = reference[identity].eq(reference[identity+'_ledger'])
        if not same.all():
            raise ValueError(f'Point {identity} disagrees with the observation ledger')
    metadata = [c for c in ['latitude', 'longitude', 'coordinate_year',
                            'measurement', 'leaf_definition',
                            'measurement_qualified', 'anatomical_leaf_verified',
                            'recorded_leaf_category'] if c in observations]
    add = [c for c in metadata if c not in points]
    result = points.merge(observations[['global_target_index', *add]],
                          on='global_target_index', how='left', sort=False,
                          validate='many_to_one')
    for coordinate in ['latitude', 'longitude']:
        expected = observations.set_index('global_target_index')[coordinate]
        if not result[coordinate].eq(result.global_target_index.map(expected)).all():
            raise ValueError('Point coordinates disagree with the observation ledger')
    return result


def _metric_resamples(points):
    """The same site multiplicity is applied to both fixed-forecast designs."""
    sites = sorted(points.site_id.unique())
    position = {site: i for i, site in enumerate(sites)}
    multiplicity = np.random.default_rng(BOOTSTRAP_SEED).multinomial(
        len(sites), np.full(len(sites), 1/len(sites)), size=BOOTSTRAP_REPLICATES)
    results = []
    for evaluation in EVALUATIONS:
        frame = points[points.evaluation.eq(evaluation)]
        pos = frame.site_id.map(position).to_numpy()
        w = frame.scientific_weight.to_numpy(float)
        y = frame.observed_percent.to_numpy(float)
        p = frame.prediction_percent.to_numpy(float)
        # Site totals retain each assessment's original relative hierarchy mass.
        totals = np.stack([np.bincount(pos, weights=value, minlength=len(sites))
                           for value in [w, w*y, w*y*y, w*(p-y)**2,
                                         w*np.abs(p-y), w*(p-y)]], axis=1)
        sampled = multiplicity @ totals
        denominator = sampled[:, 0]
        mean = sampled[:, 1]/denominator
        sst = np.maximum(sampled[:, 2]-sampled[:, 1]**2/denominator, 0.)
        r2 = np.full(BOOTSTRAP_REPLICATES, np.nan)
        np.divide(sampled[:, 3], sst, out=r2, where=sst > 1e-12)
        r2 = 1-r2
        results.append(pd.DataFrame(dict(
            evaluation=evaluation, replicate=np.arange(BOOTSTRAP_REPLICATES),
            bootstrap_seed=BOOTSTRAP_SEED, sampled_sites=len(sites),
            distinct_sites=(multiplicity > 0).sum(axis=1), weight_sum=denominator,
            observed_mean=mean, observed_variance=sst/denominator,
            sse=sampled[:, 3], sst=sst, rmse=np.sqrt(sampled[:, 3]/denominator),
            mae=sampled[:, 4]/denominator, bias=sampled[:, 5]/denominator, r2=r2)))
    return pd.concat(results, ignore_index=True)


def _r2_audit(frame):
    score = _scores(frame)
    w = frame.scientific_weight.to_numpy(float)
    y = frame.observed_percent.to_numpy(float)
    p = frame.prediction_percent.to_numpy(float)
    y_mean = np.average(y, weights=w)
    p_mean = np.average(p, weights=w)
    y_var = np.average((y-y_mean)**2, weights=w)
    p_var = np.average((p-p_mean)**2, weights=w)
    cov = np.average((y-y_mean)*(p-p_mean), weights=w)
    raw_sst = np.sum((y-y.mean())**2)
    return dict(n=len(frame), fields=frame.field_id.nunique(), sites=frame.site_id.nunique(),
                weight_sum=w.sum(), mse=score['rmse']**2, var_y=y_var,
                observed_mean=y_mean, observed_sd=np.sqrt(y_var),
                sse=np.sum(w*(p-y)**2), sst=np.sum(w*(y-y_mean)**2),
                predictive_r2=score['r2'],
                raw_unweighted_r2=1-np.sum((p-y)**2)/raw_sst if raw_sst > 0 else np.nan,
                weighted_correlation_squared=cov**2/(y_var*p_var)
                    if y_var*p_var > 0 else np.nan,
                raw_correlation_squared=np.corrcoef(y, p)[0, 1]**2
                    if np.std(y)*np.std(p) > 0 else np.nan)


def prepare_validation(points, observations):
    """Return source points and distinct diagnostics without altering forecasts.

    Density has all 40 × 40 cells, raw record counts and original scientific
    weight mass; its percentage mass sums to 100 separately in each design.
    Fields and sites retain original weights within each subgroup. The field
    RMSE ECDF instead gives each field-season one vote.
    """
    points, observations = _canonical(points), _canonical(observations)
    required = {'evaluation', 'global_target_index', 'field_id', 'site_id', 'date',
                'observed_percent', 'prediction_percent', 'scientific_weight'}
    if missing := sorted(required-set(points.columns)):
        raise ValueError(f'Prediction table is missing columns: {missing}')
    if set(points.evaluation.unique()) != set(EVALUATIONS):
        raise ValueError(f'Expected evaluation designs {EVALUATIONS}')
    if points[list(required)].isna().any().any():
        raise ValueError('Missing fixed-forecast values or identifiers')
    values = points[['observed_percent', 'prediction_percent', 'scientific_weight']]
    if not np.isfinite(values.to_numpy(float)).all() or (points.scientific_weight <= 0).any():
        raise ValueError('Forecast values must be finite and scientific weights positive')
    if ((values.iloc[:, :2] < 0) | (values.iloc[:, :2] > 100)).any().any():
        raise ValueError('Disease observations and forecasts must be in [0, 100] percent')
    if points.duplicated(['evaluation', 'global_target_index']).any():
        raise ValueError('Each target must appear exactly once per design')
    reference = points[points.evaluation.eq(EVALUATIONS[0])].set_index('global_target_index')
    for evaluation in EVALUATIONS[1:]:
        other = points[points.evaluation.eq(evaluation)].set_index('global_target_index')
        if set(reference.index) != set(other.index):
            raise ValueError('Evaluation designs must retain identical assessment targets')
        columns = ['field_id', 'site_id', 'date', 'observed_percent', 'scientific_weight']
        if not reference[columns].sort_index().equals(other[columns].sort_index()):
            raise ValueError('Evaluation designs must retain identical targets and scientific weights')
    points = _coordinates(points, observations)
    # Outcome strata can have zero target variance; their descriptive R² is
    # undefined and is never displayed. The inherited diagnostics retains this
    # convention without emitting arithmetic warnings for valid constant strata.
    with np.errstate(divide='ignore', invalid='ignore'):
        points, calibration, bias, contexts = diagnostics(points, observations)
    resamples = _metric_resamples(points)
    density, scores, audit, fields, sites = [], [], [], [], []
    lo, hi = np.meshgrid(EDGES[:-1], EDGES[:-1], indexing='ij')
    for evaluation in EVALUATIONS:
        frame = points[points.evaluation.eq(evaluation)]
        raw, *_ = np.histogram2d(frame.observed_percent, frame.prediction_percent, bins=(EDGES, EDGES))
        mass, *_ = np.histogram2d(frame.observed_percent, frame.prediction_percent,
                                 bins=(EDGES, EDGES), weights=frame.scientific_weight)
        density.append(pd.DataFrame(dict(
            evaluation=evaluation, observed_lower=lo.ravel(), observed_upper=(lo+2.5).ravel(),
            predicted_lower=hi.ravel(), predicted_upper=(hi+2.5).ravel(),
            assessments=raw.ravel().astype(int), scientific_weight_mass=mass.ravel(),
            scientific_mass_percent=100*mass.ravel()/frame.scientific_weight.sum())))
        result = dict(evaluation=evaluation, records=len(frame),
                      fields=frame.field_id.nunique(), sites=frame.site_id.nunique(), **_scores(frame))
        reps = resamples[resamples.evaluation.eq(evaluation)]
        for metric in ['rmse', 'r2']:
            result[metric+'_lower95'], result[metric+'_upper95'] = np.nanquantile(reps[metric], [.025, .975])
        result.update(bootstrap_replicates=BOOTSTRAP_REPLICATES,
                      bootstrap_seed=BOOTSTRAP_SEED,
                      interval='descriptive paired-site percentile interval, fixed forecasts')
        scores.append(result)
        audit.append(dict(evaluation=evaluation, **_r2_audit(frame)))
        for field_id, part in frame.groupby('field_id', sort=True):
            if part.site_id.nunique() != 1:
                raise ValueError('A field-season must belong to one site')
            fields.append(dict(evaluation=evaluation, field_id=field_id, site_id=part.site_id.iloc[0],
                               latitude=part.latitude.iloc[0], longitude=part.longitude.iloc[0],
                               scientific_weight_mass=part.scientific_weight.sum(), **_scores(part)))
        for site_id, part in frame.groupby('site_id', sort=True):
            sites.append(dict(evaluation=evaluation, site_id=site_id,
                              latitude=part.latitude.iloc[0], longitude=part.longitude.iloc[0],
                              fields=part.field_id.nunique(), **_scores(part)))
    return dict(points=points, density=pd.concat(density, ignore_index=True),
                scores=pd.DataFrame(scores), metric_resamples=resamples,
                calibration=calibration, bias=bias, contexts=contexts,
                fields=pd.DataFrame(fields), sites=pd.DataFrame(sites), r2_audit=pd.DataFrame(audit))


def _box(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(.8)
        spine.set_edgecolor('#353B40')
    ax.tick_params(labelsize=12)


def _label(ax, letter):
    ax.text(0., 1.04, f'({letter})', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=12)


def _legend(fig, y=.995):
    handles = [Line2D([], [], color=COLORS[key], marker=MARKERS[key],
                      linestyle='-' if i == 0 else '--', linewidth=1.1,
                      markersize=4.5, label=LABELS[key]) for i, key in enumerate(EVALUATIONS)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, y),
               ncol=2, frameon=False, fontsize=12, columnspacing=1.5, handletextpad=.5)


def density_figure(data):
    """Weight-consistent assessment density; performance belongs in the caption."""
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(19/2.54, 10.8/2.54), layout='constrained')
        outer = fig.add_gridspec(2, 2, height_ratios=[1, .16], hspace=.035, wspace=.10)
        positive = data['density'].scientific_mass_percent
        positive = positive[positive > 0]
        norm = LogNorm(vmin=positive.min(), vmax=positive.max())
        cmap = LinearSegmentedColormap.from_list('weighted_density',
            ['#D5E4ED', '#8FBBD3', '#397DA6', '#073B61'])
        for column, key in enumerate(EVALUATIONS):
            frame = data['points'][data['points'].evaluation.eq(key)]
            grid = outer[0, column].subgridspec(2, 2, width_ratios=[1, .105],
                height_ratios=[.105, 1], hspace=.025, wspace=.025)
            ax = fig.add_subplot(grid[1, 0])
            top = fig.add_subplot(grid[0, 0], sharex=ax)
            right = fig.add_subplot(grid[1, 1], sharey=ax)
            density = data['density'][data['density'].evaluation.eq(key)]
            mass = density.sort_values(['observed_lower', 'predicted_lower']).scientific_mass_percent.to_numpy().reshape(40, 40)
            mesh = ax.pcolormesh(EDGES, EDGES, np.ma.masked_equal(mass.T, 0),
                                cmap=cmap, norm=norm, rasterized=True)
            ax.plot([0, 100], [0, 100], ls='--', color='#555555', linewidth=1)
            ax.set(xlim=(0, 100), ylim=(0, 100), aspect='equal',
                   xticks=[0, 25, 50, 75, 100], yticks=[0, 25, 50, 75, 100],
                   xlabel='Observed disease (%)', ylabel='Predicted disease (%)')
            weights = 100*frame.scientific_weight/frame.scientific_weight.sum()
            top.hist(frame.observed_percent, bins=EDGES, weights=weights,
                     color=COLORS[key], edgecolor='none')
            right.hist(frame.prediction_percent, bins=EDGES, weights=weights,
                       orientation='horizontal', color=COLORS[key], edgecolor='none')
            top.set_yticks([])
            top.tick_params(axis='x', bottom=False, labelbottom=False)
            right.set_xticks([])
            right.tick_params(axis='y', left=False, labelleft=False)
            top.set_box_aspect(.105)
            right.set_box_aspect(1/.105)
            for subax in [ax, top, right]:
                _box(subax)
            _label(top, 'ab'[column])
        cax = fig.add_subplot(outer[1, :])
        cax.set_axis_off()
        bar_ax = cax.inset_axes([.16, .59, .68, .24])
        colorbar = fig.colorbar(mesh, cax=bar_ax, orientation='horizontal')
        colorbar.set_label('Share of evaluation weight (%)', fontsize=12, labelpad=4)
        ticks = [t for t in [.001, .01, .1, 1, 10, 100] if norm.vmin <= t <= norm.vmax]
        colorbar.set_ticks(ticks, labels=[f'{t:g}' for t in ticks])
        colorbar.minorticks_off()
        return fig


def diagnostic_figure(data):
    """Compact outcome bias, field spread and paired visit-context diagnostics."""
    with plt.rc_context(STYLE):
        fig=plt.figure(figsize=(19/2.54,14/2.54))
        a=fig.add_axes([.12,.58,.34,.32]);b=fig.add_axes([.64,.58,.33,.32])
        c=fig.add_axes([.27,.13,.70,.28])
        a.axhline(0,color='#858B8F',lw=.8)
        order=['First visit','Intermediate','Last visit','Only visit']
        context=data['contexts'].pivot(index='visit_context',columns='evaluation',values='rmse').reindex(order)
        for i in range(len(order)):
            c.plot([context.iloc[i]['temporal'],context.iloc[i]['spatial_50km']],[i,i],color='#BCC3C7',lw=1.2,zorder=1)
        for i,key in enumerate(EVALUATIONS):
            colour,marker=COLORS[key],MARKERS[key];style='-' if i==0 else '--'
            q=data['bias'][data['bias'].evaluation.eq(key)].sort_values('bin_lower')
            a.plot(q.observed_mean,q.bias,color=colour,marker=marker,ls=style,ms=4,lw=1.15)
            q=data['fields'][data['fields'].evaluation.eq(key)].sort_values('rmse')
            b.step(np.r_[0,q.rmse],np.r_[0,100*np.arange(1,len(q)+1)/len(q)],where='post',color=colour,ls=style,lw=1.25)
            q=data['contexts'][data['contexts'].evaluation.eq(key)].set_index('visit_context').reindex(order)
            c.scatter(q.rmse,np.arange(len(order)),color=colour,marker=marker,s=29,zorder=3)
        a.set(xlim=(0,100),xticks=[0,25,50,75,100],xlabel='Mean observed disease (%)',ylabel='Prediction bias (pp)')
        b.set(xlim=(0,data['fields'].rmse.max()*1.03),ylim=(0,102),yticks=[0,25,50,75,100],
            xlabel='Field-season RMSE (pp)',ylabel='Field-seasons (%)')
        counts=data['contexts'][data['contexts'].evaluation.eq('temporal')].set_index('visit_context')['n']
        c.set(xlim=(14,22),xticks=[14,16,18,20,22],ylim=(3.55,-.55),yticks=range(4),
            yticklabels=[label+f'\n(n={int(counts[label]):,})' for label in order],xlabel='RMSE (pp)')
        c.grid(axis='x',color='#E3E6E8',lw=.6);c.set_axisbelow(True)
        for ax,letter in zip([a,b,c],'abc'):_box(ax);_label(ax,letter)
        _legend(fig)
        return fig


def raw_scatter_figure(data):
    """Every original assessment has the same marker size and opacity per panel."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(19/2.54, 10.2/2.54), layout='constrained')
        for ax, key, letter in zip(axes, EVALUATIONS, 'ab'):
            frame = data['points'][data['points'].evaluation.eq(key)]
            ax.scatter(frame.observed_percent, frame.prediction_percent, s=9,
                       marker=MARKERS[key], color=COLORS[key], alpha=.32,
                       linewidths=0, rasterized=True)
            ax.plot([0, 100], [0, 100], '--', color='#555555', linewidth=1)
            ax.set(xlim=(0, 100), ylim=(0, 100), aspect='equal',
                   xticks=[0, 25, 50, 75, 100], yticks=[0, 25, 50, 75, 100],
                   xlabel='Observed disease (%)', ylabel='Predicted disease (%)')
            _box(ax)
            _label(ax, letter)
        return fig
