"""Field-level flag-stage predictions with interval-censored observations."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SOURCES = {
    'reused_development_2019': dict(label='BASF 2019', color='#48545d', marker='o', open=True),
    'reused_external_strict': dict(label='Corteva external', color='#235e83', marker='s', open=False),
}


def field_intervals(study):
    """Retain actual daily admissible bounds, never treating centres as truth."""
    path = Path(study)/'phenology/stage_censoring_scores.csv'
    frame = pd.read_csv(path)
    frame = frame.loc[frame.genuinely_bracketed & frame.event.isin([37, 39])
        & frame.partition.isin(SOURCES)].copy()
    for column in ('lower_exclusive', 'upper_inclusive', 'predicted_date'):
        frame[column] = pd.to_datetime(frame[column])
    frame['observed_first_possible_date'] = frame.lower_exclusive+pd.Timedelta(days=1)
    frame['observed_last_possible_date'] = frame.upper_inclusive
    assert frame[['observed_first_possible_date', 'observed_last_possible_date', 'predicted_date']].notna().all().all()
    assert frame.observed_first_possible_date.le(frame.observed_last_possible_date).all()
    for column in ('observed_first_possible_date', 'observed_last_possible_date', 'predicted_date'):
        assert frame[column].dt.year.eq(frame.season_year).all()
    frame['observed_lower_doy'] = frame.observed_first_possible_date.dt.dayofyear
    frame['observed_upper_doy'] = frame.observed_last_possible_date.dt.dayofyear
    frame['predicted_doy'] = frame.predicted_date.dt.dayofyear
    frame['interval_centre_doy_for_display'] = (frame.observed_lower_doy+frame.observed_upper_doy)/2.
    # Independent integer-day compatibility and excess-distance reconstruction.
    frame['display_compatible'] = frame.predicted_date.between(
        frame.observed_first_possible_date, frame.observed_last_possible_date)
    lower = (frame.observed_first_possible_date-frame.predicted_date).dt.days
    upper = (frame.predicted_date-frame.observed_last_possible_date).dt.days
    frame['display_interval_excess_days'] = np.maximum(np.maximum(lower, upper), 0)
    assert frame.display_compatible.eq(frame.compatible).all()
    np.testing.assert_array_equal(frame.display_interval_excess_days, frame.distance_days)
    metrics = pd.read_csv(Path(study)/'phenology/stage_censoring_metrics.csv')
    for (partition, event), group in frame.groupby(['partition', 'event']):
        row = metrics.loc[metrics.partition.eq(partition) & metrics.event.eq(event)
            & metrics.scope.eq('genuine_two_sided')].iloc[0]
        assert len(group) == row.n_field_events
        assert np.isclose(group.display_compatible.mean(), row.compatible_fraction)
        assert np.isclose(group.display_interval_excess_days.mean(), row.mean_distance_days)
    columns = ['field_id', 'partition', 'source', 'season_year', 'event',
        'lower_exclusive', 'observed_first_possible_date', 'observed_last_possible_date',
        'predicted_date', 'observed_lower_doy', 'observed_upper_doy', 'predicted_doy',
        'interval_centre_doy_for_display', 'display_compatible', 'display_interval_excess_days']
    return frame[columns].sort_values(['event', 'partition', 'field_id']).reset_index(drop=True)


def render_flag_stages(study, destination, export):
    frame = field_intervals(study)
    destination = Path(destination)
    frame.to_csv(destination/'fig2_field_stage_intervals.csv', index=False)
    minimum = min(frame.observed_lower_doy.min(), frame.predicted_doy.min())
    maximum = max(frame.observed_upper_doy.max(), frame.predicted_doy.max())
    limits = (5*np.floor((minimum-3)/5), 5*np.ceil((maximum+3)/5))
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.95), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.985, bottom=.20, top=.845, wspace=.14)
    for ax, stage, letter, title in zip(axes, [37, 39], ['a', 'b'],
            ['Flag leaf visible', 'Flag leaf fully unfolded']):
        ax.plot(limits, limits, color='#8a8a8a', linestyle='--', linewidth=1., zorder=1)
        selected = frame.loc[frame.event.eq(stage)]
        notes = []
        for source, style in SOURCES.items():
            group = selected.loc[selected.partition.eq(source)]
            # Identical field observations remain in the source table. The
            # plotted multiplicity labels preserve them without artificial jitter.
            grouped = group.groupby(['observed_lower_doy', 'observed_upper_doy', 'predicted_doy'], sort=True)
            for (lower, upper, predicted), repeated in grouped:
                centre = (lower+upper)/2.
                ax.hlines(predicted, lower, upper, color=style['color'], linewidth=1.1, alpha=.42, zorder=2)
                ax.vlines([lower, upper], predicted-.6, predicted+.6,
                    color=style['color'], linewidth=.9, alpha=.65, zorder=2)
                ax.scatter(centre, predicted, s=36+10*(len(repeated)-1), marker=style['marker'],
                    facecolors='white' if style['open'] else style['color'],
                    edgecolors=style['color'], linewidths=1., zorder=3)
                if len(repeated)>1:
                    ax.annotate(f'×{len(repeated)}', (centre, predicted), xytext=(5, 4),
                        textcoords='offset points', fontsize=8.2, color=style['color'], zorder=4)
            inside = int(group.display_compatible.sum())
            excess = group.display_interval_excess_days.mean()
            notes.append(f"{style['label']}: {inside}/{len(group)} within; mean excess {excess:.2f} d")
        ax.set(xlim=limits, ylim=limits)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'{letter}   BBCH{stage}: {title}', loc='left', fontsize=10.2, pad=10, weight='bold')
        ax.set_xlabel('Observed stage window (day of year)', fontsize=10)
        ax.grid(color='#ededed', linewidth=.65, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)
        ax.text(0., -.25, '\n'.join(notes), transform=ax.transAxes, fontsize=8.6,
            color='#3b4145', va='top', ha='left', linespacing=1.6)
    axes[0].set_ylabel('Predicted stage date (day of year)', fontsize=10)
    handles=[Line2D([], [], marker=s['marker'], linestyle='none', markersize=5.5,
        markerfacecolor='white' if s['open'] else s['color'], markeredgecolor=s['color'], label=s['label'])
        for s in SOURCES.values()]
    handles.append(Line2D([], [], color='#8a8a8a', linestyle='--', linewidth=1., label='Equal calendar date'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, .99), ncol=3,
        frameon=False, fontsize=9.4, handlelength=1.8, columnspacing=1.4)
    export(fig, destination, 'fig2_flag_stage_validation')
    return ('Figure 2 | Field-level flag-stage predictions and observed date windows. '
        '(a) BBCH37, flag-leaf visibility; (b) BBCH39, full unfolding. '
        'Horizontal lines span the daily admissible observation interval, from one day after the last below-stage assessment '
        'to the first at-or-above-stage assessment. Symbols lie at the interval centres for display; those centres are not observed event dates. '
        'The dashed diagonal indicates equal calendar date: a horizontal interval crossing the diagonal contains the predicted date. '
        'Open circles denote BASF2019 and filled squares the location-disjoint Corteva subset. '
        'Identical plotted windows/predictions are labelled by their field multiplicity. '
        'Counts within intervals and mean interval-excess distance accompany each panel; zero excess does not establish exact-date accuracy. '
        'All genuinely two-sided reused-validation records are shown: BBCH37 n=3/19 and BBCH39 n=4/30 for BASF/Corteva. '
        'Observation intervals are neither confidence intervals nor model prediction intervals.')
