"""Top-three leaf timing, functional green-area loss and conditional yield transfer.

Infection presence is not functional area loss. The area-loss operator requires
an explicit reference lamina area index and daily functional loss fraction.
Published HAD/yield slopes transfer conditionally; their ranges are not CIs.
"""

import numpy as np
import pandas as pd


def _dates(values):
    return pd.DatetimeIndex(pd.to_datetime(values)).to_numpy(dtype='datetime64[D]')


def _window(dates, starts, ends, fields):
    index=pd.DatetimeIndex(dates)
    if (not len(index) or index.hasnans or index.has_duplicates
            or not index.equals(pd.date_range(index[0],index[-1]))):
        raise ValueError('Complete consecutive daily dates are required.')
    days=_dates(index);start=_dates(starts);end=_dates(ends)
    if start.shape!=(fields,) or end.shape!=(fields,):
        raise ValueError('Each field requires a start and end date.')
    present=~np.isnat(start)&~np.isnat(end)
    if np.any(present&(end<start)):
        raise ValueError('Crop-window date order must be start before end.')
    complete=present&(start>=days[0])&(end<=days[-1])
    mask=(days[None,:]>=start[:,None])&(days[None,:]<=end[:,None])&complete[:,None]
    return days,mask,complete


def top_three_infection_overlap(dates,infection_dates,anthesis_dates,end_dates):
    """Report infected time during the declared grain-fill proxy, by leaf.

    Indicators persist after the first infection. They describe exposure of
    grain filling to infection, not damaged leaf area or a yield-loss fraction.
    """
    infection=np.asarray(infection_dates,dtype='datetime64[D]')
    if infection.ndim!=2 or infection.shape[1]<3:
        raise ValueError('Infection dates for the top three leaves are required.')
    infection=infection[:,:3]
    days,window,complete=_window(dates,anthesis_dates,end_dates,len(infection))
    infected=(days[None,:,None]>=infection[:,None,:])&~np.isnat(infection[:,None,:])
    overlap=infected&window[:,:,None]
    count=overlap.sum(axis=1).astype(float)
    duration=window.sum(axis=1).astype(float)
    fraction=np.divide(count,duration[:,None],out=np.full_like(count,np.nan),where=duration[:,None]>0)
    any_days=(infected.any(axis=2)&window).sum(axis=1).astype(float)
    all_days=(infected.all(axis=2)&window).sum(axis=1).astype(float)
    for value in [count,duration,any_days,all_days]:value[~complete]=np.nan
    return dict(infected_grain_fill_days=count,infected_grain_fill_fraction=fraction,
        grain_fill_days=duration,days_with_any_top3_infected=any_days,
        days_with_all_top3_infected=all_days,complete_window=complete,
        infection_overlap_is_yield_loss=False)


def upper_leaf_had_loss(dates,reference_leaf_area_index,functional_loss_fraction,
                       window_start_dates,window_end_dates):
    """Integrate top-three functional green-area deficit on a daily grid.

    Input area is m² leaf per m² ground, not normalized infection probability.
    Input fractions represent functional chlorotic/necrotic/associated loss,
    with any score-to-area conversion supplied and declared by the caller.
    The daily rectangle sum uses inclusive date boundaries. Outside-window
    loss contributes zero; incomplete windows produce missing results.
    """
    area=np.asarray(reference_leaf_area_index,float)
    loss=np.asarray(functional_loss_fraction,float)
    if (area.ndim!=3 or area.shape[2]<3 or loss.shape!=area.shape
            or area.shape[1]!=len(dates) or not np.isfinite(area).all()
            or not np.isfinite(loss).all() or np.any(area<0)
            or np.any((loss<0)|(loss>1))):
        raise ValueError('Finite nonnegative leaf area and functional fractions in [0,1] required.')
    _,window,complete=_window(dates,window_start_dates,window_end_dates,len(area))
    reference=(area[:,:,:3]*window[:,:,None]).sum(axis=1)
    deficit=(area[:,:,:3]*loss[:,:,:3]*window[:,:,None]).sum(axis=1)
    reference[~complete]=np.nan;deficit[~complete]=np.nan
    total_reference=reference.sum(axis=1);total_deficit=deficit.sum(axis=1)
    fraction=np.divide(total_deficit,total_reference,out=np.full(len(area),np.nan),where=total_reference>0)
    return dict(reference_had3=total_reference,lost_had3=total_deficit,
        reference_had_by_leaf=reference,lost_had_by_leaf=deficit,
        relative_had_loss=fraction,complete_window=complete,
        had_units='m2 green leaf m-2 ground day',yield_calibrated=False)


def transfer_had_yield_loss(lost_had3,slopes,*,reference_yield_t_ha=None):
    """Apply supplied literature slopes with explicit area/yield units.

    Coefficients are t ha⁻¹ per GLAI-day. No reference yield is invented and
    the linear relation is not silently clipped; excessive loss is flagged.
    """
    had=np.atleast_1d(np.asarray(lost_had3,float))
    coefficient=np.atleast_1d(np.asarray(slopes,float))
    if (had.ndim!=1 or coefficient.ndim!=1 or not len(coefficient)
            or np.any(np.isinf(had)) or np.any(had<0)
            or not np.isfinite(coefficient).all() or np.any(coefficient<0)):
        raise ValueError('Nonnegative HAD deficit and finite nonnegative slopes required.')
    absolute=had[:,None]*coefficient[None,:]
    relative,exceeds=None,None
    if reference_yield_t_ha is not None:
        reference=np.broadcast_to(np.asarray(reference_yield_t_ha,float),had.shape)
        if not np.isfinite(reference).all() or np.any(reference<=0):
            raise ValueError('Positive finite independent reference yield required.')
        relative=100*absolute/reference[:,None]
        exceeds=absolute>reference[:,None]
    return dict(absolute_loss_t_ha=absolute,relative_loss_percent=relative,
        exceeds_reference_yield=exceeds,slopes=coefficient,
        conditional_literature_transfer=True,locally_yield_calibrated=False,
        coefficient_range_is_confidence_interval=False)
