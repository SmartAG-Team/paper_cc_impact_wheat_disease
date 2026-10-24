"""Rainfall reconstruction must respect UTC bounds, gaps and interval QC."""
import numpy as np
import pandas as pd
import pytest


def interval(start, hours, amount=1., qc='1', condition='9'):
    start=pd.Timestamp(start)
    return {'start':start,'end':start+pd.Timedelta(hours=hours),'period_hours':hours,
            'amount_mm':amount,'quality_code':qc,'condition_code':condition,
            'trace':condition=='2','accepted':True,'aa_id':'AA1'}


def test_aa_documented_units_missing_and_rejected_quality():
    from models.monitoring.noaa_hourly_rain import parse_aa
    good=parse_aa('01,0012,9,1')
    assert good['amount_mm']==1.2 and good['period_hours']==1 and good['accepted']
    for text in ['99,0012,9,1','01,9999,9,1','00,0012,9,1','01,0012,I,1',
                 '01,0012,J,1','01,0012,E,1','01,0012,9,3','01,0012,9,2']:
        assert not parse_aa(text)['accepted']
    assert parse_aa('01,0000,2,1')['trace']


def test_exact_utc_daily_cover_prefers_hourly_without_double_counting():
    from models.monitoring.noaa_hourly_rain import reconstruct_daily
    intervals=[interval('2017-01-01',24,100.)]
    intervals += [interval(pd.Timestamp('2017-01-01')+pd.Timedelta(hours=h),1,.2) for h in range(24)]
    d=reconstruct_daily(pd.DataFrame(intervals),[pd.Timestamp('2017-01-01')]).iloc[0]
    assert d.precipitation_mm_station==pytest.approx(4.8)
    assert d.complete_utc_day and d.selected_interval_count==24
    assert d.rain_hours_gt_0_1mm_station==24
    assert d.latest_interval_end_utc==pd.Timestamp('2017-01-02')


def test_valid_longer_intervals_fill_hourly_gaps_but_never_imply_wet_hours():
    from models.monitoring.noaa_hourly_rain import reconstruct_daily
    intervals=[interval('2017-01-01',6,3.)]
    intervals += [interval(pd.Timestamp('2017-01-01')+pd.Timedelta(hours=h),1,.2) for h in range(6,24)]
    d=reconstruct_daily(pd.DataFrame(intervals),[pd.Timestamp('2017-01-01')]).iloc[0]
    assert d.precipitation_mm_station==pytest.approx(6.6)
    assert d.hours_from_1h_intervals==18 and d.max_accumulation_hours==6
    assert np.isnan(d.rain_hours_gt_0_1mm_station)


def test_missing_hour_and_overlapping_accumulations_do_not_make_daily_total():
    from models.monitoring.noaa_hourly_rain import reconstruct_daily
    intervals=[interval('2017-01-01',12,4),interval('2017-01-01 06:00',12,4),
               interval('2017-01-01 18:00',5,1)]
    d=reconstruct_daily(pd.DataFrame(intervals),[pd.Timestamp('2017-01-01')]).iloc[0]
    assert not d.complete_utc_day and np.isnan(d.precipitation_mm_station)
    assert d.valid_interval_union_hours==23


def test_off_midnight_24h_report_is_not_shifted_to_fit_day():
    from models.monitoring.noaa_hourly_rain import reconstruct_daily
    intervals=[interval('2017-01-01 06:00',24,10.)]
    d=reconstruct_daily(pd.DataFrame(intervals),[pd.Timestamp('2017-01-01')]).iloc[0]
    assert np.isnan(d.precipitation_mm_station) and not d.complete_utc_day


def test_conflicting_duplicate_intervals_are_rejected_not_selected_by_amount():
    from models.monitoring.noaa_hourly_rain import reconstruct_daily
    intervals=[interval('2017-01-01',24,4),interval('2017-01-01',24,5)]
    d=reconstruct_daily(pd.DataFrame(intervals),[pd.Timestamp('2017-01-01')]).iloc[0]
    assert np.isnan(d.precipitation_mm_station)
    assert d.conflicting_interval_count==1


def test_fixed_width_extracts_only_add_section_and_preserves_snow_indicators():
    from models.monitoring.noaa_hourly_rain import extract_isd
    header=list(' '*105); header[4:15]=list('03862099999');header[15:27]=list('201701020000');header[41:46]=list('FM-12')
    line=''.join(header)+'ADDAA124001291MW1731REMAA101999991'
    observations, weather=extract_isd([line])
    assert len(observations)==1 and observations.iloc[0].amount_mm==1.2
    assert observations.iloc[0].start==pd.Timestamp('2017-01-01')
    assert weather.iloc[0].manual_weather_code==73
    assert weather.iloc[0].solid_precipitation_observed


def test_positive_rejected_depths_with_only_dry_reconstructions_block_substitution():
    from models.monitoring.noaa_hourly_rain import substitution_safety
    counts=pd.DataFrame({'condition_code':['9','3'],'quality_code':['1','1'],
        'accepted':[True,False],'positive_intervals':[0,50],'n_intervals':[100,50]})
    daily=pd.DataFrame({'complete_utc_day':[True,True], 'precipitation_mm_station':[0.,0.]})
    safety=substitution_safety(counts,daily)
    assert not safety['usable_for_substitution']
    assert safety['rejected_positive_intervals']==50
    assert safety['condition3_full_qc_positive_intervals']==50
    assert safety['positive_reconstructed_days']==0
