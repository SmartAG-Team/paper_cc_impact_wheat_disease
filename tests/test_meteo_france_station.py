import numpy as np
import pandas as pd
import pytest

from models.monitoring.meteo_france_station import parse_hourly, daily_aggregate


def sample():
    times = pd.date_range('2015-06-01',periods=25,freq='h')
    return pd.DataFrame(dict(NUM_POSTE='80001001',AAAAMMJJHH=times.strftime('%Y%m%d%H'),
        RR1=[99]+[1]*23+[5],QRR1=1,T=10.,QT=1,U=92.,QU=1))


def test_accumulation_ending_midnight_completes_previous_day_without_unit_rescaling():
    h = parse_hourly(sample())
    daily = daily_aggregate(h).set_index('date')
    row = daily.loc[pd.Timestamp('2015-06-01')]
    assert row.precipitation_mm == 28
    assert row.station_precip_hours_gt_0_1mm == 24
    assert row.tmean_c == 10 and row.rh_hours_ge_90pct == 24
    # One observed hour on the next day is insufficient; missing is not zero.
    assert np.isnan(daily.loc[pd.Timestamp('2015-06-02'),'tmean_c'])


def test_unvalidated_or_missing_hours_prevent_daily_substitution_for_that_variable():
    raw = sample()
    raw.loc[7,'QRR1'] = 9
    raw.loc[8,'QU'] = 2
    raw.loc[9,'QT'] = 0  # Definitively validated remains eligible.
    h = parse_hourly(raw)
    row = daily_aggregate(h).set_index('date').loc[pd.Timestamp('2015-06-01')]
    assert np.isnan(row.precipitation_mm)
    assert np.isnan(row.station_precip_hours_gt_0_1mm)
    assert np.isnan(row.rh_mean_pct)
    assert np.isnan(row.rh_hours_ge_90pct)
    assert row.tmean_c == 10


def test_duplicate_station_hour_is_rejected():
    raw = sample()
    with pytest.raises(ValueError,match='Duplicate'):
        parse_hourly(pd.concat([raw,raw.iloc[[1]]]))


def test_precipitation_duration_origin_tracks_only_supported_substitutions():
    from models.monitoring.repairs import hybrid_weather
    dates = pd.date_range('2015-06-01',periods=3)
    weather = pd.DataFrame(dict(location_id='s',date=dates,precipitation_mm=2.,
        rain_mm=2.,tmin_c=5.,rain_hours_gt_0_1mm=2.,
        origin_rain_hours_gt_0_1mm=['DWD','reanalysis','reanalysis']))
    pairs = pd.DataFrame(dict(site_id='s',date=dates,precipitation_mm_station=[0.,3.,np.nan],
        station_precip_hours_gt_0_1mm=[0.,4.,np.nan]))
    changed,ledger = hybrid_weather(weather,pairs,{'s'},rainfall=True,
                                    meteorology=False,source='METEO_FRANCE')
    assert changed.rain_hours_gt_0_1mm.tolist()==[0.,4.,2.]
    assert changed.origin_rain_hours_gt_0_1mm.tolist()==['METEO_FRANCE','METEO_FRANCE','reanalysis']
    assert ledger.loc[ledger.variable.eq('rain_hours_gt_0_1mm'),'replaced_days'].item()==2


@pytest.mark.parametrize('module,receipt_name',[
    ('meteo_france_station','receipt.json'),('meteo_france_timing','timing_receipt.json')])
def test_source_collection_preserves_existing_scientific_receipts(tmp_path,module,receipt_name):
    import importlib
    collector = importlib.import_module('models.monitoring.'+module)
    out = tmp_path/'frozen';out.mkdir()
    receipt = out/receipt_name;receipt.write_text('original archived receipt')
    with pytest.raises(FileExistsError,match='new output'):
        collector.run(root=tmp_path,output=out)
    assert receipt.read_text()=='original archived receipt'


def test_new_verifier_version_preserves_receipts_already_bound_to_exported_data(tmp_path):
    from models.monitoring.verify_french_weather import write_verification_receipt
    from models.monitoring.inputs import sha
    first_code = tmp_path/'first.py';first_code.write_text('first verifier')
    second_code = tmp_path/'second.py';second_code.write_text('second verifier')
    output = tmp_path/'evidence'
    first = dict(verified=True,verifier_sha256=sha(first_code))
    second = dict(verified=True,verifier_sha256=sha(second_code))
    write_verification_receipt(output,first,first_code,spatial=True)
    original_receipt = (output/'verification.json').read_bytes()
    original_spatial = (output/'spatial_refit_verification.json').read_bytes()
    new_path = write_verification_receipt(output,second,second_code,spatial=True)
    assert new_path.exists()
    assert (output/'verification.json').read_bytes()==original_receipt
    assert (output/'spatial_refit_verification.json').read_bytes()==original_spatial
    assert sha(output/'verification_code_snapshot.py')==first['verifier_sha256']
