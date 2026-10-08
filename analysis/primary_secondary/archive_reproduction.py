"""Retrospective replay of v2 states and external baseline coefficients."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from calibration.primary_secondary.calibrate import BASELINES, fit_baseline, mechanism_prediction, prediction_frame, score, baseline_prediction
from calibration.primary_secondary.field_data import prepare_basf, FieldBatch, canopy_snapshot, weather_window

ROOT=Path(__file__).resolve().parents[2]
BASF=ROOT/'analysis/primary_secondary/calibration_v2_age'
FRENCH=ROOT/'analysis/primary_secondary/external_french'
OUT=BASF/'reproduction_archive'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def french_batch():
    episodes=pd.read_csv(FRENCH/'episodes.csv')
    targets=pd.read_csv(FRENCH/'targets.csv')
    past=pd.read_csv(FRENCH/'all_plot_leaf_observations.csv')
    episodes['start']=pd.to_datetime(episodes.start);episodes['end']=pd.to_datetime(episodes.end)
    past['Date']=pd.to_datetime(past.Date)
    weather=pd.read_parquet(ROOT/'data/era5/daily_weather.parquet');weather['date']=pd.to_datetime(weather.date)
    n=len(episodes);days=int(targets.day.max())
    initial=np.zeros((n,3));active=np.zeros((n,3),bool);ages=np.zeros((n,3))
    temperature=np.full((n,days),18.);humidity=np.zeros_like(temperature);rain=np.zeros_like(temperature)
    for i,e in episodes.iterrows():
        initial[i],active[i],ages[i]=canopy_snapshot(past.loc[past.n_numeric.gt(0)],e.TrialId,e.start,
            np.arange(1,4),return_age=True)
        w=weather_window(weather,e.location_id,e.start,e.end)
        temperature[i,:len(w)]=w.tmean_c;humidity[i,:len(w)]=w.rh_hours_ge_90pct;rain[i,:len(w)]=w.rain_mm
    return FieldBatch(episodes,targets,initial,active,temperature,humidity,rain,np.arange(1,4),ages)


def main():
    OUT.mkdir(exist_ok=True)
    validation=json.loads((BASF/'validation.json').read_text())
    for name,h in validation['source_sha256'].items():
        if digest(ROOT/name)!=h:raise ValueError('v2 source/code hash mismatch: '+name)
    sources=[BASF/'predictions.csv',BASF/'fits.json',BASF/'all_development_fit.json',
        FRENCH/'transfer_v2/frozen_transfer_predictions.csv',FRENCH/'episodes.csv',FRENCH/'targets.csv',
        FRENCH/'all_plot_leaf_observations.csv']
    hashes={str(p.relative_to(ROOT)):digest(p) for p in sources}
    batch=prepare_basf(ROOT);frozen=pd.read_csv(BASF/'predictions.csv')
    assignments=pd.read_csv(BASF/'fold_assignments.csv')
    fits=json.loads((BASF/'fits.json').read_text());main_fits=[f for f in fits if f['model']=='primary_secondary_hidden']
    checks=[]
    for fit in main_fits:
        ids=set(assignments.loc[assignments.fold.eq(fit['fold'])&assignments.side.eq('test'),'series_id'])
        test=batch.subset(ids);trajectory=mechanism_prediction(test,fit)
        prediction=prediction_frame(test,fit)
        original=frozen.loc[frozen.fold.eq(fit['fold'])&frozen.model.eq(fit['model'])]
        check=prediction[['source_row','predicted_percent']].merge(original[['source_row','predicted_percent']],
            on='source_row',validate='one_to_one',suffixes=('_replay','_original'))
        difference=float(abs(check.predicted_percent_replay-check.predicted_percent_original).max())
        if difference>1e-8:raise ValueError('Replayed main predictions differ: '+fit['fold'])
        np.savez_compressed(OUT/(fit['fold']+'_main_states.npz'),state=trajectory.state,damage=trajectory.damage,
            pycnidia=trajectory.pycnidia,infectious=trajectory.infectious,origin_total=trajectory.origin_total,
            primary_flux=trajectory.primary_flux,secondary_flux=trajectory.secondary_flux,
            series_id=test.episodes.series_id.to_numpy(str),ranks=test.ranks,active=test.active)
        checks.append({'fold':fit['fold'],'n_predictions':len(check),'max_prediction_difference_pp':difference,
            'max_mass_error':float(abs(trajectory.state.sum(axis=-1)-1).max()),'minimum_state':float(trajectory.state.min())})
    full=json.loads((BASF/'all_development_fit.json').read_text())
    trajectory=mechanism_prediction(batch,full);fine=mechanism_prediction(batch,full,.125)
    np.savez_compressed(OUT/'all_development_main_states.npz',state=trajectory.state,damage=trajectory.damage,
        pycnidia=trajectory.pycnidia,infectious=trajectory.infectious,origin_total=trajectory.origin_total,
        primary_flux=trajectory.primary_flux,secondary_flux=trajectory.secondary_flux,
        eighth_day_state=fine.state,series_id=batch.episodes.series_id.to_numpy(str),ranks=batch.ranks,active=batch.active)
    baseline_fits=[fit_baseline(batch,name) for name in BASELINES]
    (OUT/'all_development_baseline_fits.json').write_text(json.dumps(baseline_fits,indent=2)+'\n')
    external=french_batch();old=pd.read_csv(FRENCH/'transfer_v2/frozen_transfer_predictions.csv')
    rows=external.targets.loc[~external.targets.conditioning].copy()
    replay=[];baseline_checks=[]
    for fit in baseline_fits:
        p=rows[['series_id','Date']].copy();p['predicted_percent']=100*baseline_prediction(external,fit)
        p['model']=fit['model']+'_BASF_frozen'
        matched=p.merge(old.loc[old.model.eq(p.model.iloc[0]),['series_id','Date','predicted_percent']],
            on=['series_id','Date'],validate='one_to_one',suffixes=('_replay','_original'))
        difference=float(abs(matched.predicted_percent_replay-matched.predicted_percent_original).max())
        if difference>1e-8:raise ValueError('External statistical baseline replay differs: '+fit['model'])
        baseline_checks.append({'model':fit['model'],'n_targets':len(p),'max_prediction_difference_pp':difference})
        replay.append(p)
    pd.concat(replay,ignore_index=True).to_csv(OUT/'replayed_french_baseline_predictions.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'main_state_replay_checks.csv',index=False)
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'role':'retrospective exact reproduction; not contemporaneous original execution',
        'source_sha256':hashes,'all_sources_unchanged':all(digest(ROOT/p)==h for p,h in hashes.items()),
        'outer_main_state_replays':checks,'external_baseline_replays':baseline_checks,
        'development_fit_rmse_pp':score(prediction_frame(batch,full))['rmse_pp'],
        'archived_fit_rmse_pp':full['training_rmse_pp'],
        'max_quarter_vs_eighth_day_damage_difference_pp':float(100*abs(trajectory.damage-fine.damage).max()),
        'recorded_time_step_difference_pp':validation['max_quarter_vs_eighth_day_damage_difference_pp'],
        'natural_primary_onset_validated':False,'publication_ready':False}
    (OUT/'replay_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['source_sha256','outer_main_state_replays']},indent=2))


if __name__=='__main__':main()
