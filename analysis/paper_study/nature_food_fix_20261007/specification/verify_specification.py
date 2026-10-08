"""Independent arithmetic correspondence of specification and unchanged runtime."""
from dataclasses import replace
from datetime import date, timedelta
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from model.wheat_stb import EngineConfig, WeatherProvider, WheatSTBSimulation
from model.wheat_stb.phenology import PhenologyModel
from model.wheat_stb.leaf import LeafCanopyModel
from model.wheat_stb.weather import WeatherDay
from model.seasonal_septoria.overwinter import _advance_day
from model.seasonal_septoria.wetness import duration_exposure

ROOT = Path(__file__).resolve().parents[4]
DEST = Path(__file__).resolve().parent
CLIMATE = ROOT/'analysis/paper_study/overwinter_leaf_model_20261006/climate'


def advance_equations(state, residue, t, exposure, rain, active, renewal, area,
        imported, p, substeps):
    x=state.copy(); q=residue.copy(); flows=np.zeros((4,8)); m=p.latent_stages
    for i in range(8):
        if active[i]:
            x[i,1:]*=1-renewal[i];x[i,0]=1-x[i,1:].sum()
        else:
            x[i]=0;x[i,0]=1
    eta=1/substeps;theta=max(t,0)/18;splash=-np.expm1(-rain/p.rain_scale_mm)
    probs=-np.expm1(-eta*theta*np.array([m/p.latent_reference_days,
        1/p.nonsporulating_reference_days,1/p.infectious_reference_days]))
    qL,qN,qI=probs
    for _ in range(substeps):
        out=x.copy();denom=sum(area[j] for j in range(8) if active[j])
        for i in range(8):
            if not active[i]:continue
            nearby=sum(math.exp(-abs(i-j)/p.rank_distance_scale)*x[j,m+2]*area[j]
                for j in range(8) if active[j])/denom
            contact=sum(x[j,m+2]*area[j] for j in range(8) if active[j] and abs(i-j)==1)/denom
            hazards=np.array([
                p.primary_scale*exposure*q[1]*((1-p.local_airborne_fraction)*splash*
                    math.exp(-(7-i)/p.rank_distance_scale)+p.local_airborne_fraction),
                p.primary_scale*exposure*imported,
                p.secondary_scale*exposure*splash*nearby,
                p.secondary_scale*exposure*p.contact_fraction*(1-area[i])*contact*(i!=7)])
            total=hazards.sum();new=x[i,0]*(-np.expm1(-eta*total))
            if total:flows[:,i]+=new*hazards/total
            out[i,0]=x[i,0]-new
            out[i,1]=(1-qL)*x[i,1]+new
            for j in range(2,m+1):out[i,j]=(1-qL)*x[i,j]+qL*x[i,j-1]
            out[i,m+1]=(1-qN)*x[i,m+1]+qL*x[i,m]
            out[i,m+2]=(1-qI)*x[i,m+2]+qN*x[i,m+1]
            out[i,m+3]=x[i,m+3]+qI*x[i,m+2]
        mu=theta*exposure/p.residue_maturation_reference_days
        nu=theta/p.residue_decay_reference_days
        q=np.array([q[0]*math.exp(-eta*(mu+nu)),
            math.exp(-eta*nu)*(q[1]+q[0]*(-math.expm1(-eta*mu)))])
        x=out
    return x,q,flows


def sampling_variance(influence, weights, strata):
    result=0.
    for h in np.unique(strata):
        values=influence[strata==h];n=len(values);W=weights[strata==h].sum()
        result+=W**2*np.square(values-values.mean()).sum()/(n*(n-1))
    return result


def main():
    manifest=json.loads((DEST/'source_manifest.json').read_text())
    original={x['path']:x['sha256'] for x in manifest['files']}
    cfg=EngineConfig.from_json(ROOT/'examples/wheat_stb/configuration_standardized_had.json')
    errors={}; counts={}
    # Equations S1–S6, including previous-day gates and current-day V multiplier.
    pheno=PhenologyModel(cfg.phenology,latitude=cfg.latitude,sowing_date=cfg.sowing_date)
    G=U=A=V=0.;maxerr=0.
    temperatures=[-5,0,3,10,16,18,25,31,39]
    for d in range(600):
        t=temperatures[(d//17)%len(temperatures)];tx=t+7
        day=WeatherDay(cfg.sowing_date+timedelta(days=d),t,tx,80,1)
        g=max(0,min(t,20)) if t<=30 else 20-2*(t-30)
        decl=.4093*math.sin(2*math.pi*(day.date.timetuple().tm_yday-81)/365)
        ell=24/math.pi*math.acos(np.clip(-math.tan(math.radians(cfg.latitude))*math.tan(decl),-1,1))
        pp=np.clip(1-.09*(16-ell),0,1)
        u=g*(pp if cfg.phenology.photoperiod_onset_gdd<=G<cfg.phenology.photoperiod_stop_gdd else 1)
        active=U>=cfg.phenology.vernalization_onset_tpp;sensitive=U<cfg.phenology.vernalization_stop_tpp
        v=0 if t<=-4 or t>=16 else (t+4)/4 if t<0 else 1 if t<=10 else (16-t)/6
        candidate=V+v*sensitive
        if candidate<10 and tx>30:candidate=max(0,V-.5*(tx-30))
        if active:V=candidate
        factor=.3+.7*min(V/40,1) if active and sensitive else 1
        G+=g;U+=u;A+=u*factor
        observed=pheno.integrate(pheno.calc_rates(day))
        maxerr=max(maxerr,np.max(np.abs(np.array([G,U,A,V])-
            np.array([observed.cumulative_gdd,observed.cumulative_tpp,observed.cumulative_tpv,observed.cumulative_vernalization]))))
    errors['tpv_clocks_max_abs']=float(maxerr);counts['tpv_days']=600
    # Equations S8–S11 against current daily capacities and state proposals.
    simulation=WheatSTBSimulation(cfg,WeatherProvider.from_csv(ROOT/'examples/wheat_stb/weather.csv'))
    simulation.run();result=simulation.finalize();days=result.daily;priorA=0.;priorArea=np.zeros(8);priorIncluded=False
    leaf_component=LeafCanopyModel(cfg.leaf)
    B=np.maximum(dict(cfg.leaf.stage_thresholds)[10],dict(cfg.leaf.stage_thresholds)[37]-np.arange(7)*cfg.leaf.rank_spacing_units)
    E=B+dict(cfg.leaf.stage_thresholds)[39]-dict(cfg.leaf.stage_thresholds)[37]
    maxerr=0.;ref=lost=0.;entered=False;ended=False
    thresholds=dict(cfg.leaf.stage_thresholds)
    for day,forcing in zip(days,simulation.weather):
        active=(priorA>=B)&day.crop_included&priorIncluded
        area=np.clip((day.cumulative_tpv-B)/(E-B),0,1)*active
        J=np.clip((thresholds[39]-day.cumulative_tpv)/(thresholds[39]-thresholds[31]),0,1)
        juvenile=day.crop_included and priorIncluded and priorA>=thresholds[10] and J>0
        area=np.append(area,J*juvenile)
        maxerr=max(maxerr,np.max(abs(area-np.asarray(day.host_area))))
        renewal=np.divide(np.maximum(area-priorArea,0),area,out=np.zeros(8),where=area>0)
        renewal[7]=(-np.expm1(-max(forcing.tmean_c,0)/300))*juvenile
        leaf_rates=leaf_component.calc_rates(day.cumulative_tpv,forcing.tmean_c,day.crop_included)
        maxerr=max(maxerr,np.max(abs(renewal-leaf_rates.renewal)))
        leaf_component.integrate(leaf_rates)
        if day.crop_included and day.cumulative_tpv>=thresholds[65]:entered=True
        if entered and not ended and day.crop_included:
            lais=cfg.yield_model.standardized_upper3_lai*area[:3]/3
            ref+=lais.sum();lost+=lais@np.array(day.symptomatic_fraction[:3])
            if day.cumulative_tpv>=thresholds[85]:ended=True
        priorA=day.cumulative_tpv;priorArea=area;priorIncluded=day.crop_included
    errors['leaf_capacity_max_abs']=float(maxerr);counts['daily_example_days']=len(days)
    assert ended
    errors['reference_had_abs']=abs(ref-result.summary['reference_had3'])
    errors['lost_had_abs']=abs(lost-result.summary['lost_had3'])
    errors['conditional_yield_max_abs']=float(np.max(abs(np.array(result.summary['conditional_yield_loss_t_ha'])-
        lost*np.array(cfg.yield_model.coefficients_t_ha_per_glai_day))))
    # Equations S19–S21, including the finite bisection and exact humidity endpoints.
    weather_error=0.;weather_cases=0
    for t in [-5,0,18,31,39]:
        for rh in [0,50,85,90,100]:
            for rain in [0,2,20]:
                tx=t+7
                hourly=np.array([t+max(tx-t,0)*math.cos(2*math.pi*(h+.5)/24) for h in range(24)])
                saturation=np.exp(17.625*hourly/(243.04+hourly));low=0.;high=float(saturation.max())
                for _ in range(30):
                    vapor=(low+high)/2
                    if np.minimum(vapor/saturation,1).mean()<rh/100:low=vapor
                    else:high=vapor
                reconstructed=100*np.minimum(((low+high)/2)/saturation,1)
                if rh in (0,100):reconstructed[:]=rh
                humid=reconstructed>=90-1e-7
                rainfraction=-np.expm1(-rain/(24*cfg.rain_rate_mm_hour))
                exposure=np.mean(np.exp(-.5*((hourly-18)/8)**2)*(humid+(~humid)*rainfraction))
                observed=float(duration_exposure(t,tx,rh,rain,cfg.rain_rate_mm_hour)['exposure'])
                weather_error=max(weather_error,abs(exposure-observed));weather_cases+=1
    errors['weather_exposure_max_abs']=weather_error;counts['weather_cases']=weather_cases
    event_error=0.
    for i,label in enumerate([f'F{x}' for x in range(1,8)]+['juvenile']):
        for key,variable in [('first_infection_date_by_leaf','affected_fraction'),('first_symptom_date_by_leaf','symptomatic_fraction')]:
            first=next((x.date.isoformat() for x in days if x.crop_included and getattr(x,variable)[i]>=cfg.detection_fraction),None)
            assert first==result.summary[key][label]
    counts['leaf_event_dates']=16
    # Equations S13–S18, including active-slot reset, dilution, simultaneous updates and residue identity.
    rng=np.random.default_rng(20261007);maxerr=mass=source_total=0.;cases=0
    for m in [1,3,5]:
        p=replace(cfg.disease,latent_stages=m)
        for n in [2,4,7]:
            for t in [-5,0,18,35]:
                active=rng.random(8)>.25;active[0]=True
                area=rng.uniform(.1,1,8)*active;renewal=rng.uniform(0,.2,8)*active
                state=rng.dirichlet(np.ones(m+4),size=8);residue=rng.uniform(.1,1,2)
                expected=advance_equations(state,residue,t,.6,2,active,renewal,area,.1,p,n)
                actual=_advance_day(state,residue,t,.6,2,active,renewal,area,.1,p.vector(),m,n)
                for a,b in zip(expected,actual):maxerr=max(maxerr,float(np.max(abs(a-b))))
                mass=max(mass,float(np.max(abs(actual[0].sum(axis=1)-1))))
                source_total=max(source_total,abs(actual[1].sum()-residue.sum()*math.exp(-max(t,0)/18/p.residue_decay_reference_days)))
                cases+=1
    errors['disease_source_pathway_max_abs']=maxerr;errors['tissue_conservation_max_abs']=mass
    errors['analytic_source_decay_max_abs']=source_total;counts['disease_cases']=cases
    # Reconstruct all archived current-model period, paired and ensemble estimates from moments.
    draws=pd.read_csv(ROOT/'data/paper_study/regional_parameter_uncertainty/spatial_draws.csv').sort_values('spatial_draw_id')
    weights=draws.area_mean_weight.to_numpy();strata=draws.stratum.to_numpy()
    worst=0.;n_estimates=0
    for kind, moment_name, summary_name, ensemble_name, mean_col, period_key, coverage_col, ensmean_col, ensmcse_col in [
        ('period','draw_period_moments','period_means_by_gcm_ssp','ensemble_period_means','mean','period','metric_area_time_coverage','gcm_mean','spatial_mcse_gcm_mean'),
        ('paired','paired_draw_period_moments','paired_period_changes_by_gcm_ssp','ensemble_paired_period_changes','change','future_period','common_paired_area_time_coverage','gcm_mean_change','spatial_mcse_gcm_mean_change')]:
        moments=pd.read_parquet(CLIMATE/f'{moment_name}.parquet')
        summaries=pd.read_parquet(CLIMATE/f'{summary_name}.parquet')
        ensembles=pd.read_parquet(CLIMATE/f'{ensemble_name}.parquet')
        influence={};means={}
        groupkeys=['model','scenario',period_key,'metric']
        summary_index=summaries.set_index(groupkeys)
        for key,part in moments.groupby(groupkeys):
            part=part.sort_values('spatial_draw_id');x=part.numerator.to_numpy();z=part.denominator.to_numpy()
            coverage=weights@z;mu=(weights@x)/coverage;psi=(x-mu*z)/coverage
            mcse=math.sqrt(sampling_variance(psi,weights,strata));row=summary_index.loc[key]
            worst=max(worst,abs(mu-row[mean_col]),abs(mcse-row.spatial_mcse),abs(coverage-row[coverage_col]))
            influence[key]=psi;means[key]=mu;n_estimates+=1
            if kind=='paired':
                delta=(weights@(part.future_numerator.to_numpy()-part.reference_numerator.to_numpy()))/coverage
                worst=max(worst,abs(delta-mu))
        for row in ensembles.itertuples():
            keys=[k for k in means if k[1]==row.scenario and k[2]==getattr(row,period_key) and k[3]==row.metric]
            psi=np.mean([influence[k] for k in keys],axis=0);mu=np.mean([means[k] for k in keys])
            mcse=math.sqrt(sampling_variance(psi,weights,strata))
            worst=max(worst,abs(mu-getattr(row,ensmean_col)),abs(mcse-getattr(row,ensmcse_col)));n_estimates+=1
    errors['spatial_ratio_paired_mcse_max_abs']=float(worst);counts['spatial_estimates']=n_estimates
    with (DEST/'Parameter_inventory.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==manifest['inventory_rows']
    assert set(r['role'] for r in rows)=={'fixed','fitted','scenario'}
    for row in rows:
        json.loads(row['value'])
        assert row['source_sha256']==original[row['provenance']]
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in original.items())
    assert all(v<1e-9 for v in errors.values()),errors
    output=dict(status='equations_match_unchanged_implementation', errors=errors,counts=counts,
        source_hashes_unchanged=True,parameter_inventory_rows=len(rows),
        scientific_scope='Numerical equation and output-arithmetic correspondence; not predictive validation.',
        unverified_scope='No new climate-forcing replay or biological/yield accuracy claim.',
        specification_sha256=hashlib.sha256((DEST/'Supplementary_Model_Specification.md').read_bytes()).hexdigest(),
        parameter_inventory_sha256=hashlib.sha256((DEST/'Parameter_inventory.csv').read_bytes()).hexdigest())
    (DEST/'equation_verification.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
