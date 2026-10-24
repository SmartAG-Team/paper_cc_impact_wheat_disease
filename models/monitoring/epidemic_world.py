"""Assumed epidemic response worlds for structural pretraining, without real labels."""
import numpy as np
import pandas as pd


def simulate_epidemics(weather,valid_weather,arrival_day,primary_rate,secondary_rate,
        latent_days_18c,susceptibility,exposure_delay):
    """Seven exchangeable channels; visible[q] excludes day-q forcing.

    latent_days_18c is a continuous-rate progression timescale L. The three
    discrete daily stages have mean transit 3/(1-exp(-3/L)) at constant18C,
    approximately9.6..36.5days for the assumedL=8..35 range, not exactlyL.
    """
    weather=np.asarray(weather,float);valid=np.asarray(valid_weather,bool)
    n,days,_=weather.shape
    primary=np.asarray(primary_rate,float)[:,None];secondary=np.asarray(secondary_rate,float)[:,None]
    latency=np.asarray(latent_days_18c,float)[:,None]
    susceptibility=np.asarray(susceptibility,float)
    onset=np.asarray(arrival_day,int)[:,None]+np.asarray(exposure_delay,int)
    if (weather.shape[-1]!=6 or valid.shape!=(n,days) or susceptibility.shape!=(n,7) or onset.shape!=(n,7)
            or not np.isfinite(weather).all() or np.any(latency<=0) or np.any(primary<0)
            or np.any(secondary<0) or np.any(susceptibility<0)):
        raise ValueError('Finite forcing and nonnegative epidemic response parameters required')
    state=np.zeros((n,7,5),float);state[:,:,0]=1
    visible=np.zeros((n,days+1,7),np.float32)
    mass_error=0.;minimum=0.
    for day in range(days):
        temperature=weather[:,day,0]*12+12
        humidity=np.clip(weather[:,day,1]*20+80,0,100)
        rain=np.maximum(np.expm1(weather[:,day,2]*2),0)
        suitability=np.exp(-.5*((temperature-18)/8)**2)
        wet=np.maximum(1/(1+np.exp(-(humidity-85)/5)),-np.expm1(-rain/1.5))
        splash=-np.expm1(-rain/2)
        # A shared field reservoir couples exchangeable nominal score channels.
        reservoir=state[:,:,4].mean(axis=1,keepdims=True)
        hazard=suitability[:,None]*wet[:,None]*susceptibility*(primary+secondary*reservoir*splash[:,None])
        active=valid[:,day,None]&(day>=onset)
        infected=state[:,:,0]*(-np.expm1(-hazard))*active
        progression=(-np.expm1(-3*np.maximum(temperature,0)[:,None]/(18*latency)))*valid[:,day,None]
        transfer=state[:,:,1:4]*progression[:,:,None]
        state=np.stack([state[:,:,0]-infected,state[:,:,1]+infected-transfer[:,:,0],
            state[:,:,2]+transfer[:,:,0]-transfer[:,:,1],state[:,:,3]+transfer[:,:,1]-transfer[:,:,2],
            state[:,:,4]+transfer[:,:,2]],axis=2)
        visible[:,day+1]=state[:,:,4]
        mass_error=max(mass_error,float(np.max(np.abs(state.sum(axis=2)-1))))
        minimum=min(minimum,float(state.min()))
    if not np.isfinite(state).all() or mass_error>1e-10 or minimum< -1e-12:
        raise RuntimeError('Nonfinite or nonconservative synthetic epidemic')
    return dict(visible=visible,last_state=state,maximum_mass_error=mass_error,minimum_compartment=minimum)


def world_inputs(batch,season_year,query_weights):
    """An explicit whitelist excludes all measured disease values."""
    result={name:np.asarray(batch[name]).copy() for name in ['weather','valid_weather','observed_mask',
        'static','sources','query_field','query_day','query_leaf']}
    result.update(starts=list(batch['starts']),season_year=np.asarray(season_year,int),
        query_weights=np.asarray(query_weights,float),source_field_ids=list(batch.get('field_ids',map(str,range(len(batch['weather']))))))
    assert len(result['season_year'])==len(result['weather'])
    return result


def make_worlds(inputs,count=2048,seed=20261022):
    if 'observed' in inputs or 'observed_percent' in inputs:
        raise ValueError('Simulation accepts only the explicit label-free input object')
    rng=np.random.default_rng(seed);n=len(inputs['weather'])
    qfield=inputs['query_field'];original_weight=inputs['query_weights']
    field_mass=np.bincount(qfield,weights=original_weight,minlength=n)
    if np.any(field_mass<=0):raise ValueError('Positive source-field scientific mass required')
    source=rng.choice(n,count,replace=True,p=field_mass/field_mass.sum())
    weather=inputs['weather'][source].copy();valid=inputs['valid_weather'][source].copy()
    starts=pd.to_datetime(np.asarray(inputs['starts'])[source])
    years=inputs['season_year'][source]
    # Season metadata is fixed before any query; no maximum future query date is used.
    jan1=pd.to_datetime([f'{year}-01-01' for year in years])
    day_of_year=rng.integers(15,186,size=count)
    arrival=np.asarray((jan1-starts).days)+day_of_year-1
    primary=10**rng.uniform(-4,-.9,size=count)
    secondary=10**rng.uniform(-2,.3,size=count)
    healthy=rng.random(count)<.15;primary[healthy]=0;secondary[healthy]=0
    latency=rng.uniform(8,35,size=count)
    susceptibility=np.exp(rng.normal(0,.6,size=(count,7)))
    susceptibility/=susceptibility.mean(axis=1,keepdims=True)
    # IID channel parameters impose no numerical-code rank or anatomical hierarchy.
    delay=rng.integers(-21,43,size=(count,7))
    simulated=simulate_epidemics(weather,valid,arrival,primary,secondary,latency,susceptibility,delay)
    visible=simulated['visible'][:,:weather.shape[1]]
    mask=inputs['observed_mask'][source].copy()
    noise_scale=rng.uniform(0,.07,size=(count,1,1))
    measured=np.clip(visible+rng.normal(size=visible.shape)*noise_scale*np.sqrt(visible*(1-visible)),0,1)
    observed=np.where(mask,measured,0).astype(np.float32)
    fields=[];days=[];leaves=[];weights=[]
    for virtual,actual in enumerate(source):
        take=np.flatnonzero(qfield==actual)
        fields.extend([virtual]*len(take));days.extend(inputs['query_day'][take]);leaves.extend(inputs['query_leaf'][take])
        weights.extend((original_weight[take]/field_mass[actual]).tolist())
    fields,days,leaves=np.asarray(fields,int),np.asarray(days,int),np.asarray(leaves,int)
    targets=100*simulated['visible'][fields,days,leaves].astype(float)
    result=dict(weather=weather,valid_weather=valid,observed=observed,observed_mask=mask,
        static=inputs['static'][source].copy(),sources=inputs['sources'][source].copy(),
        query_field=fields,query_day=days,query_leaf=leaves,query_ids=np.arange(len(fields)),
        field_ids=[f'synthetic_{i}' for i in range(count)],starts=[str(d.date()) for d in starts],
        history_ids=[[] for _ in range(count)])
    parameters=dict(source_field_index=source,arrival_day=arrival,primary_rate=primary,secondary_rate=secondary,
        latent_days_18c=latency,susceptibility=susceptibility,exposure_delay=delay,healthy=healthy)
    return result,targets,np.asarray(weights,float),parameters,{
        'maximum_mass_error':simulated['maximum_mass_error'],'minimum_compartment':simulated['minimum_compartment'],
        'virtual_fields':count,'query_targets':len(targets),'observed_disease_values_used':False,
        'assumptions_not_empirical_parameter_estimates':True}
