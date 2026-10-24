import numpy as np


def _forcing(days=120):
    weather=np.zeros((1,days,6),np.float32)
    weather[:,:,0]=(18-12)/12;weather[:,:,1]=(95-80)/20
    weather[:,:,2]=np.log1p(5)/2;weather[:,:,5]=1
    return weather


def test_simulated_epidemic_conserves_mass_and_allows_full_range_growth():
    from models.monitoring.epidemic_world import simulate_epidemics
    weather=_forcing();valid=np.ones(weather.shape[:2],bool)
    params=dict(arrival_day=np.array([10]),primary_rate=np.array([.1]),secondary_rate=np.array([1.]),
        latent_days_18c=np.array([8.]),susceptibility=np.ones((1,7)),exposure_delay=np.zeros((1,7),int))
    result=simulate_epidemics(weather,valid,**params)
    assert result['maximum_mass_error']<1e-12 and result['minimum_compartment']>=-1e-12
    np.testing.assert_array_equal(result['visible'][:,:11],0)
    assert np.min(result['visible'][:,-1])>.95
    assert np.isfinite(result['visible']).all()
    cold=weather.copy();cold[:,:,0]=(-5-12)/12
    np.testing.assert_array_equal(simulate_epidemics(cold,valid,**params)['visible'],0)


def test_simulated_trajectory_uses_only_completed_past_weather():
    from models.monitoring.epidemic_world import simulate_epidemics
    weather=_forcing();valid=np.ones(weather.shape[:2],bool)
    params=dict(arrival_day=np.array([0]),primary_rate=np.array([.03]),secondary_rate=np.array([.2]),
        latent_days_18c=np.array([14.]),susceptibility=np.ones((1,7)),exposure_delay=np.zeros((1,7),int))
    first=simulate_epidemics(weather,valid,**params)['visible']
    other=weather.copy();other[:,45:,0]=-2;other[:,45:,2]=0
    second=simulate_epidemics(other,valid,**params)['visible']
    np.testing.assert_array_equal(first[:,:46],second[:,:46])
    assert np.max(np.abs(first[:,46:]-second[:,46:]))>.1


def test_world_generation_is_blind_to_all_real_disease_values_and_preserves_mass():
    from models.monitoring.epidemic_world import world_inputs,make_worlds
    weather=np.repeat(_forcing(),2,axis=0)
    mask=np.zeros((2,120,7),bool);mask[0,40,0]=True;mask[1,30,2]=True
    batch=dict(weather=weather,valid_weather=np.ones((2,120),bool),observed_mask=mask,
        observed=np.zeros((2,120,7),np.float32),static=np.zeros((2,3),np.float32),sources=np.array([0,1]),
        starts=['2015-01-01']*2,query_field=np.array([0,0,1,1]),query_day=np.array([40,80,30,90]),
        query_leaf=np.array([0,0,2,2]))
    inputs=world_inputs(batch,[2015,2015],np.array([.1,.4,.25,.25]))
    first,target,mass,_,audit=make_worlds(inputs,count=8,seed=4)
    batch['observed'][:]=np.nan
    second,other,other_mass,_,_=make_worlds(world_inputs(batch,[2015,2015],np.array([.1,.4,.25,.25])),count=8,seed=4)
    for key in ['weather','observed','observed_mask','static','query_field','query_day','query_leaf']:
        np.testing.assert_array_equal(first[key],second[key])
    np.testing.assert_array_equal(target,other);np.testing.assert_array_equal(mass,other_mass)
    np.testing.assert_allclose(np.bincount(first['query_field'],weights=mass),1,atol=1e-15)
    assert target.min()>=0 and target.max()<=100 and not audit['observed_disease_values_used']
    assert 'observed' not in inputs
