"""Within-model weather and host-development counterfactuals."""

import numpy as np

from .terminal_ensemble import simulate_terminal_ensemble


def aligned_crop_inputs(weather, active, renewal, sow_indices, endpoints, required_days):
    """Index each cell by days since scenario sowing, retaining endpoint causality.

    Weather must cover the longest host duration used by either counterfactual.
    Padded values beyond that duration never enter a simulated state update.
    """
    sow_indices,endpoints,required_days = [np.asarray(x,int) for x in
                                          [sow_indices,endpoints,required_days]]
    n = len(sow_indices); days = int(required_days.max())
    if n == 0 or np.any(required_days < 1):raise ValueError('No valid crop windows.')
    if any(value.shape[0]!=n for value in weather.values()):raise ValueError('Weather membership mismatch.')
    if np.any(sow_indices < 0) or np.any(sow_indices+required_days > active.shape[1]):
        raise ValueError('Counterfactual extends beyond actual weather coverage.')
    aligned = {key:np.zeros((n,days)) for key in weather}
    host = np.zeros((n,days,active.shape[2]),bool); growth = np.zeros_like(host,dtype=float)
    duration = endpoints-sow_indices+1
    if np.any(duration<1) or np.any(duration>required_days):raise ValueError('Invalid host endpoint.')
    for cell in range(n):
        start,need,length = sow_indices[cell],required_days[cell],duration[cell]
        for key,value in weather.items():aligned[key][cell,:need] = value[cell,start:start+need]
        host[cell,:length] = active[cell,start:start+length]
        growth[cell,:length] = renewal[cell,start:start+length]
    return aligned,host,growth,duration-1


def decompose_weather_host(base,future,parameters):
    """Symmetric two-factor decomposition; these are model counterfactuals."""
    outcomes = {}
    mass_error = 0.
    for weather_case,host_case in [(0,0),(1,0),(0,1),(1,1)]:
        weather = [base,future][weather_case][0]
        _,active,renewal,endpoint = [base,future][host_case]
        result = simulate_terminal_ensemble(weather['tmean_c'],weather['rh_mean_pct'],
            weather['precipitation_mm'],active,renewal,endpoint,parameters)
        outcomes[f'y{weather_case}{host_case}'] = result['damage'][:,:,:3].mean(axis=2)*100
        mass_error = max(mass_error,result['mass_error'])
    weather_effect = .5*((outcomes['y10']-outcomes['y00'])+(outcomes['y11']-outcomes['y01']))
    host_effect = .5*((outcomes['y01']-outcomes['y00'])+(outcomes['y11']-outcomes['y10']))
    total = outcomes['y11']-outcomes['y00']
    np.testing.assert_allclose(weather_effect+host_effect,total,rtol=0,atol=1e-12)
    return dict(**outcomes,weather_effect=weather_effect,host_effect=host_effect,
        total_effect=total,mass_error=mass_error)
