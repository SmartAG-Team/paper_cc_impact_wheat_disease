"""Rolling terminal states for numerically equivalent parameter ensembles."""

import numpy as np
from numba import njit,prange


@njit(cache=True,parallel=True)
def _terminal(t,h,r,active,renewal,endpoints,coefficients,stages,kernels,substeps):
    episodes,days,leaves = active.shape
    draws = len(stages)
    damage = np.zeros((draws,episodes,leaves))
    pycnidia = np.zeros_like(damage)
    mass_errors = np.zeros((draws,episodes))
    minima = np.zeros((draws,episodes))
    dt = 1./substeps
    for flat in prange(draws*episodes):
        draw,episode = flat//episodes,flat%episodes
        (alpha,beta,latent_days,nonsporulating_days,infectious_days,
         humidity_midpoint,humidity_scale,rain_scale,optimum,width) = coefficients[draw]
        count = stages[draw]
        block = count+3
        current = np.zeros((leaves,1+2*block))
        current[:,0] = 1.
        q_removed = -np.expm1(-dt/infectious_days)
        mass_error,min_state = 0.,0.
        for day in range(endpoints[episode]+1):
            for leaf in range(leaves):
                if not active[episode,day,leaf]:
                    current[leaf,:] = 0.;current[leaf,0] = 1.
                else:
                    retained = 0.
                    for compartment in range(1,current.shape[1]):
                        current[leaf,compartment] *= 1-renewal[episode,day,leaf]
                        retained += current[leaf,compartment]
                    current[leaf,0] = 1-retained
            thermal = max(t[episode,day],0.)/18.
            thermal_establishment = np.exp(-.5*((t[episode,day]-optimum)/width)**2)
            humidity_response = 1./(1+np.exp(-max(-700.,min(700.,(h[episode,day]-humidity_midpoint)/humidity_scale))))
            splash = -np.expm1(-r[episode,day]/rain_scale)
            establish = thermal_establishment*(1-(1-humidity_response)*(1-splash))
            q_latent = -np.expm1(-dt*count*thermal/latent_days)
            q_sporulate = -np.expm1(-dt*thermal/nonsporulating_days)
            for _ in range(substeps):
                infectious = np.zeros(leaves)
                for leaf in range(leaves):
                    if active[episode,day,leaf]:
                        infectious[leaf] = current[leaf,1+count+1]+current[leaf,1+block+count+1]
                updated = current.copy()
                for leaf in range(leaves):
                    if not active[episode,day,leaf]:
                        continue
                    imported,denominator = 0.,0.
                    for other in range(leaves):
                        if active[episode,day,other]:
                            imported += kernels[draw,leaf,other]*infectious[other]
                            denominator += kernels[draw,leaf,other]
                    primary_hazard = alpha*establish
                    secondary_hazard = beta*establish*splash*imported/denominator
                    total_hazard = primary_hazard+secondary_hazard
                    newly_infected = current[leaf,0]*(-np.expm1(-total_hazard*dt))
                    p_flow,s_flow = 0.,0.
                    if total_hazard>0:
                        p_flow = newly_infected*primary_hazard/total_hazard
                        s_flow = newly_infected*secondary_hazard/total_hazard
                    updated[leaf,0] -= newly_infected
                    updated[leaf,1] += p_flow
                    updated[leaf,1+block] += s_flow
                    for origin in range(2):
                        start = 1+origin*block
                        for latent in range(count):
                            transfer = current[leaf,start+latent]*q_latent
                            updated[leaf,start+latent] -= transfer
                            updated[leaf,start+latent+1] += transfer
                        visible = start+count
                        to_infectious = current[leaf,visible]*q_sporulate
                        to_removed = current[leaf,visible+1]*q_removed
                        updated[leaf,visible] -= to_infectious
                        updated[leaf,visible+1] += to_infectious-to_removed
                        updated[leaf,visible+2] += to_removed
                current = updated
            for leaf in range(leaves):
                total = 0.
                for compartment in range(current.shape[1]):
                    total += current[leaf,compartment]
                    min_state = min(min_state,current[leaf,compartment])
                mass_error = max(mass_error,abs(total-1))
        for leaf in range(leaves):
            for origin in range(2):
                visible = 1+origin*block+count
                damage[draw,episode,leaf] += current[leaf,visible]+current[leaf,visible+1]+current[leaf,visible+2]
                pycnidia[draw,episode,leaf] += current[leaf,visible+1]+current[leaf,visible+2]
        mass_errors[draw,episode],minima[draw,episode] = mass_error,min_state
    return damage,pycnidia,np.max(mass_errors),np.min(minima)


def simulate_terminal_ensemble(temperature,humidity,rain,host_active,host_renewal,
                               endpoints,parameters,time_step=.25):
    """Match full trajectories at independently supplied endpoint day indices.

    No timestep or infection response is approximated. Weather and host inputs
    are shared across parameter draws, while their complete route-specific
    current state is retained until the endpoint. Post-endpoint forcing never
    contributes. This function estimates terminal damage and pycnidia only.
    """
    t,h,r = [np.asarray(x,float) for x in (temperature,humidity,rain)]
    active_raw = np.asarray(host_active)
    active,renewal = active_raw.astype(bool),np.asarray(host_renewal,float)
    end_raw = np.asarray(endpoints)
    if (t.ndim!=2 or t.size==0 or h.shape!=t.shape or r.shape!=t.shape
            or active.ndim!=3 or active.shape[:2]!=t.shape or active.shape[2]==0
            or renewal.shape!=active.shape or not all(np.isfinite(x).all() for x in [t,h,r,renewal])
            or np.any((h<0)|(h>100)) or np.any(r<0) or np.any((active_raw!=0)&(active_raw!=1))
            or np.any((renewal<0)|(renewal>1)) or np.any(renewal[~active]!=0)):
        raise ValueError('Invalid daily weather or normalized host fractions.')
    if (end_raw.shape!=(len(t),) or not np.isfinite(end_raw).all()
            or np.any(end_raw!=end_raw.astype(int)) or np.any((end_raw<0)|(end_raw>=t.shape[1]))):
        raise ValueError('Every endpoint must be an integer day inside the forcing window.')
    if not parameters or not np.isfinite(time_step) or not 0<time_step<=1:
        raise ValueError('Nonempty parameter draws and a valid integration step required.')
    coefficients,stages,kernels = [],[],[]
    ranks = np.arange(active.shape[2])
    for p in parameters:
        values = (p.alpha,p.beta,p.latent_days,p.nonsporulating_days,p.infectious_days,
                  p.humidity_midpoint,p.humidity_scale,p.rain_scale,p.temperature_optimum,p.temperature_width)
        positive = (p.latent_days,p.nonsporulating_days,p.infectious_days,p.humidity_scale,
                    p.rain_scale,p.temperature_width,p.rank_distance_scale)
        if (not np.isfinite(values).all() or p.alpha<0 or p.beta<0
                or not all(np.isfinite(x) and x>0 for x in positive)
                or not 0<=p.humidity_midpoint<=100 or not isinstance(p.latent_stages,(int,np.integer))
                or p.latent_stages<1):
            raise ValueError('Invalid parameter draw.')
        coefficients.append(values);stages.append(p.latent_stages)
        kernels.append(np.exp(-abs(ranks[:,None]-ranks[None,:])/p.rank_distance_scale))
    damage,pycnidia,mass_error,minimum = _terminal(t,h,r,active,renewal,end_raw.astype(int),
        np.asarray(coefficients),np.asarray(stages,int),np.asarray(kernels),int(np.ceil(1/time_step)))
    return dict(damage=damage,pycnidia=pycnidia,mass_error=float(mass_error),minimum_state=float(minimum))
