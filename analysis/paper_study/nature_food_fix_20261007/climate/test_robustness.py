import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

P=Path(__file__).with_name('run.py')
spec=importlib.util.spec_from_file_location('climate_robustness_local',P)
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)


def test_senescence_weight_uses_stage_elapsed_time():
    stage={65:np.array([2]),85:np.array([6])}
    weight=run.reference_decline(stage,8,0.)
    np.testing.assert_allclose(weight[0],[1,1,1,.75,.5,.25,0,0])
    np.testing.assert_allclose(run.reference_decline(stage,8,.5)[0],[1,1,1,.875,.75,.625,.5,.5])


def test_shared_draw_mcse_keeps_covariance():
    draws=pd.DataFrame({'stratum':np.repeat(np.arange(16),4),'area_mean_weight':np.full(64,1/64)})
    x=np.tile([-1.,0.,0.,1.],16)
    v=run.shared_mcse([x,x],draws)
    expected=np.sqrt(16*(1/16)**2*np.var(x[:4],ddof=1)/4)
    assert abs(v-expected)<1e-14
    assert run.shared_mcse([x,-x],draws)<1e-14


def test_joint_profiles_are_jointly_admissible():
    profiles=run.stage_profile_variants()
    assert set(profiles)=={'stage_profile_early37','stage_profile_late37'}
    for record in profiles.values():
        assert np.all(np.diff(list(record['thresholds'].values()))>0)
        assert record['joint_loss_days']<=record['minimum_loss_days']+1+1e-12
        assert record['validation_used'] is False
