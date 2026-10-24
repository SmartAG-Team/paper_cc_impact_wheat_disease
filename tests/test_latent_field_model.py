import numpy as np
import pandas as pd


def test_shared_field_phenotype_is_inferred_only_from_earlier_assessment_days():
    from models.monitoring.latent_field_model import fit_latent_field,predict_latent_field
    training=pd.DataFrame(dict(field_id=np.repeat(np.arange(18),3),source='BASF',
        date=np.tile(['2014-05-01','2014-05-02','2014-05-03'],18),leaf_index=0,
        observed_percent=np.repeat(np.repeat([10.,50.,90.],6),3),global_target_index=np.arange(54)))
    fitted=fit_latent_field(np.zeros((54,1)),training,np.full(54,50.),np.ones(54),
        config=dict(iterations=8,max_iter=30))
    query=pd.DataFrame(dict(field_id=['new_high']*4+['new_low']*2,source='BASF',
        date=['2015-05-01','2015-05-02','2015-05-02','2015-05-03','2015-05-01','2015-05-02'],
        leaf_index=[0,0,1,0,0,0],observed_percent=[90.,80.,85.,10.,10.,50.]))
    baseline=np.full(6,50.);x=np.zeros((6,1))
    value,prob=predict_latent_field(fitted,x,query,baseline)
    np.testing.assert_array_equal(value[[0,4]],[50.,50.])
    assert value[1]>80 and value[2]>80 and value[5]<20
    np.testing.assert_allclose(prob.sum(axis=1),1.)
    altered=query.copy();altered.loc[1:3,'observed_percent']=[0.,0.,100.]
    changed,_=predict_latent_field(fitted,x,altered,baseline)
    np.testing.assert_array_equal(value[:3],changed[:3])
    assert value[3]!=changed[3]
