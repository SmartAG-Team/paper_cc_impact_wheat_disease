"""Arithmetic fixtures guarding hierarchy, paired resampling and censoring."""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd


def module(filename='statistics.py'):
    path = Path(__file__).with_name(filename)
    assert path.exists(), 'statistical calculation module is not implemented'
    spec = importlib.util.spec_from_file_location('revision_statistics', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_duplicate_dates_do_not_overweight_a_leaf_or_field():
    # A has two fields, one with two leaves; B has one field/leaf.
    f = pd.DataFrame(dict(coordinate_year=['A','A','A','A','B'],
        field_id=['a','a','a','b','c'], leaf_index=[0,0,1,0,0]))
    np.testing.assert_allclose(module().hierarchy_weights(f), [.0625,.0625,.125,.25,.5])


def test_cluster_draw_reuses_paired_complete_group_and_preserves_weight():
    # A mean loss difference=(2+6)/2=4, B difference=-2. Full mean=1.
    f = pd.DataFrame(dict(coordinate_year=['A','A','B'],field_id=['a','a','b'],
        leaf_index=[0,1,0], model=[2.,6.,0.], benchmark=[0.,0.,2.]))
    draws = np.array([[0,0],[0,1],[1,1]])
    point, boot = module().paired_bootstrap(f, draws, kind='mean')
    assert point['difference'] == 1.
    np.testing.assert_allclose(boot['difference'], [4.,1.,-2.])
    np.testing.assert_allclose(boot['model'], [4.,2.,0.])
    np.testing.assert_allclose(boot['benchmark'], [0.,1.,2.])


def test_rmse_difference_is_difference_of_roots_after_group_aggregation():
    f = pd.DataFrame(dict(coordinate_year=['A','B'],field_id=['a','b'],leaf_index=[0,0],
        observed=[0.,0.], model=[3.,4.], benchmark=[0.,0.]))
    point, boot = module().paired_bootstrap(f, np.array([[0,1],[0,0],[1,1]]),kind='severity')
    assert point['rmse_difference'] == np.sqrt(12.5)
    np.testing.assert_allclose(boot['rmse_difference'], [np.sqrt(12.5),3.,4.])
    assert point['mae_difference'] == 3.5


def test_class_absent_draw_is_missing_and_not_fabricated_as_zero():
    f = pd.DataFrame(dict(coordinate_year=['A','B'],field_id=['a','b'],leaf_index=[0,0],
        observed=[True,False],model=[True,False],benchmark=[False,True]))
    point, boot = module().paired_bootstrap(f,np.array([[0,0],[0,1],[1,1]]),kind='detection')
    assert point['balanced_accuracy_difference'] == 1.
    np.testing.assert_allclose(boot['sensitivity_difference'],[1.,1.,np.nan],equal_nan=True)
    np.testing.assert_allclose(boot['specificity_difference'],[np.nan,1.,1.],equal_nan=True)
    np.testing.assert_allclose(boot['balanced_accuracy_difference'],[np.nan,1.,np.nan],equal_nan=True)


def test_stage_fit_membership_excludes_heldout_fields_and_disease_columns():
    f = pd.DataFrame(dict(field_id=['training','heldout'],event=[37,37],value=[999.,0.]))
    selected = module().training_stage_membership(f, {'training'})
    assert selected.field_id.tolist() == ['training']
    assert 'value' not in selected.columns


def test_preflag_zero_exclusion_requires_ordered_two_bound_observed_stage():
    f = pd.DataFrame(dict(leaf_index=[0,0,0,1,0],observed_percent=[0.,0.,0.,0.,1.],
        stage_from=[31.,31.,39.,31.,31.],stage_to=[32.,np.nan,31.,32.,32.]))
    assert module().preflag_zero_mask(f,37).tolist() == [True,False,False,False,False]


def test_exact_two_group_resampling_retains_multinomial_multiplicity():
    impl = module('exact_bootstrap.py')
    assert hasattr(impl,'exact_mean_bootstrap'), 'exact resampling distribution is not implemented'
    # Ordered samples [A,A], [A,B], [B,A], [B,B] have means 0,1,1,2.
    values, probability = impl.exact_mean_bootstrap([0.,2.])
    np.testing.assert_allclose(values,[0.,1.,2.])
    np.testing.assert_allclose(probability,[.25,.5,.25])
