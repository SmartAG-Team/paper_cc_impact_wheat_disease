"""Guard consolidated imports and the executable paper reproduction contract."""
import importlib
from pathlib import Path
import pytest
from paper.workflow import ROOT,parse_args,selected_stages
from paper.analyses.food_security.provenance import prepare_destination


def test_old_and_canonical_model_names_share_dataclass_identity():
    old=importlib.import_module('model.seasonal_septoria.overwinter')
    new=importlib.import_module('models.seasonal_septoria.overwinter')
    assert old is new
    assert isinstance(old.OverwinterParameters(),new.OverwinterParameters)


def test_figure_reproduction_requires_models_and_all_scientific_analyses():
    stages=selected_stages(['figures'])
    assert set(['monitoring','process','climate','attribution','exposure','climate-response','adaptation','yield']).issubset(stages)
    assert stages[-1]=='figures'
    assert selected_stages(None)[-1]=='validate'


def test_dry_run_accepts_new_reproduction_path_without_creating_it():
    path=ROOT/'data/reproductions/not_created_by_dry_run'
    assert not path.exists()
    args=parse_args(['--dry-run','--output',str(path),'--stage','figures'])
    assert args.output==path and args.dry_run
    assert not path.exists()
    with pytest.raises(SystemExit):parse_args(['--output',str(ROOT/'data/final_climate')])


def test_new_analyses_can_write_only_to_fresh_reproduction_data(tmp_path):
    result=prepare_destination(tmp_path/'data/reproductions/run/attribution',tmp_path)
    assert result.is_dir()
    with pytest.raises(FileExistsError):prepare_destination(result,tmp_path)
    with pytest.raises(ValueError):prepare_destination(tmp_path/'data/final_climate/new',tmp_path)


def test_partial_runs_never_claim_unexecuted_climate_or_process_rebuilds():
    from paper.workflow import completion_flags
    assert completion_flags({'inputs':{'status':'passed'}})==dict(continental_census_recomputed=False,sampled_process_experiments_rerun=False)
    assert completion_flags({'climate':{'full_census_recomputed':True},'attribution':{'scientific_simulations_rerun':True}})==dict(continental_census_recomputed=True,sampled_process_experiments_rerun=False)
    assert completion_flags({'attribution':{'scientific_simulations_rerun':True},'adaptation':{'scientific_simulations_rerun':True}})['sampled_process_experiments_rerun']
