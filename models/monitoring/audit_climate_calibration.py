"""Authenticate climate-model calibration scope and guard scientific deployment."""
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from .innovation_common import ROOT, PARENT, load_cohort
from .inputs import sha

REQUIRED_GATES = ['autonomous_input_contract', 'no_satellite_ancestry', 'fold_fit_excludes_query_labels',
    'comparable_endpoint_mapping', 'magnitude_validated', 'original_goal_passed', 'autonomous_goal_passed',
    'scenario_sowing_inputs_ready', 'climate_perturbations_pass']


def deployment_decision(evidence):
    gates = {name: evidence.get(name) is True for name in REQUIRED_GATES}
    return dict(qualified=all(gates.values()), gates=gates,
        failed=[name for name, passed in gates.items() if not passed])


def audit():
    out = ROOT/'outputs/disease_improvement/climate_calibration_audit_20261022'
    if out.exists():
        raise FileExistsError('Preserve previous climate audit')
    directory = ROOT/'data/final_climate'
    manifest = json.loads((directory/'scientific_manifest.json').read_text())
    authenticated = []
    records = [*manifest['core_sources'], *manifest['implementation_sources'],
        *[item for item in manifest['inputs'] if any(name in item['path'] for name in [
            'phenology_parameters.json', 'disease_parameters.json', 'stage_thresholds.json'])]]
    for record in records:
        assert sha(ROOT/record['path']) == record['sha256'], record['path']
        authenticated.append(record['path'])
    fit = json.loads((directory/'disease_parameters.json').read_text())
    stages = json.loads((directory/'stage_thresholds.json').read_text())
    runtime = json.loads((directory/'original_runtime_configuration.json').read_text())
    membership_path = ROOT/'data/final_study/evidence/seasonal_calibration_v1/field_season_membership.csv'
    membership = pd.read_csv(membership_path)
    old_digest = stages['source_sha256']['analysis/paper_study/seasonal_calibration_v1/field_season_membership.csv']
    assert sha(membership_path) == old_digest
    calibration = membership.loc[membership.partition.eq('calibration')]
    assert len(calibration) == fit['calibration_fields']
    obs, _, _, _, splits, _ = load_cohort()
    overlaps = []
    for split in splits:
        query = obs.set_index('global_target_index').loc[split['test']]
        fields = set(calibration.field_id)&set(query.field_id)
        overlaps.append(dict(fold=split['fold'], query_year=int(query.season_year.iloc[0]),
            original_calibration_fields_in_query=len(fields), field_ids=sorted(fields),
            original_calibration_uses_later_season=bool(calibration.season_year.max() > query.season_year.min())))
    archival_fit = ROOT/'data/final_study/evidence/overwinter_leaf_model/disease/overwinter_source_model/frozen_selected_fit.json'
    assert sha(archival_fit) == sha(directory/'disease_parameters.json')
    source_module = ROOT/'analysis/methods/seasonal_model/disease/run.py'
    # The retained module supplies prediction arithmetic, not the missing
    # parameter-selection trace. Do not turn metadata into independently
    # verified selection/optimization provenance.
    original_selection_paths = [archival_fit.with_name('selection.json'), archival_fit.with_name('frozen_selection.json')]
    available_selection = [str(p.relative_to(ROOT)) for p in original_selection_paths if p.exists()]
    flag = obs.loc[obs.source.eq('BASF')&obs.leaf_index.eq(0), ['global_target_index', 'field_id', 'date']]
    evidence = dict(autonomous_input_contract=True, no_satellite_ancestry=True,
        fold_fit_excludes_query_labels=False, comparable_endpoint_mapping=False,
        magnitude_validated=False, original_goal_passed=False, autonomous_goal_passed=False,
        scenario_sowing_inputs_ready=False, climate_perturbations_pass=False)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        authenticated_scientific_sources=authenticated,
        recorded_fit_scope=dict(disease_magnitude_fit=fit['disease_magnitude_fit'],
            validation_outcomes_used=fit['validation_outcomes_used'], calibration_fields=len(calibration),
            calibration_year_counts=calibration.groupby('season_year').size().to_dict(),
            original_validation_years=sorted(membership.loc[membership.partition.eq('validation'), 'season_year'].unique().tolist())),
        original_selection=dict(recorded_sha256=fit['selection_sha256'], available_selection_paths=available_selection,
            membership_authenticated=True, original_optimization_trace_independently_verified=False),
        current_fold_comparison=overlaps,
        interpretation='The original fit has its own 2017–2018 calibration/2019 validation; it is not independent validation under the newer 2015–2019 outer splits. This does not prove leakage in its original 2019 design.',
        process=dict(initial_local_source=1., imported_pressure=fit['fitted']['constant_imported_pressure'],
            magnitude='Normalized affected-tissue proxy; calibrated healthy-area loss and empirical yield linkage remain unestablished',
            natural_senescence='Not measured or estimated by retained leaf_host policy',
            prefix_issue='Full-season climate wrapper marks trajectories invalid until the terminal stage is reached; daily observation validation needs a causal prefix adapter, without requiring future completion',
            current_sowing='Frozen climate runner uses declared calendar scenario; promoted RF sowing additionally requires the preceding annual-temperature year'),
        stage_identification={str(row['event']): dict(poorly_identified=row['poorly_identified'],
            genuine_two_sided_fields=row['genuine_two_sided_fields']) for row in stages['diagnostics']},
        donor_label_caution='EJA research_validation/prepare.py declares internal85→official87. The full ancestry of the frozen climate terminal label must be reconciled before interpreting its85 label as observed soft dough.',
        measurement=dict(retained_observations=len(obs), nominal_source_endpoints=int(obs.groupby(['source', 'leaf_index']).ngroups),
            BASF_code0_flag_targets=len(flag), BASF_code0_fields=int(flag.field_id.nunique()),
            full_EPPO_sampling_protocol_verified=False, other_native_anatomical_mappings_verified=False,
            policy='Keep all original rows; fixed F1 mapping only for independently documented BASF code0. Unknown native codes need an explicit statistical observation operator, not invented rank equivalence.'),
        frozen_claims=dict(retrospective_disease_validation=runtime['retrospective_disease_validation'],
            absolute_field_yield_forecast=runtime['absolute_field_yield_forecast'], conditional_yield_output=runtime['conditional_yield_output']),
        deployment_scope='Qualification of the proposed new RF-sown severity model, not a change to the original frozen calendar-scenario replay',
        deployment=deployment_decision(evidence), frozen_artifacts_modified=False,
        model_parameters_fitted=False)
    sources = [directory/name for name in ['scientific_manifest.json', 'disease_parameters.json',
        'phenology_parameters.json', 'stage_thresholds.json', 'original_runtime_configuration.json']]
    sources += [membership_path, archival_fit, source_module, ROOT/'model/seasonal_septoria/leaf_phenology.py',
        ROOT.parent/'EJA-PGN-Wheat/code/research_validation/prepare.py', PARENT/'training_observation_ledger.csv']
    result['source_sha256'] = {str(path): sha(path) for path in sources}
    result['auditor_sha256'] = sha(Path(__file__))
    out.mkdir(parents=True)
    flag.to_csv(out/'qualified_flag_endpoint_ids.csv', index=False)
    calibration.to_csv(out/'original_calibration_fields.csv', index=False)
    result['artifacts_sha256'] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    (out/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ['recorded_fit_scope', 'measurement', 'deployment']}, indent=2))
    return out


if __name__ == '__main__':
    audit()
