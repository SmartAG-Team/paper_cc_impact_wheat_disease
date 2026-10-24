"""Render the verified autonomous and optional-monitoring comparisons."""
import json
from pathlib import Path

import pandas as pd

from .innovation_common import ROOT
from .inputs import sha


def write():
    autonomous = ROOT/'outputs/disease_improvement/autonomous_weather_20261022'
    monitoring = ROOT/'outputs/disease_improvement/autonomous_monitoring_20261022'
    calibration = ROOT/'outputs/disease_improvement/severity_calibration_support_repair_20261022'
    process = ROOT/'outputs/disease_improvement/process_establishment_20261022'
    audit = ROOT/'outputs/disease_improvement/climate_calibration_audit_20261022'
    verification = ROOT/'outputs/disease_improvement/calibration_process_verification_20261022'
    repair = ROOT/'outputs/disease_improvement/scenario_chronology_repair_20261022'
    variety = ROOT/'outputs/disease_improvement/variety_metadata_audit_20261022'
    for out in [autonomous, monitoring]:
        assert json.loads((out/'verification.json').read_text())['passed']
        assert json.loads((out/'independent_verification.json').read_text())['passed']
    metrics = pd.read_csv(autonomous/'metrics.csv')
    groups = pd.read_csv(autonomous/'subgroup_metrics.csv')
    other_metrics = pd.read_csv(monitoring/'metrics.csv')
    other_groups = pd.read_csv(monitoring/'subgroup_metrics.csv')
    decisions = json.loads((monitoring/'all_goal_decisions.json').read_text())
    comparisons = json.loads((autonomous/'autonomous_comparisons.json').read_text())
    verified = json.loads((verification/'verification.json').read_text()); assert verified['passed']
    repair_receipt = json.loads((repair/'verification.json').read_text()); assert repair_receipt['passed']
    suite = repair_receipt['test_suite']
    new_tables = []
    for directory in [calibration, process]:
        new_metrics = pd.read_csv(directory/'metrics.csv')
        new_groups = pd.read_csv(directory/'subgroup_metrics.csv')
        names = list(json.loads((directory/'all_goal_decisions.json').read_text()))
        lines = []
        for name in names:
            a = new_metrics.loc[new_metrics.model.eq(name)].set_index('evaluation')
            b = new_groups.loc[new_groups.model.eq(name)&new_groups.population.eq('severe_above75')].set_index('evaluation')
            lines.append(f"| `{name}` | {a.loc['temporal','rmse']:.3f} | {a.loc['spatial_50km','rmse']:.3f} | {b.loc['temporal','rmse']:.3f} | {b.loc['spatial_50km','rmse']:.3f} |")
        new_tables.append('\n'.join(lines))
    process_metrics = pd.read_csv(process/'metrics.csv')
    flag_metrics = pd.read_csv(process/'qualified_flag_metrics.csv')
    flag_process = flag_metrics.loc[flag_metrics.model.eq('process_establishment')].set_index('evaluation')
    flag_control = flag_metrics.loc[flag_metrics.model.eq('climate_reference')].set_index('evaluation')
    labels = dict(climate_reference='Weather control', climate_renewal_shared='Development and delay, shared',
        climate_renewal_phase='Development and delay, phase mixture')
    rows = []
    for name, label in labels.items():
        a = metrics.loc[metrics.model.eq(name)].set_index('evaluation')
        b = groups.loc[groups.model.eq(name)&groups.population.eq('severe_above75')].set_index('evaluation')
        rows.append(f"| {label} | {a.loc['temporal','rmse']:.3f} | {a.loc['spatial_50km','rmse']:.3f} | {b.loc['temporal','rmse']:.3f} | {b.loc['spatial_50km','rmse']:.3f} |")
    monitor_rows = []
    for name in decisions:
        row = other_metrics.loc[other_metrics.model.eq(name)&other_metrics.evaluation.eq('temporal')].iloc[0]
        severe = other_groups.loc[other_groups.model.eq(name)&other_groups.evaluation.eq('temporal')&other_groups.population.eq('severe_above75')].iloc[0]
        spatial = other_metrics.loc[other_metrics.model.eq(name)&other_metrics.evaluation.eq('spatial_50km')]
        spatial_text = f'{spatial.rmse.iloc[0]:.3f}' if len(spatial) else 'Not assessed'
        monitor_rows.append(f'| `{name}` | {row.rmse:.3f} | {severe.rmse:.3f} | {abs(severe.bias):.3f} | {spatial_text} |')
    control = metrics.loc[metrics.model.eq('climate_reference')].set_index('evaluation')
    process_gain = 100*(1-process_metrics.loc[process_metrics.model.eq('process_establishment')].set_index('evaluation').rmse/control.rmse)
    phase = metrics.loc[metrics.model.eq('climate_renewal_phase')].set_index('evaluation')
    gains = 100*(1-phase.rmse/control.rmse)
    all_achieved = any(d['achieved'] for d in decisions.values())
    auto_table = '\n'.join(rows); monitor_table = '\n'.join(monitor_rows)
    intervals = comparisons['climate_renewal_phase']['paired_rmse_difference_intervals']
    body = f'''# Autonomous climate-compatible wheat disease assessment

## Information conditions and deployment audit

Satellite predictors and satellite-trained auxiliary representations are excluded. The autonomous model uses daily mean temperature, maximum temperature, relative humidity and total precipitation, together with calendar, coordinates, estimated sowing and a declared source-native severity endpoint. Neither current-season disease nor observed BBCH enters its runtime inputs. Qualified station precipitation remains preferred; measured rain hours, humid hours, Tmin and liquid-phase conversions are absent from this experiment.

The archived process candidate has 72 estimator inputs: 37 nominally process-oriented predictors and 35 observed-history channels. The earlier history-blind formulation retains nine observed-BBCH descriptors and a Tmin-derived frost clock. The archived weather RF is observation-free at runtime but relies on Tmin-dependent, later-trained crop-development representations, query-day weather and different historical training memberships. None supplies the matched four-variable autonomous comparison across the current 20 splits. Monitoring RMSE consequently cannot establish autonomous climate accuracy.

The nighttime satellite retrieval was cancelled before qualified features or disease fitting. Earlier satellite experiments remain provenance archives and have no role in the new predictors, auxiliary training or adoption decision. The frozen climate kernel and publication Figure 1 are unchanged.

## Cohort, timing and model structure

All 2,304 dated untreated targets remain in the backbone training cohort. Every design retains the original 2,017 assessment records, 192 field-seasons and 121 locations. Five temporal and 15 chronology-plus-50 km splits preserve original identities and hierarchy weights. Every training season precedes its assessment season. Weather ends on the day before each query, and future weather or assessment schedules cannot alter an earlier forecast.

The sowing prior uses 19,722 observations from 1,559 qualified historical stations before 2013, with previous-calendar annual mean temperature and latitude as RF predictors. The conditional-mean dates have complete four-variable weather prefixes for all 218 fields; no fallback or observation deletion occurs. Crop-development supervision uses 10,181 historical cycles at 632 stations before 2013. All admitted documented coordinates exceed the existing 50 km exclusion from current disease locations. Runtime development increments require Tmean and Tmax only.

The control contains 34 weather/calendar/geographic/nominal-endpoint/generic-process predictors. The candidate adds 59 historical development summaries and 36 thermal-delay exposure descriptors. Three assumed delays of 10/20/30 thermal days at 18 °C combine with continuous-host and modeled BBCH31/51 eligibility. Pending, expressed, recently expressed and bounded cumulative dose summaries describe hypothetical mechanisms; they are neither measured leaf wetness nor validated native-leaf anatomy. Four modeled phases define an optional mixture of experts.

All formulations predict full severity percentages with zero observation anchors. Each uses 20 paired 80% coordinate-year members and the registered 200-iteration HGB recipe. Scientific mass is preserved before phase weighting. Fitting and prediction contain no hidden disease-history channels.

## Autonomous performance

Table 1. Complete-support autonomous RMSE in percentage points. Severe observations exceed 75% severity and retain their full-cohort relative scientific weights.

| Formulation | Temporal RMSE | Spatial RMSE | Temporal severe RMSE | Spatial severe RMSE |
|---|---:|---:|---:|---:|
{auto_table}

The phase formulation reduces total RMSE by {gains.loc['temporal']:.2f}% temporally and {gains.loc['spatial_50km']:.2f}% spatially relative to the matched autonomous control. Both renewal formulations improve severe-case RMSE and absolute bias in both designs. Neither achieves 10% total-error reduction in both designs. Site-resampling intervals for the phase-minus-control RMSE difference are [{intervals['temporal'][0]:.3f}, {intervals['temporal'][1]:.3f}] temporally and [{intervals['spatial_50km'][0]:.3f}, {intervals['spatial_50km'][1]:.3f}] spatially. These are descriptive intervals after adaptive internal development, without independent confirmation.

## Optional same-field monitoring

The autonomous global weather response can supply a prior for separate observation-conditioned monitoring. Covariance calibration uses only historical ensemble predictions from members excluding the entire query coordinate-year. Records without an excluded-group member retain missing OOB predictions and do not calibrate the covariance; their targets remain in backbone fitting and evaluation. Missing OOB rows across the five temporal fits number 0, 85, 18, 20 and 76, accounting for 0–2.381% of original training mass. OOB ensemble size varies and has no calibrated uncertainty interpretation.

Six prespecified combinations use the three frozen priors and linear or logit discrepancy updates. The unchanged shared-field Matern and same-nominal-code exponential kernel uses strictly earlier measurements in the same field. All query-day native codes remain withheld together, other fields cannot update a query, and first visits preserve the autonomous prior. No additional HGB model is fitted. The no-observation climate run remains the separately evaluated autonomous prior.

Table 2. Optional-monitoring results in percentage points. Spatial assessment requires one candidate to pass every original temporal goal condition.

| Formulation | Temporal RMSE | Temporal severe RMSE | Temporal severe absolute bias | Spatial RMSE |
|---|---:|---:|---:|---:|
{monitor_table}

The original monitoring objective is {'achieved' if all_achieved else 'not achieved'}. Its thresholds remain 15.61851480377759 temporally and 15.866048678379531 spatially, together with improved severe-case RMSE and absolute bias in both designs. A screened-out spatial result remains unassessed. Monitoring improvement, when present, does not qualify the autonomous climate model.

## Verification and scientific limits

The initial autonomous and monitoring implementation passed 517 tests with seven pre-existing NumPy timedelta deprecation warnings. Independent numerical reconstruction reproduces all 2,304 autonomous predictor rows, all 20 saved fits and 2,400 HGB heads, original memberships and weights, zero anchors, fitted initial means, saved forecasts and both separately labeled goal sets. Optional monitoring has separate OOB, source-hash, covariance-objective and forecast replay receipts. The subsequent calibration, process and resistance-scenario checks are reported below. No model is promoted.

Autonomous prediction still has a substantial accuracy gap. The experiments identify an information mismatch and the performance of the tested formulations; they do not establish a unique weather, observation or biological cause, or an irreducible error floor. Native endpoint equivalence, cultivar and resistance, field inoculum, exact severity denominator and canopy wetness remain unverified. No additional observation is labelled erroneous or removed by these experiments.

Future RF sowing requires the calendar year preceding autumn sowing, an additional year beyond the frozen two-year climate loader. Source-native severity predictions also lack a generally verified mapping to anatomical F1/F2/F3. Climate integration and yield-loss use therefore remain unqualified. The unchanged frozen climate kernel is a separate scientific implementation.

## Historical severity calibration

Six fixed calibration formulations use only excluded-coordinate-year predictions from the frozen autonomous ensemble. The logit map fits location and positive slope under weighted bounded squared error; isotonic calibration fits a monotone weighted response. Covered rows retain their original relative scientific mass. A spatial training population contains only four coordinate-years, all included in every 80% member after rounding; its six calibration states preserve the uncalibrated predictions through an explicit identity fallback. Missing OOB values remain missing and never receive fitted predictions. The incomplete first execution and its failure amendment remain archived separately from the complete repaired execution.

Table 3. Calibration RMSE in percentage points on the complete original assessment support. Severe observations exceed 75% severity.

| Formulation | Temporal RMSE | Spatial RMSE | Temporal severe RMSE | Spatial severe RMSE |
|---|---:|---:|---:|---:|
{new_tables[0]}

Lower total error after calibration coexists with worse severe-case error. No calibration formulation passes either complete goal set. A spatial total-RMSE reduction alone does not satisfy the severe-case requirements.

## Climate calibration ancestry and causal process validation

The frozen climate parameters record no disease-magnitude fitting. Independently authenticated membership contains 20 calibration fields in 2017 and eight in 2018, followed by 45 validation fields in 2019. Those parameters and their associated leaf-stage calibration cannot supply independent constants for the newer 2015–2018 outer folds. This ancestry does not establish leakage in the original 2019 validation. The original parameter-selection trace remains unavailable; its recorded hash alone does not independently verify the optimization.

A separate daily prefix adapter agrees exactly with the frozen process equations on completed seasons and matched prefixes. It emits in-progress states without requiring future terminal-stage completion. Future missing weather cannot change earlier prefix states. The unchanged full-season wrapper retains its census validity rule.

Only 115 retained BASF code 0 targets, spanning 55 fields, have a qualified anatomical F1 interpretation. Other source-native codes remain nominal categories. Their training-only convex observation operators combine eight modeled canopy slots without asserting recovered anatomy or full EPPO protocol compliance. Every original assessment record remains scored.

The new process bank uses 27 combinations of primary infection scale, secondary infection scale and latent reference duration. A disjoint pre-2013 crop-development prior and qualified RF sowing replace the old disease-cohort calibration ancestry. Flag visibility and unfolding are assumed clock fractions between stages 31 and 51. The terminal donor label is observed 87, internally aliased to 85; it does not establish soft-dough or anthesis qualification. Delayed introductions at zero, two and six effective wet days plus an escaped component represent hypothetical establishment variation. Each outer fold fits disease magnitude, nominal operators and optional mixture weights only on its historical training fields.

Table 4. Shared-equation process RMSE in percentage points. The generic arm fixes physical parameters; the selected arm fits physical parameters; the establishment arm additionally fits the introduction/escape mixture.

| Formulation | Temporal RMSE | Spatial RMSE | Temporal severe RMSE | Spatial severe RMSE |
|---|---:|---:|---:|---:|
{new_tables[1]}

The establishment formulation reduces autonomous total RMSE by {process_gain.loc['temporal']:.2f}% temporally and {process_gain.loc['spatial_50km']:.2f}% spatially against the matched weather control. Severe-case RMSE and absolute bias deteriorate in both designs. Both total-error thresholds pass, while the complete autonomous goal and the original monitoring goal remain unmet. These adaptive internal comparisons do not qualify climate deployment.

Every selected establishment fit uses the largest tested primary and secondary infection scales. The search therefore does not establish interior estimates for these two rates; broader prespecified ranges remain a sensitivity requirement.

On the separately qualified BASF flag-leaf endpoint, establishment-model RMSE is {flag_process.loc['temporal','rmse']:.3f} temporally and {flag_process.loc['spatial_50km','rmse']:.3f} spatially, compared with {flag_control.loc['temporal','rmse']:.3f} and {flag_control.loc['spatial_50km','rmse']:.3f} for the weather control. The deterioration on an anatomically identified endpoint further limits physical interpretation of the complete-support gain.

## Variety resistance and climate scenarios

An explicit nonnegative susceptibility multiplier scales primary and secondary infection rates. Its neutral value is 1; lower values represent assumed lower susceptibility. It is a scenario parameter rather than a numerical conversion of an official resistance scale. A variety name alone supplies no verified rating. Exact field-to-variety joins require unique field identities and source provenance; unresolved and unseen varieties retain the pooled neutral setting.

Independent inspection of both original binary disease tables finds no cultivar column: BASF contains 2,276 rows and 17 columns, and Corteva contains 112,270 rows and 16 columns. Names in unrelated external trial cohorts cannot establish identities for these field-seasons. No cultivar-specific effect or performance gain is estimated until verified linked trial metadata is available. Historical training-only estimation and a pooled fallback remain necessary for any subsequent variety model.

The closest available station-match files contain meteorological station identities, including ORLEANS BRICY and München-Flughafen, without cultivar columns. A matching weather-station name cannot establish a wheat variety. The exact field-key linkage template contains 218 field-seasons and no admitted cultivar observations.

Six process scenarios recompute development and disease jointly under baseline weather, +2°C warming, precipitation multipliers of 0.8 and 1.2, and susceptibility multipliers of 0.25 and 2. Numerical checks verify finite bounded states and compartment mass conservation; they do not establish empirical climate response or a universal monotone warming effect. All 81 bank trajectories, 1,080 process fits, 120 calibration states, original weights, saved predictions and metric/goal calculations have replay receipts. The expanded repository suite passes {suite['passed']} tests with {suite['warnings']} pre-existing warnings. The proposed RF-sown severity model remains unqualified because endpoint equivalence, severe-case accuracy, scenario sowing support and healthy-area/yield interpretation remain unresolved.

Independent review identified a public-interface chronology defect: a pre-sowing query in a mixed-validity batch could index a future trajectory state. Per-query validation now rejects dates on or before estimated sowing before simulation. All current queries occur at least 160 days after estimated sowing, and all six scenario damage and host trajectories reproduce exactly after the guard. Original process-bank code, fits and numerical results remain unchanged; the repair has separate provenance and regression evidence.

## Numerical archive

Autonomous inputs, per-query timing and sowing provenance reside in [data/disease_improvement/autonomous_weather_20261022](../data/disease_improvement/autonomous_weather_20261022). Complete autonomous fits, metrics, predictions and verification receipts reside in [autonomous_weather_20261022](../outputs/disease_improvement/autonomous_weather_20261022); optional-monitoring equivalents reside in [autonomous_monitoring_20261022](../outputs/disease_improvement/autonomous_monitoring_20261022). Numerical CSV/NPZ data remain independent of the displays for reproducibility.

Severity calibration resides in [severity_calibration_support_repair_20261022](../outputs/disease_improvement/severity_calibration_support_repair_20261022), process fits in [process_establishment_20261022](../outputs/disease_improvement/process_establishment_20261022), and numerical qualification in [calibration_process_verification_20261022](../outputs/disease_improvement/calibration_process_verification_20261022). The [climate calibration audit](../outputs/disease_improvement/climate_calibration_audit_20261022/audit.json) and [variety metadata audit](../outputs/disease_improvement/variety_metadata_audit_20261022/audit.json) preserve source-specific evidence.

The [chronology repair receipt](../outputs/disease_improvement/scenario_chronology_repair_20261022/verification.json) binds the reviewed source, corrected interface, unchanged scenario outputs and latest test suite. The [variety linkage template](../outputs/disease_improvement/variety_metadata_audit_20261022/field_variety_link_template.csv) preserves exact current field identities separately from unverified cultivar information.
'''
    destination = ROOT/'docs/Autonomous_Climate_Disease_Assessment_20261022.md'
    destination.write_text(body)
    manifest = dict(report=str(destination.relative_to(ROOT)), report_sha256=sha(destination), generator_sha256=sha(Path(__file__)),
        sources={str(p.relative_to(ROOT)): sha(p) for out in [autonomous, monitoring, calibration, process, audit, verification, repair, variety] for p in out.iterdir()
            if p.is_file() and p.suffix in ['.csv', '.json']})
    path = ROOT/'outputs/disease_improvement/autonomous_assessment_20261022'
    path.mkdir(exist_ok=True)
    (path/'report_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(destination)
    return destination


if __name__ == '__main__':
    write()
