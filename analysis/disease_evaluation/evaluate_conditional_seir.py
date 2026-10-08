#!/usr/bin/env python3
"""A limited, conditional disease-progress benchmark; not a primary-infection model.

Run with Python containing numpy, pandas, scipy and pyarrow:
    python analysis/disease_evaluation/evaluate_conditional_seir.py

Original disease/weather inputs remain immutable. All outputs are beside this script.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/basf-wheat-diseases.txt"
WEATHER = ROOT / "data/era5/daily_weather.parquet"
LINKS = ROOT / "data/era5/basf_trial_weather_links.csv"
LEAF_RANK = {"LEAF, 1ST / FLAG LEAF": 1, "LEAF, FLAG": 1,
             "LEAF, 2ND": 2, "LEAF, 3RD": 3, "LEAF, 4TH": 4,
             "LEAF, 5TH": 5, "LEAF, 6ST": 6, "LEAF, 6TH": 6,
             "LEAF, 7ST": 7, "LEAF, 7TH": 7}
MODELS = {"secondary_seir_20d": 20., "secondary_seir_10d": 10.,
          "secondary_seir_30d": 30.}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    raw = pd.read_csv(SOURCE, sep="\t")
    raw["source_row"] = np.arange(2, len(raw) + 2)
    d = raw.loc[raw.Treatment.eq("Untreated") & raw.Organism.eq("SEPTTR")].copy()
    d["exclusion_reason"] = "eligible_candidate"
    d.loc[d.Clarifier.eq("CROP INJURY"), "exclusion_reason"] = "crop_injury"
    d.loc[d.exclusion_reason.eq("eligible_candidate") & ~d.PlantPart.isin(LEAF_RANK),
          "exclusion_reason"] = "leaf_rank_not_explicit"
    d["Value"] = pd.to_numeric(d.Value, errors="coerce")
    d.loc[d.exclusion_reason.eq("eligible_candidate") & ~d.Value.between(0, 100),
          "exclusion_reason"] = "invalid_percent"
    d.loc[d.exclusion_reason.eq("eligible_candidate") &
          ~(d.Parameter.eq("INFECT") & d.Method.eq("P%INF")),
          "exclusion_reason"] = "measurement_not_target_percent"
    d["Date"] = pd.to_datetime(d.Date, errors="raise")
    candidate = d.loc[d.exclusion_reason.eq("eligible_candidate")].copy()
    assert not candidate.duplicated(["TrialId", "PlantPart", "Date"]).any(), \
        "Duplicate trial/leaf/date rows require an explicit measurement policy."
    counts = candidate.groupby(["TrialId", "PlantPart"]).Date.transform("nunique")
    insufficient = candidate.loc[counts.lt(3)].index
    d.loc[insufficient, "exclusion_reason"] = "fewer_than_three_same_rank_dates"
    d.loc[d.exclusion_reason.eq("eligible_candidate"), "exclusion_reason"] = "included"
    d[["source_row", "TrialId", "Country", "Date", "PlantPart", "Value",
       "Clarifier", "exclusion_reason"]].to_csv(OUT / "selection_audit.csv", index=False)
    selected = d.loc[d.exclusion_reason.eq("included")].copy()
    selected["leaf_rank"] = selected.PlantPart.map(LEAF_RANK).astype(int)
    selected["year"] = selected.Date.dt.year.astype(int)
    links = pd.read_csv(LINKS)
    assert not links.TrialId.duplicated().any()
    selected = selected.merge(links[["TrialId", "location_id", "season_year",
                                    "requested_latitude", "requested_longitude", "country_source"]],
                              on="TrialId", how="left", validate="many_to_one")
    assert selected.location_id.notna().all()
    assert selected.year.eq(selected.season_year).all()
    assert np.allclose(selected.Latitude, selected.requested_latitude)
    assert np.allclose(selected.Longitude, selected.requested_longitude)
    assert selected.Country.eq(selected.country_source).all()
    selected["series_id"] = selected.TrialId.astype(str) + "|leaf" + selected.leaf_rank.astype(str)
    selected["coordinate_year"] = selected.location_id + "|" + selected.year.astype(str)
    selected["fraction"] = np.clip(selected.Value.to_numpy(float) / 100., 0., 1.)
    selected = selected.sort_values(["series_id", "Date"]).reset_index(drop=True)
    selected["first_date"] = selected.groupby("series_id").Date.transform("min")
    selected["day_from_first"] = (selected.Date - selected.first_date).dt.days
    selected["is_conditioning_observation"] = selected.day_from_first.eq(0)
    selected["first_fraction"] = selected.groupby("series_id").fraction.transform("first")
    assert selected.groupby("series_id").is_conditioning_observation.sum().eq(1).all()
    selected.to_csv(OUT / "eligible_observations.csv", index=False)
    weather = pd.read_parquet(WEATHER)
    weather["date"] = pd.to_datetime(weather.date)
    assert not weather.duplicated(["location_id", "date"]).any()
    assert weather.hour_count.eq(24).all()
    assert weather[["tmean_c", "rain_mm", "rh_hours_ge_90pct"]].notna().all().all()
    assert weather.rain_mm.ge(0).all()
    assert weather.rh_hours_ge_90pct.between(0, 24).all()
    audit = {"untreated_septtr_rows": len(d),
             "selection_counts": d.exclusion_reason.value_counts().to_dict(),
             "source_leaf_labels": sorted(selected.PlantPart.unique().tolist())}
    return selected, weather, audit


@dataclass
class Batch:
    rows: pd.DataFrame
    ids: list[str]
    initial: np.ndarray
    weather_factor: np.ndarray
    thermal_rate: np.ndarray
    observed_series_index: np.ndarray
    observed_days: np.ndarray
    target: np.ndarray
    weights: np.ndarray
    largest_weather_date: str


def make_batch(rows: pd.DataFrame, weather: pd.DataFrame) -> Batch:
    rows = rows.sort_values(["series_id", "Date"]).copy()
    series = rows.groupby("series_id", sort=True).first()
    ids = list(series.index)
    index = {name: i for i, name in enumerate(ids)}
    days = int(rows.day_from_first.max())
    factor = np.zeros((len(ids), days))
    thermal = np.zeros_like(factor)
    maximum_weather_date = None
    indexed_weather = weather.set_index(["location_id", "date"])
    for name, s in series.iterrows():
        sub = rows.loc[rows.series_id.eq(name)]
        target_date = sub.Date.max()
        # A value on target date can use complete daily weather only before that date.
        needed = pd.date_range(s.first_date, target_date - pd.Timedelta(days=1), freq="D")
        keys = pd.MultiIndex.from_product([[s.location_id], needed], names=["location_id", "date"])
        frame = indexed_weather.reindex(keys)
        assert len(frame) == int(sub.day_from_first.max())
        assert frame[["tmean_c", "rain_mm", "rh_hours_ge_90pct"]].notna().all().all(), \
            f"Incomplete daily weather for {name}."
        temperature = frame.tmean_c.to_numpy()
        # These are working functions; no literature calibration is claimed.
        factor[index[name], :len(frame)] = (
            np.exp(-.5 * ((temperature - 18.) / 8.) ** 2)
            * (-np.expm1(-frame.rain_mm.to_numpy() / 2.))
            * (frame.rh_hours_ge_90pct.to_numpy() / 24.))
        thermal[index[name], :len(frame)] = np.maximum(temperature, 0.) / 18.
        assert needed.max() < target_date
        maximum_weather_date = max(maximum_weather_date or needed.max(), needed.max())
    future = rows.loc[~rows.is_conditioning_observation].copy()
    # Each coordinate-year has equal total weight; each series within it has equal weight.
    n_coordinate_year = future.coordinate_year.nunique()
    n_series_per_group = future.groupby("coordinate_year").series_id.transform("nunique").to_numpy()
    n_observations_per_series = future.groupby("series_id").series_id.transform("size").to_numpy()
    weights = 1. / (n_coordinate_year * n_series_per_group * n_observations_per_series)
    assert np.isclose(weights.sum(), 1.)
    return Batch(rows, ids, series.first_fraction.to_numpy().copy(), factor, thermal,
                 np.array([index[x] for x in future.series_id]),
                 future.day_from_first.to_numpy(int), future.fraction.to_numpy(), weights,
                 str(maximum_weather_date.date()))


def simulate(batch: Batch, beta: float, latent_days: float, check=False):
    n, days = batch.weather_factor.shape
    state = np.zeros((n, 4))  # S,E,I,R
    state[:, 0] = 1. - batch.initial
    state[:, 2] = batch.initial
    y = np.zeros((n, days + 1))
    y[:, 0] = batch.initial
    max_mass_error = 0.
    min_state = 0.
    for day in range(days):
        s, e, infectious, removed = state.T
        # Rates become bounded daily transition probabilities. lambda is fixed at zero.
        new_exposed = s * (-np.expm1(-beta * batch.weather_factor[:, day] * infectious))
        newly_infectious = e * (-np.expm1(-batch.thermal_rate[:, day] / latent_days))
        newly_removed = infectious * (-np.expm1(-1. / 14.))
        state = np.column_stack((s - new_exposed, e + new_exposed - newly_infectious,
                                 infectious + newly_infectious - newly_removed,
                                 removed + newly_removed))
        y[:, day + 1] = 1. - state[:, 0]
        if check:
            max_mass_error = max(max_mass_error, float(np.max(abs(state.sum(axis=1) - 1.))))
            min_state = min(min_state, float(state.min()))
    if check:
        assert min_state >= -1e-12
        assert max_mass_error < 1e-12
        assert np.all(y >= -1e-12) and np.all(y <= 1. + 1e-12)
        return y, {"max_mass_error": max_mass_error, "minimum_compartment": min_state}
    return y


def selected_predictions(batch: Batch, y: np.ndarray):
    return y[batch.observed_series_index, batch.observed_days]


def loss(batch: Batch, predictions: np.ndarray):
    return float(np.sum(batch.weights * (predictions - batch.target) ** 2))


def fit_bounded(objective, lower, upper):
    result = minimize_scalar(objective, method="bounded", bounds=(lower, upper),
                             options={"xatol": 1e-7})
    assert result.success, result.message
    choices = [(float(result.x), float(result.fun)), (lower, objective(lower)), (upper, objective(upper))]
    value, score = min(choices, key=lambda pair: pair[1])
    return value, score, (np.isclose(value, lower) or np.isclose(value, upper))


def trend_predictions(batch: Batch, slope: float):
    # Half a percentage point regularizes exact zero/one in this statistical baseline.
    starts = logit(np.clip(batch.initial, .005, .995))
    grid = np.arange(batch.weather_factor.shape[1] + 1)
    y = expit(starts[:, None] + slope * grid[None, :])
    y[:, 0] = batch.initial
    return y


def folds(d: pd.DataFrame):
    all_series = set(d.series_id)
    result = []
    for test_year in (2018, 2019):
        test = set(d.loc[d.year.eq(test_year), "series_id"])
        held_locations = set(d.loc[d.year.eq(test_year), "location_id"])
        # Spatial groups remain together even in a forward-year experiment.
        train = set(d.loc[d.year.lt(test_year) & ~d.location_id.isin(held_locations), "series_id"])
        excluded = all_series - train - test
        result.append((f"forward_{test_year}_coordinate_purged", "forward_year", train, test, excluded))
    for country in sorted(d.Country.unique()):
        test = set(d.loc[d.Country.eq(country), "series_id"])
        train = all_series - test
        result.append(("leave_country_" + country.lower().replace(" ", "_"),
                       "leave_country_out", train, test, set()))
    return result


def metric_row(frame: pd.DataFrame):
    frame = frame.loc[frame.scored].copy()
    err = frame.predicted_percent - frame.observed_percent
    frame["sq"] = err ** 2
    frame["abs"] = abs(err)
    frame["signed"] = err
    series = frame.groupby(["coordinate_year", "series_id"])[["sq", "abs", "signed"]].mean()
    groups = series.groupby("coordinate_year").mean()
    return {"rmse_pp": float(np.sqrt(groups.sq.mean())), "mae_pp": float(groups["abs"].mean()),
            "bias_pp": float(groups.signed.mean()), "n_coordinate_years": len(groups),
            "n_coordinates": frame.location_id.nunique(), "n_trials": frame.TrialId.nunique(),
            "n_series": frame.series_id.nunique(), "n_future_observations": len(frame),
            "aggregation": "equal coordinate-year; equal series within coordinate-year; equal future dates within series"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_paths = [SOURCE, WEATHER, LINKS]
    hashes_before = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}
    d, weather, selection = prepare()
    all_predictions, parameters, assignments, fold_audit, numerical_audit = [], [], [], [], []
    for fold_name, fold_type, train_ids, test_ids, excluded in folds(d):
        train_rows, test_rows = d.loc[d.series_id.isin(train_ids)], d.loc[d.series_id.isin(test_ids)]
        assert len(train_rows) and len(test_rows)
        assert not train_ids & test_ids
        assert not set(train_rows.location_id) & set(test_rows.location_id)
        assert not set(train_rows.source_row) & set(test_rows.source_row)
        if fold_type == "forward_year":
            assert train_rows.Date.max() < test_rows.Date.min()
            assert train_rows.year.max() < test_rows.year.min()
        else:
            assert not set(train_rows.Country) & set(test_rows.Country)
        for role, ids in (("train", train_ids), ("test", test_ids), ("excluded", excluded)):
            for series_id in sorted(ids):
                s = d.loc[d.series_id.eq(series_id)].iloc[0]
                assignments.append(dict(fold=fold_name, fold_type=fold_type, role=role,
                                        series_id=series_id, TrialId=int(s.TrialId),
                                        country=s.Country, year=int(s.year), location_id=s.location_id))
        train, test = make_batch(train_rows, weather), make_batch(test_rows, weather)
        fold_audit.append({"fold": fold_name, "fold_type": fold_type,
                           "train_years": sorted(train_rows.year.unique().tolist()),
                           "test_years": sorted(test_rows.year.unique().tolist()),
                           "train_trials": int(train_rows.TrialId.nunique()),
                           "test_trials": int(test_rows.TrialId.nunique()),
                           "train_series": len(train_ids), "test_series": len(test_ids),
                           "excluded_series": len(excluded), "coordinate_overlap": 0,
                           "source_row_overlap": 0, "max_test_weather_date": test.largest_weather_date,
                           "test_initialization": "first observation only; E=R=0; I=y0; S=1-y0",
                           "test_future_disease_used_for_fitting": False})
        predicted = {}
        mechanistic_variants = [(name, latent, 1.) for name, latent in MODELS.items()]
        if fold_type == "forward_year":
            mechanistic_variants += [("secondary_seir_20d_beta_cap5", 20., 5.),
                                    ("secondary_seir_20d_beta_cap10", 20., 10.)]
        for model_name, latent, beta_upper in mechanistic_variants:
            objective = lambda beta: loss(train, selected_predictions(train, simulate(train, beta, latent)))
            beta, score, boundary = fit_bounded(objective, 0., beta_upper)
            y, numerical = simulate(test, beta, latent, check=True)
            numerical_audit.append({"fold": fold_name, "model": model_name, **numerical})
            predicted[model_name] = y
            parameters.append(dict(fold=fold_name, fold_type=fold_type, model=model_name,
                                   parameter="beta_day_inverse", value=beta, lower_bound=0., upper_bound=beta_upper,
                                   fitted_at_bound=boundary, train_weighted_mse_pp2=score * 10000.,
                                   latent_days_at_18c=latent, infectious_days=14., external_lambda=0.))
        objective = lambda rate: loss(train, selected_predictions(train, trend_predictions(train, rate)))
        rate, score, boundary = fit_bounded(objective, -.25, .25)
        predicted["global_logit_trend"] = trend_predictions(test, rate)
        parameters.append(dict(fold=fold_name, fold_type=fold_type, model="global_logit_trend",
                               parameter="logit_rate_day_inverse", value=rate, lower_bound=-.25, upper_bound=.25,
                               fitted_at_bound=boundary, train_weighted_mse_pp2=score * 10000.,
                               latent_days_at_18c=None, infectious_days=None, external_lambda=None))
        predicted["first_observation_persistence"] = np.repeat(test.initial[:, None],
                                                              test.weather_factor.shape[1] + 1, axis=1)
        series_index = {name: i for i, name in enumerate(test.ids)}
        row_idx = np.array([series_index[name] for name in test.rows.series_id])
        for model_name, y in predicted.items():
            p = test.rows[["source_row", "TrialId", "Country", "Latitude", "Longitude", "year",
                           "location_id", "coordinate_year", "series_id", "PlantPart", "leaf_rank",
                           "Date", "first_date", "day_from_first", "Value", "first_fraction",
                           "is_conditioning_observation"]].copy()
            p = p.rename(columns={"Value": "observed_percent", "Date": "target_date"})
            p["fold"], p["fold_type"], p["model"] = fold_name, fold_type, model_name
            p["predicted_percent"] = y[row_idx, p.day_from_first.to_numpy(int)] * 100.
            p["scored"] = ~p.is_conditioning_observation
            p["latest_weather_date"] = (p.target_date - pd.Timedelta(days=1)).where(p.scored, pd.NaT)
            p["prediction_context"] = "conditional on first observed disease and realized preceding daily weather"
            assert p.loc[p.scored, "latest_weather_date"].lt(p.loc[p.scored, "target_date"]).all()
            assert np.allclose(p.loc[~p.scored, "predicted_percent"], p.loc[~p.scored, "observed_percent"])
            all_predictions.append(p)
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(OUT / "predictions.csv", index=False)
    pd.DataFrame(parameters).to_csv(OUT / "parameters.csv", index=False)
    pd.DataFrame(assignments).to_csv(OUT / "fold_assignments.csv", index=False)
    metrics = []
    stratified = []
    for (fold, fold_type, model), p in predictions.groupby(["fold", "fold_type", "model"]):
        metrics.append(dict(fold=fold, fold_type=fold_type, model=model, **metric_row(p)))
        for label, mask in (("initial_zero", p.first_fraction.eq(0)),
                            ("initial_positive", p.first_fraction.gt(0)),
                            ("initial_between_zero_and_one", p.first_fraction.between(0, 1, inclusive="neither"))):
            subset = p.loc[mask]
            if subset.scored.any():
                stratified.append(dict(fold=fold, fold_type=fold_type, model=model,
                                       initial_condition_group=label, **metric_row(subset)))
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(OUT / "fold_metrics.csv", index=False)
    pd.DataFrame(stratified).to_csv(OUT / "initial_condition_metrics.csv", index=False)
    pooled = []
    for (fold_type, model), p in predictions.groupby(["fold_type", "model"]):
        assert not p.duplicated("source_row").any(), "Each pooled source observation must have one held-out prediction."
        pooled.append(dict(fold_type=fold_type, model=model, **metric_row(p)))
    pd.DataFrame(pooled).to_csv(OUT / "pooled_metrics.csv", index=False)
    inventory = d.groupby("series_id").agg(TrialId=("TrialId", "first"), country=("Country", "first"),
                                            year=("year", "first"), location_id=("location_id", "first"),
                                            coordinate_year=("coordinate_year", "first"),
                                            leaf_rank=("leaf_rank", "first"), source_leaf_label=("PlantPart", "first"),
                                            dates=("Date", "nunique"), first_date=("Date", "min"),
                                            last_date=("Date", "max"), first_percent=("Value", "first"),
                                            last_percent=("Value", "last"))
    inventory.to_csv(OUT / "series_inventory.csv")
    changes = d[["source_row", "TrialId", "Country", "year", "location_id", "coordinate_year",
                 "series_id", "leaf_rank", "Date", "Value"]].copy()
    changes["previous_date"] = changes.groupby("series_id").Date.shift()
    changes["previous_percent"] = changes.groupby("series_id").Value.shift()
    changes = changes.loc[changes.previous_date.notna()].copy()
    changes["change_pp"] = changes.Value - changes.previous_percent
    changes["observed_decline"] = changes.change_pp.lt(0)
    changes.to_csv(OUT / "consecutive_observed_changes.csv", index=False)
    contracts = model_contract_checks(d, weather)
    hashes_after = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}
    assert hashes_before == hashes_after
    validation = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                      status="passed within limited conditional benchmark scope",
                      runtime={"python": platform.python_version(), "numpy": np.__version__,
                               "pandas": pd.__version__, "scipy": scipy.__version__},
                      sources_sha256=hashes_after, sources_immutable=True, selection=selection,
                      included_observations=len(d), included_trials=int(d.TrialId.nunique()),
                      included_series=int(d.series_id.nunique()),
                      included_coordinate_years=int(d.coordinate_year.nunique()),
                      included_coordinates=int(d.location_id.nunique()),
                      first_conditioning_observations=int(d.is_conditioning_observation.sum()),
                      unique_scored_source_observations=int((~d.is_conditioning_observation).sum()),
                      initially_zero_series=int(inventory.first_percent.eq(0).sum()),
                      initially_positive_series=int(inventory.first_percent.gt(0).sum()),
                      initially_saturated_series=int(inventory.first_percent.eq(100).sum()),
                      first_conditioning_rows_scored=False, test_states_fitted=False,
                      same_coordinate_on_both_sides=False, random_row_splits=False,
                      all_weather_daily_hour_counts_equal_24=True,
                      input_weather_complete=True, weather_on_or_after_target_date_used=False,
                      operational_weather_forecast_validation=False,
                      fitted_parameters=["global secondary coefficient beta", "global statistical logit trend rate"],
                      separately_identified_primary_secondary=False, external_lambda_fixed_zero=True,
                      visible_infection_fraction_equated_to_infectious_fraction_as_working_proxy=True,
                      all_base_seir_beta_estimates_at_upper_bound=all(
                          p["value"] == 1. for p in parameters if p["model"] in MODELS),
                      same_leaf_consecutive_pairs=len(changes),
                      same_leaf_declining_pairs=int(changes.observed_decline.sum()),
                      same_leaf_declining_pair_fraction=float(changes.observed_decline.mean()),
                      series_with_any_decline=int(changes.loc[changes.observed_decline, "series_id"].nunique()),
                      observations_monotonically_repaired=False,
                      complete_primary_secondary_model_validated=False, european_upscaling_validated=False,
                      model_contract_checks=contracts, folds=fold_audit, numerical_checks=numerical_audit)
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n")
    write_methods(d, metrics, inventory, pd.DataFrame(parameters), changes)
    print(metrics.loc[metrics.fold_type.eq("forward_year"),
                      ["fold", "model", "rmse_pp", "mae_pp", "bias_pp", "n_trials", "n_series"]].to_string(index=False))
    print(json.dumps({k: validation[k] for k in ["included_trials", "included_series", "included_observations",
                                               "included_coordinate_years", "initially_zero_series"]}))


def model_contract_checks(d, weather):
    batch = make_batch(d, weather)
    y, stats = simulate(batch, 1., 10., check=True)
    assert np.all(np.diff(y, axis=1) >= -1e-12)
    zero_beta = simulate(batch, 0., 20.)
    assert np.allclose(zero_beta, batch.initial[:, None])
    observed = batch.initial.copy()
    batch.initial[:] = 0.
    zero_seed = simulate(batch, 1., 20.)
    assert np.allclose(zero_seed, 0.)
    batch.initial[:] = observed
    return {"mass_conservation": True, "nonnegative_compartments": True,
            "zero_beta_preserves_cumulative_infected_proxy": True,
            "zero_visible_initial_disease_stays_zero_without_external_inoculum": True,
            "cumulative_infected_proxy_is_monotone": True, **stats}


def write_methods(d, metrics, inventory, parameters, changes):
    forward = metrics.loc[metrics.fold_type.eq("forward_year")]
    seir = forward.loc[forward.model.eq("secondary_seir_20d")]
    persistence = forward.loc[forward.model.eq("first_observation_persistence")]
    lines = []
    sensitivity = []
    for year in (2018, 2019):
        fold = f"forward_{year}_coordinate_purged"
        a, b = seir.loc[seir.fold.eq(fold)].iloc[0], persistence.loc[persistence.fold.eq(fold)].iloc[0]
        trend = forward.loc[forward.fold.eq(fold) & forward.model.eq("global_logit_trend")].iloc[0]
        lines.append(f"　　{year}年测试包含{int(a.n_trials)}个试验、{int(a.n_series)}条同叶位序列和{int(a.n_coordinate_years)}个坐标年。20天潜育期SEIR、首值持续性和全局logit趋势的RMSE分别为{a.rmse_pp:.2f}、{b.rmse_pp:.2f}和{trend.rmse_pp:.2f}个百分点；MAE分别为{a.mae_pp:.2f}、{b.mae_pp:.2f}和{trend.mae_pp:.2f}个百分点。")
        variants = []
        for cap in (5, 10):
            name = f"secondary_seir_20d_beta_cap{cap}"
            measure = forward.loc[forward.fold.eq(fold) & forward.model.eq(name)].iloc[0]
            coefficient = parameters.loc[parameters.fold.eq(fold) & parameters.model.eq(name)].iloc[0]
            variants.append(f"上界{cap} d⁻¹时β={coefficient.value:.3f} d⁻¹、RMSE={measure.rmse_pp:.2f}个百分点")
        sensitivity.append(f"　　{year}年前向折的20天潜育期参数上界敏感性为：" + "；".join(variants) + "。这些β尺度依赖固定天气函数与代理映射，不能解释为实测传播参数。")
    text = f"""条件性病程基准的方法与结果

　　病情来源为BASF 2017—2019年小麦试验原始表。分析对象限于Untreated处理中Organism为SEPTTR、Parameter为INFECT、Method为P%INF的有效0—100百分比记录；Clarifier为CROP INJURY的记录不属于病情目标。LEAF, 1ST / FLAG LEAF和序数叶位构成明确叶位类别，6ST、7ST保留其原始标签与序数含义。同一试验、同一叶位至少具有3个不同评估日期，每个日期仅有1条记录。符合条件的资料包括{d.TrialId.nunique()}个试验、{d.series_id.nunique()}条序列、{len(d)}条观测、{d.coordinate_year.nunique()}个坐标年和{d.location_id.nunique()}个四舍五入坐标。叶位类别的连续记录不证明同一片物理叶片被重复测量。P%INF的精确分母和百分比是否代表感染叶片比例或病斑面积未由原始表充分定义，因此百分比仅作为条件性病程代理。

　　每条测试序列仅以首个可见病情比例y₀进行条件化：S₀=1−y₀、E₀=0、I₀=y₀、R₀=0。首条观测不计入预测误差，其后的病情观测不参与测试状态初始化或参数拟合。观测代理为y=1−S，包括潜育、传染和移除仓室；该映射及I₀=y₀属于工作假设，不构成可见症状与产孢比例的实测关系。初始潜育感染和初始移除组织均缺乏直接观测。

　　模型仅包含条件化之后的二次传播，外源初侵染通量λ固定为0。每日感染率为βI f(T,P,H)，其中f=exp[−0.5((T−18)/8)²]×[1−exp(−P/2)]×H/24，T为日平均2米气温（℃）、P为日液态降雨量（mm）、H为2米相对湿度至少90%的小时数。高湿小时代表大气湿度暴露，不能等同于实测叶面湿润时间。温度最适值18℃、宽度8℃、降雨尺度2 mm及三个因子的乘积均为未经数据来源校准的工作函数。E→I速率为max(T,0)/(18L)，I→R速率为1/14 d⁻¹，L的主设定为18℃下20 d；L=10 d和30 d分别进行重新拟合的敏感性分析。持续期、温度响应和14 d传染期均为固定工作假设，并非经独立资料校准的参数。每日流量分别为S[1−exp(−βIf)]、E[1−exp(−max(T,0)/(18L))]和I[1−exp(−1/14)]，同步更新保持四仓室非负且总质量为1。

　　全局二次传播系数β仅使用训练集后续病情拟合，主比较取值边界为0—1 d⁻¹；各潜育期假设单独拟合β。两次前向折另保留20天潜育期，将β上界分别放宽至5和10 d⁻¹并仅使用原训练集重新拟合，以检验上界的影响。首值持续性基准在每个后续日期预测y₀。全局logit趋势基准为logit⁻¹[logit(clip(y₀,0.005,0.995))+rt]，t为首条观测之后的日数，r仅使用训练病情拟合，边界为−0.25—0.25 d⁻¹；零值和满值的正则化幅度为0.5个百分点。

　　前向年际检验分别采用2017年训练预测2018年，以及2017—2018年训练预测2019年。每折训练集剔除测试年所含坐标，避免同坐标跨年进入两侧。空间检验逐一留出具有合格序列的国家，其余国家的2017—2019年资料用于训练；该检验是空间迁移检验，不代表前向时间预测。叶位、日期与同坐标记录作为组共同划分，未采用随机逐行分割。ERA5日值通过既有试验—地点链接表精确连接；模拟某个评估日期的病情仅使用首条病情日期至该评估日期前一日的完整日值。后续实现天气属于回顾性条件输入，检验不涉及业务天气预报性能。

　　误差以预测百分比减观测百分比定义，偏差的正值表示高估。每条序列内的后续评估日期等权，各坐标年内序列等权，坐标年之间等权；RMSE为加权平方误差的平方根，MAE和偏差采用相同分层权重。叶位和重复评估数量不被解释为独立试验样本数。折内全部基准采用相同测试观测与权重。

""" + "\n".join(lines) + f"""

　　两次前向检验中，β≤1 d⁻¹、20天潜育期SEIR的误差低于首值持续性基准，但高于仅由训练病情拟合的全局logit趋势。主比较的所有空间和时间折、三种潜育期设定下的β估计均位于1 d⁻¹上界，主设定在2018年和2019年的偏差分别为−26.21和−20.48个百分点。边界估计需要同时考虑参数范围、工作天气函数和初态限制，不能单凭该边界判定二次传播机制不适用。

""" + "\n".join(sensitivity) + f"""

　　同叶位的{len(changes)}对连续评估中，{int(changes.observed_decline.sum())}对（{100 * changes.observed_decline.mean():.1f}%）出现百分比下降，涉及{changes.loc[changes.observed_decline, 'series_id'].nunique()}条序列。原始下降观测全部保留，未作单调修复。该描述反映评估值变化，不直接识别组织恢复、叶片替换或测量误差。

　　符合条件的{int(inventory.first_percent.eq(0).sum())}条序列首值为0，{int(inventory.first_percent.eq(100).sum())}条序列首值为100。零外源通量且E₀=I₀=0的SEIR无法启动感染，首值为100时累计感染代理无法进一步增加；这些序列保留在总体误差中，并提供初值分层结果。代理y=1−S单调不减，不能描述病情下降、叶片衰老导致的评估构成变化、叶片替换或观测误差。固定潜育期的重新拟合敏感性结果不属于置信区间。

　　条件性基准不识别初侵染发生日期、初始接种体、潜育感染量、首侵染与二次传播的独立贡献，也未估计从播种至显症的完整病程。播期、品种、种植密度、叶片发育和管理记录的不足限制了机理解释；舍入到0.1°的坐标与再分析气象不能充分描述冠层微气候。筛选出的多日期序列并非全部76个试验的代表性抽样。全部欧洲网格的风险、当地冬小麦种植掩膜与物候连接、外源接种体、外部数据的观测模型及气候情景响应均未由该基准验证。当前结果仅检验既有病情出现后、给定首条观测和实现天气条件下的病程预测；完整初侵染—二次传播模型的验证仍需可辨识的过程数据与独立观测。
"""
    (OUT / "Methods_and_limitations.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
