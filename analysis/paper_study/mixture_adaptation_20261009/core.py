"""Small, explicit estimators for empirical management comparisons.

No disease-to-yield conversion or climate adaptation model is implemented.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 20261009


def yield_t_ha(value, unit):
    factors = {"g_m2": 0.01, "dt_ha": 0.1}
    if unit not in factors:
        raise ValueError(f"Unsupported yield unit: {unit}")
    return value * factors[unit]


def complete_mean(values, expected=4):
    a = np.asarray(values, dtype=float)
    return float(a.mean()) if len(a) == expected and np.isfinite(a).all() else np.nan


def kernel_weight(grain_mass, tkw):
    """Equal-area samples: mass / inferred kernel number, in source TKW units."""
    y, k = np.asarray(grain_mass, float), np.asarray(tkw, float)
    if len(y) != len(k) or not len(y) or not np.isfinite(y).all() or not np.isfinite(k).all() or (k <= 0).any() or (y < 0).any() or y.sum() <= 0:
        return np.nan
    return float(y.sum() / (y / k).sum())


def yield_contrast(mixture, controls):
    a = np.asarray(controls, dtype=float)
    if not len(a) or not np.isfinite(a).all() or not np.isfinite(mixture) or a.mean() <= 0:
        raise ValueError("All constituent controls and a positive reference are required")
    baseline = float(a.mean())
    return {"baseline": baseline, "delta": float(mixture - baseline),
            "relative_percent": float(100 * (mixture - baseline) / baseline),
            "hindsight_best_delta": float(mixture - a.max())}


def environment_means(data, value):
    by_composition = data.groupby(["environment_id", "composition_id"], sort=True)[value].mean()
    return by_composition.groupby("environment_id").mean().rename(value).reset_index()


def dependency_groups(data):
    adjacency = {}
    for a, b in zip(data.constituent_1, data.constituent_2):
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    assigned = {}
    for node in sorted(adjacency):
        if node in assigned:
            continue
        label, todo = f"component_{len(set(assigned.values())) + 1:02}", [node]
        while todo:
            current = todo.pop()
            if current not in assigned:
                assigned[current] = label
                todo.extend(sorted(adjacency[current] - assigned.keys()))
    return np.array([assigned[x] for x in data.constituent_1])


def site_bootstrap(environments, value, draws=10000, seed=SEED):
    """Cluster sites, then give each environment occurrence equal weight."""
    groups = [g[value].dropna().to_numpy(float) for _, g in environments.groupby("site_id", sort=True)]
    groups = [x for x in groups if len(x)]
    if len(groups) < 2:
        return (np.nan, np.nan)
    totals, counts = np.array([x.sum() for x in groups]), np.array([len(x) for x in groups])
    idx = np.random.default_rng(seed).integers(0, len(groups), size=(draws, len(groups)))
    draws_mean = totals[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return tuple(float(x) for x in np.quantile(draws_mean, [0.025, 0.975]))


def transfer_predictions(environments, value):
    """Fixed training-mean model; no held-out data used in fitting or selection."""
    out = []
    for site in sorted(environments.site_id.unique()):
        train = environments[environments.site_id.ne(site)]
        test = environments[environments.site_id.eq(site)]
        if len(train):
            for _, r in test.iterrows():
                out.append({**r.to_dict(), "scheme": "leave_site_out", "prediction": train[value].mean(),
                            "training_environments": "|".join(sorted(train.environment_id)), "zero_reference": 0.})
    for year in sorted(environments.year.unique())[1:]:
        train = environments[environments.year.lt(year)]
        for _, r in environments[environments.year.eq(year)].iterrows():
            out.append({**r.to_dict(), "scheme": "forward_year", "prediction": train[value].mean(),
                        "training_environments": "|".join(sorted(train.environment_id)), "zero_reference": 0.})
    return pd.DataFrame(out)


def map_release(current, previous):
    """Map unique current payloads to unique older payloads, preserving aliases.

    Identical anonymous rows are conservatively one record, not extra replication.
    Nonidentical older records sharing the current payload are ambiguous.
    """
    current, previous = current.copy(), previous.copy()
    common = [c for c in current.columns if c in previous.columns]
    current["_current_line"] = np.arange(2, len(current) + 2)
    previous["_previous_line"] = np.arange(2, len(previous) + 2)
    now = current.groupby(common, dropna=False, sort=False, as_index=False).agg(
        current_source_lines=("_current_line", lambda x: "|".join(map(str, x))))
    old_cols = [c for c in previous.columns if c != "_previous_line"]
    old = previous.groupby(old_cols, dropna=False, sort=False, as_index=False).agg(
        previous_source_lines=("_previous_line", lambda x: "|".join(map(str, x))))
    now["_payload_id"] = np.arange(len(now))
    merged = now.merge(old, on=common, how="left", indicator=True, validate="one_to_many")
    counts = merged.groupby("_payload_id").size()
    merged["mapping_status"] = np.where(merged._merge.eq("left_only"), "unmatched",
                                         np.where(merged._payload_id.map(counts).eq(1), "unique", "ambiguous"))
    audit = merged[["_payload_id", "current_source_lines", "previous_source_lines", "mapping_status"]].copy()
    return merged[merged.mapping_status.eq("unique")].drop(columns=["_merge", "_payload_id"]), audit


def control_baseline(controls, environment, group, constituents, value):
    selected = controls[controls.environment_id.eq(environment) & controls.control_group.eq(group)]
    values, ids, n = [], [], []
    for cultivar in constituents:
        d = selected[selected.cultivar.eq(cultivar) & selected[value].notna()]
        if not len(d):
            return None
        values.append(float(d[value].mean()))
        ids.append("|".join(sorted(d.plot_id.astype(str))))
        n.append(len(d))
    return {"values": values, "control_ids": ids, "control_n": n}


def applicability():
    common = {"validated_climate_adaptation": False, "stb_mediated_grain_loss": False,
              "european_production_benefit": False, "independent_environment_validation": False,
              "validated_commercial_adoption": False, "measured_economic_return": False,
              "annual_production_stability": False, "disease_yield_mediation_identified": False}
    return {
        "schema_version": "1.0",
        "france": {**common, "randomized_within_trial_allocation": True,
                   "causal_constituent_contrast": True,
                   "causal_scope": "randomized_within_trial_complete_cases_only; composition_unreplicated; interference_and_missingness_assumptions_required; no_claim_of_statistical_significance",
                   "descriptive_association": True, "selection_on_observed_yields": "hindsight_best_constituent_only",
                   "independent_combination_replication": False, "environment_count": 1,
                   "population": "Mauguio 2018 durum inbred lines; alternating rows; post-assessment fungicide",
                   "uncertainty_status": "No defensible independent-environment or shared-control cluster interval; influence ranges only",
                   "allowed_claim": "Within-trial observed yield difference against matching constituent pure stands, conditional on observed complete harvests",
                   "measured_tradeoffs": ["raw yield downside", "thousand-kernel weight as yield component"]},
        "swiss": {**common, "randomized_within_trial_allocation": False,
                  "mixture_only_randomization_documented": True, "pure_check_corandomization_documented": False,
                  "causal_constituent_contrast": False, "descriptive_association": True,
                  "selection_on_observed_yields": "hindsight_best_constituent_only",
                  "source_control_exchangeability": False,
                  "control_group": "year x postcode x ww; ww management meaning unresolved",
                  "allowed_claim": "Noncausal, same-environment matched-source diagnostic, separated by source trial code",
                  "uncertainty_status": "site-cluster bootstrap stability only; nonexchangeability remains",
                  "measured_tradeoffs": ["observed yield downside", "protein percent dry matter", "specific weight kg/hl"]},
        "claim_classes": {
            "descriptive_association": {"france": "supported_within_observed_sample", "swiss": "supported_within_matched_source_groups"},
            "randomized_within_trial_effect": {"france": "conditional_on_complete_case_and_interference_assumptions; no_independent_combination_replication", "swiss": "not_supported_for_external_constituent_controls"},
            "selection_on_observed_yields": {"france": "hindsight_diagnostic_only", "swiss": "hindsight_diagnostic_only"},
            "climate_adaptation": {"france": "not_validated", "swiss": "not_validated"},
            "disease_mediated_grain_loss": {"france": "not_identified", "swiss": "not_identified"}}
    }
