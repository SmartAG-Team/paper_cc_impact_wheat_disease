"""Separate statistical prototype; no crop or pathogen process model imports."""
import numpy as np
import pandas as pd
from scipy.optimize import nnls


def endpoint_eligibility(frame, require_positive_untreated_stb=False):
    nontargets = ["yellow_rust", "brown_rust", "mildew", "leaf_spot"]
    pair = frame[["stb", "stb_treated"]].notna().all(axis=1)
    measured_zero = frame[[c for d in nontargets for c in [d, d + "_treated"]]].eq(0).all(axis=1)
    mask = pair & measured_zero & frame.no_positive_nontarget_at_any_recorded_date
    if require_positive_untreated_stb:
        mask &= frame.stb > 0
    return mask


def trial_weights(frame):
    if frame.empty or frame["record"].isna().any():
        raise ValueError("Nonempty identified trials are required")
    return 1.0 / frame.groupby("record")["record"].transform("size").to_numpy(float)


def signed_response(untreated_yield, control_yield):
    u, c = np.asarray(untreated_yield, float), np.asarray(control_yield, float)
    if not np.isfinite(u).all() or not np.isfinite(c).all() or (c <= 0).any():
        raise ValueError("Finite yields and positive disease-control yield are required")
    return 1.0 - u / c


def fit_origin(frame, features):
    if not features or "R" in features:
        raise ValueError("At least one disease feature, excluding the response, is required")
    x = frame[features].to_numpy(float)
    y = frame["R"].to_numpy(float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("All predictors and responses must be finite; missing is not zero")
    w = trial_weights(frame)
    weighted_x = x * np.sqrt(w)[:, None]
    weighted_y = y * np.sqrt(w)
    nonnull = np.any(x != 0, axis=0)
    beta = np.zeros(len(features), dtype=float)
    if len(features) == 1:
        denominator = float(np.dot(w * x[:, 0], x[:, 0]))
        if denominator > 0:
            beta[0] = max(0.0, float(np.dot(w * x[:, 0], y)) / denominator)
    elif nonnull.any():
        beta[nonnull] = nnls(weighted_x[:, nonnull], weighted_y)[0]
    rank = int(np.linalg.matrix_rank(weighted_x))
    return {
        "features": list(features), "coefficients": beta.tolist(),
        "baseline": float(np.dot(w, y) / w.sum()),
        "training_trials": sorted(frame["record"].unique().tolist()),
        "training_rows": len(frame), "training_trial_count": frame["record"].nunique(),
        "null_predictors": [f for f, active in zip(features, nonnull) if not active],
        "negative_feature_values": {f: int((x[:, j] < 0).sum()) for j, f in enumerate(features)},
        "negative_responses": int((y < 0).sum()), "weighted_design_rank": rank,
        "rank_deficient": rank < len(features),
        "nonzero_predictor_count": int(nonnull.sum()),
        "origin_constrained": True, "ridge_penalty": 0,
    }


def predict(frame, features, fit):
    if list(features) != fit["features"]:
        raise ValueError("Predictor order must match the training fit")
    x = frame[features].to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError("Predictions require finite measured disease differences")
    return x @ np.asarray(fit["coefficients"])


def fit_damage(frame, features):
    u = frame.untreated_yield_native.to_numpy(float)
    c = frame.treated_yield_native.to_numpy(float)
    if not np.isfinite(u).all() or not np.isfinite(c).all() or (u <= 0).any() or (c <= 0).any():
        raise ValueError("Log yield-ratio fitting requires finite positive yields in both arms")
    transformed = frame.copy()
    transformed["R"] = np.log(c/u)
    fit = fit_origin(transformed, features)
    weights = trial_weights(frame)
    fit["baseline"] = float(np.dot(weights, frame.R) / weights.sum())
    fit["family"] = "damage"
    fit["fitted_response"] = "log(Y_C/Y_U); equivalent residual objective to log(Y_U/Y_C)=-X*beta"
    return fit


def predict_damage(frame, features, fit):
    return -np.expm1(-predict(frame, features, fit))


def evaluate_split(train, test, features, fold, family="linear"):
    if set(train.record) & set(test.record):
        raise ValueError("A trial cannot appear in both training and test")
    if family not in {"linear", "damage"}:
        raise ValueError("Unknown response family")
    fit = fit_origin(train, features) if family == "linear" else fit_damage(train, features)
    out = test.copy()
    out["fold"] = fold
    out["prediction"] = predict(test, features, fit) if family == "linear" else predict_damage(test, features, fit)
    out["linear_score"] = predict(test, features, fit)
    out["baseline_prediction"] = fit["baseline"]
    out["evaluation_weight"] = trial_weights(test)
    out["negative_response"] = out.R < 0
    out["negative_any_feature"] = (out[features] < 0).any(axis=1)
    out["prediction_above_one"] = out.prediction > 1
    out["prediction_at_numerical_upper_bound"] = out.prediction == 1
    memberships = []
    for role, subset in [("train", train), ("test", test)]:
        cols = [c for c in ["row_id", "record", "control_id", "cultivar", "year"] if c in subset]
        member = subset[cols].copy()
        member["fold"], member["role"] = fold, role
        member["weight"] = trial_weights(subset)
        memberships.append(member)
    return out, fit, pd.concat(memberships, ignore_index=True)


def leave_one_trial_out(frame, features, family="linear"):
    if frame.record.nunique() < 2 or frame.row_id.duplicated().any():
        raise ValueError("Unique rows from at least two trials are required")
    predictions, fits, memberships = [], [], []
    for trial in sorted(frame.record.unique()):
        train, test = frame[frame.record != trial], frame[frame.record == trial]
        out, fit, member = evaluate_split(train, test, features, trial, family=family)
        fit["held_out_trial"] = trial
        predictions.append(out)
        fits.append(fit)
        memberships.append(member)
    return pd.concat(predictions, ignore_index=True), fits, pd.concat(memberships, ignore_index=True)


def paired_integral(untreated, treated):
    if untreated.date.duplicated().any() or treated.date.duplicated().any():
        raise ValueError("Duplicate assessment dates need source resolution")
    knots = untreated[["date", "stb"]].merge(treated[["date", "stb"]], on="date", suffixes=("_untreated", "_treated"), validate="one_to_one")
    knots = knots.dropna(subset=["stb_untreated", "stb_treated"]).sort_values("date").reset_index(drop=True)
    if len(knots) < 2:
        raise ValueError("At least two genuinely paired numerical dates are required")
    days = (pd.to_datetime(knots.date) - pd.to_datetime(knots.date.iloc[0])).dt.days.to_numpy(float)
    if not np.isfinite(knots[["stb_untreated", "stb_treated"]].to_numpy(float)).all():
        raise ValueError("Paired measured values must be finite")
    auc_u = float(np.trapezoid(knots.stb_untreated, days))
    auc_c = float(np.trapezoid(knots.stb_treated, days))
    return {"paired_points": len(knots), "window_days": int(days[-1]),
            "window_start": knots.date.iloc[0], "window_end": knots.date.iloc[-1],
            "auc_untreated_percentage_days": auc_u,
            "auc_treated_percentage_days": auc_c,
            "delta_stb_percentage_days": auc_u - auc_c}, knots


def arithmetic_metrics(predictions):
    w = trial_weights(predictions)
    y = predictions.R.to_numpy(float)
    p = predictions.prediction.to_numpy(float)
    b = predictions.baseline_prediction.to_numpy(float)
    mse = float(np.dot(w, (p - y) ** 2) / w.sum())
    bmse = float(np.dot(w, (b - y) ** 2) / w.sum())
    return {"rows": len(predictions), "trials": predictions.record.nunique(),
            "model_rmse": np.sqrt(mse), "model_mae": float(np.dot(w, abs(p-y)) / w.sum()),
            "model_bias": float(np.dot(w, p-y) / w.sum()),
            "baseline_rmse": np.sqrt(bmse), "baseline_mae": float(np.dot(w, abs(b-y)) / w.sum()),
            "baseline_bias": float(np.dot(w, b-y) / w.sum()),
            "mse_skill_against_calibration_mean": 1 - mse/bmse if bmse else None,
            "negative_observed": int((y < 0).sum()), "negative_predicted": int((p < 0).sum()),
            "prediction_above_one": int((p > 1).sum())}
