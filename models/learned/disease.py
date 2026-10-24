"""Frozen rolling monitoring with validated field and earlier-day histories."""
import joblib

from .integrity import DATA, verify
from .disease_math import design, predict_queries as checked_predict_queries


def load_model():
    verify("disease")
    return joblib.load(DATA / "current/disease.joblib")


def predict_queries(weather_crop_features, queries, observed_history, *, fitted=None, strength=None):
    """Return severity percentages and contributing earlier observation identities."""
    verify("disease")
    if fitted is None:
        fitted = load_model()
    return checked_predict_queries(fitted, weather_crop_features, queries, observed_history, strength=strength)
