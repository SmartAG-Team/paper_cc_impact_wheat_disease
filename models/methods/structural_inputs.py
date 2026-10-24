"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study')
import numpy as np
import pandas as pd
from analysis.methods.empirical_helpers import attach_maxima
from model.seasonal_septoria.regional import tpv_accumulation

def development_inputs(data, weather, phenology):
    attach_maxima(data, weather)
    accumulated = np.zeros_like(data.temperature)
    for meta in data.metadata.itertuples():
        field, days = (int(meta.field_index), int(meta.forcing_days))
        dates = pd.date_range(meta.sowing_date, periods=days)
        accumulated[field, :days] = tpv_accumulation(data.temperature[field:field + 1, :days], data.maximum_temperature[field:field + 1, :days], [meta.latitude], dates.dayofyear, np.ones((1, days), bool), phenology)[0]
        accumulated[field, days:] = accumulated[field, days - 1]
    thresholds = {row['BBCH']: row['Cumulative_t_pp_v_GDD'] for row in phenology['thresholds']}
    return (accumulated, thresholds)
