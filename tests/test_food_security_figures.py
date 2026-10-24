from pathlib import Path

import numpy as np
import pandas as pd

from paper.analyses.food_security.figures import render


def test_unavailable_benchmark_scores_export_visible_unavailable_state(tmp_path):
    source = tmp_path / 'benchmark'
    source.mkdir()
    rows = [dict(source='pooled', test_year=0, endpoint=endpoint, model=model,
                 n_required=2, n_available=0, status='no_paired_support', rmse=np.nan)
            for model in ['training_leaf_mean', 'calendar_ridge', 'weather_calendar_rf']
            for endpoint in ['assessment', 'window_integral']]
    pd.DataFrame(rows).to_csv(source / 'summary.csv', index=False)
    result = render(tmp_path)
    assert 'autonomous_disease_benchmark.svg' in result['files']
    svg = (tmp_path / 'figures/autonomous_disease_benchmark.svg').read_text()
    assert 'Unavailable' in svg
