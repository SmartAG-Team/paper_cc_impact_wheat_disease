"""Public repository links must survive manuscript typography formatting."""
from analysis.paper_study.nature_food_submission_20261008.build import polish


def test_repository_url_survives_text_polishing():
    url = 'https://github.com/SmartAG-Team/paper_cc_impact_wheat_disease'
    paragraph = f'Data and code are available at {url}. SSP585 results are included.'
    assert polish(paragraph) == paragraph
