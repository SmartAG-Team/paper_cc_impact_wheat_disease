"""Figure order must survive caption/reference content and repeat citations."""
import pytest

from paper.scripts.figure_placement import cited_figures,place_figures


def test_places_once_after_first_citing_paragraph_and_preserves_references():
    source='## Results\n\nResult (Fig. 1a).\n\nLater Fig. 1 and Figure 2.\n\n## Figure captions\n\nFigure 1. Legend.\n\nFigure 2. Legend.\n\n## References\n\nAuthor (2020). Figure 3 in a title.\n'
    result=place_figures(source,['fig1','fig2'])
    assert 'Result (Fig. 1a).\n\n![Figure 1](figures/fig1.png)\n\nLater' in result
    assert result.count('(figures/fig1.png)')==1
    assert result.split('## References\n\n')[1]==source.split('## References\n\n')[1]
    assert place_figures(result,['fig1','fig2'])==result


def test_main_and_supporting_ids_and_ranges_are_distinct():
    assert cited_figures('Figs. 2–4 and Figure S2.')==['fig2','fig3','fig4','figs2']
    assert cited_figures('Figures S1–S3 and Fig. 10b')==['figs1','figs2','figs3','fig10']
    result=place_figures('## Results\n\nSee Fig. S1, then Fig. 1.\n',['fig1'])
    assert '(figures/figs1.png)' not in result


def test_missing_body_citation_cannot_be_satisfied_by_a_caption():
    with pytest.raises(ValueError,match='lack a body-text citation'):
        place_figures('## Results\n\nNo reference.\n\n## Figure captions\n\nFigure 1. Caption.\n',['fig1'])


def test_old_image_positions_removed_and_invalid_order_detected():
    result=place_figures('## Results\n\n![Old](figures/fig1.png)\n\nFirst Fig. 1.\n',['fig1'])
    assert result.index('First Fig. 1.')<result.index('(figures/fig1.png)')
    with pytest.raises(ValueError,match='differs from numbering'):
        place_figures('## Results\n\nFigure 2 before Figure 1.\n',['fig1','fig2'])


def test_contents_list_does_not_replace_the_first_scientific_citation():
    source='## Contents\n\nFigures S1–S3.\n\n## S1. Methods\n\nObserved errors (Fig. S1).\n\n## S2. Support\n\nRetained cases (Fig. S2).\n\n## S3. Yield\n\nAnnual records (Fig. S3).\n'
    result=place_figures(source,['figs1','figs2','figs3'])
    assert '(figures/' not in result.split('## S1. Methods')[0]
    assert 'Observed errors (Fig. S1).\n\n![Figure S1]' in result
