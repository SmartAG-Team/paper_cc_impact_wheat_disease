def test_acquisition_uses_query_dates_and_covers_cross_year_history():
    from models.monitoring.gather_regional_canopy import acquisition_months
    months = acquisition_months(['2014-03-01','2014-04-02'])
    assert months[0] == ('2013-11-01','2013-12-01')
    assert months[-1] == ('2014-04-01','2014-05-01')
    assert len(months) == 6
