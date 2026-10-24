import numpy as np
import torch
from test_dynamic_data import fixture


def test_weather_sequence_uses_only_preceding_days_and_supports_full_range():
    from models.monitoring.dynamic_data import build_sequences
    from models.monitoring.weather_sequence_model import WeatherSequenceModel
    w,f,o=fixture();b=build_sequences(w,f,o,o.drop(columns='observed_percent'))
    torch.manual_seed(42);m=WeatherSequenceModel(3);m.eval()
    x=torch.zeros(3,3);base=torch.full((3,),.3)
    a=m(b,x,base).detach().numpy()
    b['weather'][0,2:]=100
    c=m(b,x,base).detach().numpy()
    np.testing.assert_array_equal(a[:2],c[:2])
    assert abs(a[2]-c[2])>1e-5
    with torch.no_grad():
        for p in m.head.parameters():p.zero_()
        m.head[-1].bias.fill_(3.)
    value=m(b,x,base).detach().numpy()
    assert (value>90).all() and (value<=100).all()
