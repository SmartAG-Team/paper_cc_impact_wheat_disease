"""The onset map must distinguish ordinary timing changes from rare extremes."""
import numpy as np
from matplotlib.colors import Normalize
from . import figures


def test_usual_onset_advances_have_distinct_colours_with_a_rare_extreme():
    values=np.array([-95.1555556,-40.,-30.,-20.,.1444444,np.nan])
    scale=getattr(figures,'onset_change_scale',lambda x:figures.discrete_scale(x,[.5,1,2,5,10,20],2))
    norm,cmap,ticks=scale(values)
    assert isinstance(norm,Normalize)
    colours=cmap(norm(np.array([-40.,-30.,-25.])))
    assert len(np.unique(colours,axis=0))==3
    assert np.isclose(norm(0),.5)
    assert norm.vmin<=np.nanmin(values) and norm.vmax>=np.nanmax(values)
    assert list(ticks)==[-100.,-50.,0.,50.,100.]
