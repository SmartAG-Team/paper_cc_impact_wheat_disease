"""Exact small-group mean resampling distribution for independent verification."""
import math
import numpy as np


def exact_mean_bootstrap(group_means):
    x=np.asarray(group_means,float)
    n=len(x)
    if n<1 or n>10 or not np.isfinite(x).all():
        raise ValueError('One to ten finite group means required for exact enumeration')
    support={}
    def visit(remaining,counts):
        if len(counts)==n-1:
            multiplicities=(*counts,remaining)
            mean=float(np.dot(multiplicities,x)/n)
            probability=math.factorial(n)/math.prod(math.factorial(k) for k in multiplicities)/n**n
            support[mean]=support.get(mean,0.)+probability
        else:
            for count in range(remaining+1):visit(remaining-count,(*counts,count))
    visit(n,())
    values=np.array(sorted(support))
    probabilities=np.array([support[value] for value in values])
    return values,probabilities
