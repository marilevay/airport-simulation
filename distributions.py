import scipy.stats as sts
import numpy as np

def sample_truncated_normal(mu, sigma, lower_bound=0, upper_bound=np.inf):
    a = (lower_bound - mu) / sigma
    b = (upper_bound - mu) / sigma
    return float(sts.truncnorm.rvs(a, b, loc=mu, scale=sigma, size=1))