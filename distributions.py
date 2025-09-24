import scipy.stats as sts
import numpy as np

def sample_truncated_normal(mu, sigma, lower_bound=0, upper_bound=np.inf):
    a, b = abs((lower_bound - mu)/sigma), (upper_bound - mu)/sigma
    distribution = sts.truncnorm(a, b)
    return distribution.rvs(size=1)[0] # return a single number from the array