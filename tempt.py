import time
import numpy as np
import random, math
import os
import sys
import scipy.stats as stats
import matplotlib.pyplot as plt

mu = 2
lower = 1.8
upper = 2.2
sigma = 0.1
rwdRng = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

plt.hist(rwdRng.rvs(10000))
plt.show()
