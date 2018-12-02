import numpy as np
from timer import Timer

with Timer():
    x = np.zeros(100000)

with Timer():
    x = [0 for _ in range(100000)]