import numpy as np
from timer import Timer
import random


with Timer():
    for _ in range(10000):
        threshold1 = np.random.random(10001)

with Timer():
    for _ in range(10000):
        threshold2 = np.random.uniform(size=10001)

with Timer():
    x = threshold1[random.randint(1, 10001)]

with Timer():
    x = threshold2[random.randint(1, 10001)]