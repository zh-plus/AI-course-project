from timer import Timer
import numpy as np
import random


def avg(l):
    return sum(l) / len(l)


l = random.sample(range(1, 500000), 1000)

with Timer():
    x = np.average(l)

print(x)

with Timer():
    x = avg(l)

print(x)