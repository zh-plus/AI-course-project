from timer import Timer
import numpy as np
import random

rand_happen = lambda p: True if random.random() <= p else False


def rand_happen1(p):
    return True if random.random() <= p else False


with Timer():
    for _ in range(1000000):
        rand_happen(0.8)

with Timer():
    for _ in range(1000000):
        rand_happen1(0.8)
