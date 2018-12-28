import numpy as np
from timer import Timer

a = np.random.randn(1300, 10) * 10
b = np.random.randn(1300, 10).T * 10

with Timer():
    for _ in range(1000):
        c = a @ b
print(c)


with Timer():
    for _ in range(1000):
        c = np.dot(a, b)
print(c)