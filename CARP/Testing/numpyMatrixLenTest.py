import numpy as np
from time import time

arr = np.array([[1, 2, 3],
                [2, 3, 4],
                [3, 4, 5]])
tic = time()
for x in range(10000000):
    test = len(arr)
print(time() - tic, 's')

tic = time()
for x in range(10000000):
    test = arr.shape
print(time() - tic, 's')