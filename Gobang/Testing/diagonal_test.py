import numpy as np


state: np.ndarray = np.array([
    [ 1, 2, 1,  1],
    [-1, 2, 2,  1],
    [ 3, 0, 3,  3],
    [-1, 4, 1, -4]
])

print(state[::-1])