import numpy as np

vertices = 5
arr = np.full((vertices, vertices), np.inf)
np.fill_diagonal(arr, 0)

print(arr)