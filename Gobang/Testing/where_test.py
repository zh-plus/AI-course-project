import numpy as np

state: np.ndarray = np.array([
    [1, 1, -1, 1],
    [-1, -1, 0, 1],
    [1, 0, 1, 1],
    [-1, 1, 1, -1]
])


# state[state == 1], state[state == -1] = -1, 1
print(state)
