import numpy as np
from numpy.core.multiarray import ndarray

this_state: ndarray = np.array([
    [-1, 1, 0, 1],
    [1, 0, 1, 2],
    [1, -1, 1, 4],
    [5, 6, 7, 8]
])

move = 1, 1
x, y = move
board_size = 4
player = -1

this_state = this_state[max(0, x - 1): min(board_size, x + 2), max(0, y - 1): min(board_size, y + 2)]
player_num = np.sum(this_state == player)

print(this_state)
print(player_num)