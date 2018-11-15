import sys
from pprint import pprint

import numpy as np

move = (0, 2)
board_size = 15
board = np.random.choice((0, 1, -1), (15, 15), p=(0.8, 0.1, 0.1))   # type: np.ndarray

x, y = move
up, down = max(0, x - 4), min(board_size, x + 5)
left, right = max(0, y - 4), min(board_size, y + 5)
partial_board = board[up: down, left: right]  # patial of the total board

padded_board = np.zeros((9, 9), int)  # the padded 9 * 9 board

up, down = max(0, 4 - x), min(9, 4 + (board_size - x))
left, right = max(0, 4 - y), min(9, 4 + (board_size - y))
padded_board[up: down, left: right] = partial_board

testing_lines = list(zip(*[(padded_board[x, x], padded_board[8 - x, x], padded_board[4, x], padded_board[x, 4]) for x in range(9)]))

pprint(board)
pprint(partial_board)
pprint(padded_board)
pprint(testing_lines)
