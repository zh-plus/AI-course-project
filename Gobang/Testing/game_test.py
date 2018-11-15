import random
from time import time
from pprint import pprint
import sys
import copy
import re
from collections import defaultdict

import numpy as np


class Game():
    def __init__(self, color):
        self.color = color
        self.situations = {
            'win5': 100000,  # 连5
            'alive4': 100000,  # 连4
            'double_rush4': 100000,  # 双冲3
            'double_alive3': 100000,  # 双活3
            'rush4': 7000,  # 冲4
            'alive3': 4500,  # 活3
            'sleep3': 1000,  # 眠3
            'special_sleep3': 900,  # 特型眠3
            'fake-alive3': 500,  # 假眠3
            'alive2': 100,  # 活2
            'sleep2': 10,  # 眠2
            'alive1': 10,  # 活1
            'sleep1': 2,  # 眠1
        }

        self.actions_time = 0
        self.terminal_test_time = 0
        self.evaluate_time = 0
        self.evaluate_num = 0

    def _heuristic_test_1(self, board, move, player):  # 9 * 9 test
        mark = 0

        board_size = len(board)

        x, y = move
        up, down = max(0, x - 4), min(board_size, x + 5)
        left, right = max(0, y - 4), min(board_size, y + 5)
        partial_board = board[up: down, left: right]  # patial of the total board

        padded_board = np.zeros((9, 9), int)  # the padded 9 * 9 board

        up, down = max(0, 4 - x), min(9, 4 + (board_size - x))
        left, right = max(0, 4 - y), min(9, 4 + (board_size - y))
        padded_board[up: down, left: right] = partial_board

        lines = list(zip(*[(padded_board[x, x], padded_board[8 - x, x], padded_board[4, x], padded_board[x, 4]) for x in range(9)]))

        # the len_LengthNum[blank] = mark
        len_five = {0: 100, 1: 25, 2: 15}
        len_four = {0: 4, 1: 10, 2: 8}
        len_three = {0: 3, 1: 4, 2: 4}
        len_two = {0: 2, 1: 3, 2: 0}
        len_one = {0: 1, 1: 0, 2: 0}
        len_value = {1: len_one, 2: len_two, 3: len_three, 4: len_four, 5: len_five}

        opponent = 1 if player != 1 else 2
        for line in lines:
            p = line.index(player)
            initial_state = (0, 0)
            blank, length = initial_state
            while p < 9:
                length += 1
                if line[p] == 0:
                    blank += 1

                if blank > 2:
                    blank, length = initial_state
                    continue

                if p == 8 and blank == length:
                    break

                if length == 5:
                    mark += len_value[5][blank]
                    blank, length = initial_state
                elif line[p] == opponent:
                    blank, length = initial_state
                elif p == 8 or line[p + 1] == opponent or (blank == 2 and length != 2 and line[p + 1] == 0):
                    mark += len_value[length][blank]
                    blank, length = initial_state

                p += 1

        return mark

    def _heuristic_test_2(self, board, move, chessman):  # 5 * 5 test
        mark = 0
        # TODO
        return mark

    def heuristic_evaluate(self, state: np.ndarray, move, player):
        next_board = state.copy()
        next_board[move[0]][move[1]] = player

        mark = 0  # the returned value of heuristic definition

        # Tirstly, testing the 4 directions line in 9*9 board
        mark += self._heuristic_test_1(next_board, move, player)

        # Then, testing the 8 grids and 8 groups in 5*5 board
        mark += self._heuristic_test_2(next_board, move, player)

        return mark

    def has_neighboor(self, state, move):
        board_size = len(state)

        x, y = move
        up, down = max(0, x - 3), min(board_size, x + 4)
        left, right = max(0, y - 3), min(board_size, y + 4)
        partial_board = state[up: down, left: right]  # patial of the total board

        return 1 in partial_board or 2 in partial_board

    def actions(self, state, player):  # generate the available move, then use heuristic evaluation to sort
        tic = time()

        valid_move = (state == 0).nonzero()
        valid_move = list(zip(valid_move[0], valid_move[1]))

        if 1 in state or 2 in state:
            valid_move = [move for move in valid_move if self.has_neighboor(state, move)]

        valid_move = sorted(valid_move, key=lambda m: self.heuristic_evaluate(state, m, player), reverse=True)

        self.actions_time += (time() - tic)

        return valid_move

    def result(self, state, player, action):
        next_state = state.copy()
        next_state[action[0]][action[1]] = player
        return next_state

    def terminal_test_great(self, state: np.ndarray, action):
        tic = time()

        to_str = lambda array: ''.join(map(str, array))
        continuity_test = lambda string: '11111' in string or '22222' in string

        x, y = action

        board_size = len(state)

        horizontal = to_str(state[x, max(0, y - 4): min(board_size, y + 5)])
        vertical = to_str(state[max(0, x - 4): min(board_size, x + 5), y])
        lr_diag = to_str(state.diagonal(y - x))
        rl_diag = to_str(state[:, ::-1].diagonal(14 - x - y))
        if continuity_test(horizontal) or continuity_test(vertical) or continuity_test(lr_diag) or continuity_test(rl_diag):
            return True

        self.terminal_test_time += (time() - tic)
        return False

    def re_counter(self, string, pattern):
        return len(re.findall(pattern, string))

    def _check_substr(self, string, *substrings):
        for substring in substrings:
            if substring in string:
                return True
        return False

    def _check_lines(self, record, *lines):
        """
        In this function, 1 --> my chessman, 2 --> opponent
        :param record:
        :param lines:
        :return:
        """
        for line in lines:
            line = '*' + line + '*'  # the wall of chessboard
            exist = lambda *substrings: self._check_substr(line, *substrings)
            if exist('11111'):
                record['win5'] += 1
                continue
            if exist('011110'):
                record['alive4'] += 1
                continue
            if exist('211110', '011112', '11101', '10111', '11011', '*11110', '01111*'):
                record['rush4'] += 1
                continue
            if exist('001110', '011100', '011010', '010110'):
                record['alive3'] += 1
                continue
            if exist('211100', '001112', '211010', '010112', '210110', '011012', '*11100', '00111*', '*11010', '01011*', '*10110', '01101*'):
                record['sleep3'] += 1
                continue
            if exist('11001', '10011', '10101'):
                record['special_sleep3'] += 1
                continue
            if exist('2011102', '*011102', '201110*', '*01110*'):
                record['fake-alive3'] += 1
                continue
            if exist('001100', '011000', '000110', '001010', '010100', '010010'):
                record['alive2'] += 1
                continue
            if exist('211000', '000112', '10001', '2010102', '210100', '001012', '2011002', '2001102', '210010', '010012', '*10010', '01001*',
                     '*11000', '00011*', '*01010*', '201010*', '*010102', '*10100', '00101*', '*011002', '200110*', '201100*', '*001102'):
                record['sleep2'] += 1
                continue
            if exist('010'):
                record['alive1'] += 1
                continue
            if exist('210', '012', '*10', '01*'):
                record['sleep1'] += 1

        return record

    def evaluate(self, state: np.ndarray):
        tic = time()
        situations = self.situations

        to_str = lambda array: ''.join(map(str, array))

        # the opponent state 1 --> 2, 2 --> 1
        opponent_state = state.copy()
        opponent_state[opponent_state == 1] = 3
        opponent_state[opponent_state == 2] = 1
        opponent_state[opponent_state == 3] = 2

        print(state)
        print(opponent_state)

        my_record = defaultdict(int)
        opponent_record = defaultdict(int)
        reverse_state = state[::-1]
        for i in range(len(state)):
            horizontal = to_str(state[i])
            vertical = to_str(state[..., i])
            lr_diag = to_str(state.diagonal(i))
            rl_diag = to_str(reverse_state.diagonal(-i))

            my_record = self._check_lines(my_record, horizontal, vertical, lr_diag, rl_diag)

            horizontal = to_str(opponent_state[i])
            vertical = to_str(opponent_state[..., i])
            lr_diag = to_str(opponent_state.diagonal(i))
            rl_diag = to_str(opponent_state.diagonal(-i))
            opponent_record = self._check_lines(opponent_record, horizontal, vertical, lr_diag, rl_diag)

        my_mark = sum(map(lambda situ: situations[situ] * my_record[situ], my_record.keys()))
        opponent_mark = sum(map(lambda situ: situations[situ] * opponent_record[situ], opponent_record.keys()))

        self.evaluate_time += (time() - tic)
        self.evaluate_num += 1

        return (my_mark - opponent_mark) if self.color == 1 else (opponent_mark - my_mark)


if __name__ == '__main__':
    color = 1
    game = Game(color)

    chessboard = np.zeros((15, 15), dtype=np.int)
    chessboard[0, 0:2] = 2
    chessboard[0:2, 15 - 1] = 2
    chessboard[1, 6:8] = 1
    chessboard[2:4, 8] = 1

    chessboard[4, 10] = 2
    chessboard[4, 8] = 1

    mark = game.evaluate(chessboard)
    print(mark)
