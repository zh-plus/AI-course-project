import random
from collections import defaultdict

import numpy as np

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)
infinity = np.inf


class Game():
    def __init__(self, color):
        self.color = color
        self.situations = {
            'win5': 100000,  # 连5
            'alive4': 10000,  # 连4
            'double_rush4': 10000,  # 双冲3
            'double_alive3': 10000,  # 双活3
            'continue-rush4': 900,  # 冲4
            'jump-rush4': 800,
            'alive3': 600,  # 活3
            'continue-sleep3': 100,  # 眠3
            'jump-sleep3': 75,  # 眠3
            'special_sleep3': 60,  # 特型眠3
            'fake-alive3': 60,  # 假眠3
            'alive2': 50,  # 活2
            'sleep2': 10,  # 眠2
            'alive1': 10,  # 活1
            'sleep1': 2,  # 眠1
        }

        # the len_LengthNum[blank] = mark
        len_five = {0: 100, 1: 25, 2: 15}
        len_four = {0: 4, 1: 10, 2: 8}
        len_three = {0: 3, 1: 4, 2: 4}
        len_two = {0: 2, 1: 3, 2: 0}
        len_one = {0: 1, 1: 0, 2: 0}
        self.len_value = {1: len_one, 2: len_two, 3: len_three, 4: len_four, 5: len_five}

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

        opponent = 1 if player != 1 else 2
        for line in lines:
            if not player in line:
                continue

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
                    mark += self.len_value[5][blank]
                    blank, length = initial_state
                elif line[p] == opponent:
                    blank, length = initial_state
                elif p == 8 or line[p + 1] == opponent or (blank == 2 and length != 2 and line[p + 1] == 0):
                    mark += self.len_value[length][blank]
                    blank, length = initial_state

                p += 1

        return mark

    def _heuristic_test_2(self, state, move, player):  # 5 * 5 test
        state[move[0]][move[1]] = player

        # if move == (9, 3):
        #     print(state)

        mark = 0
        situations = self.situations

        if player == 2:
            state = state.copy()
            state[state == 1] = 3
            state[state == 2] = 1
            state[state == 3] = 2

        to_str = lambda array: ''.join(map(str, array))

        reverse_state = state[::-1]

        x, y = move
        board_size = len(state)
        up, down = max(0, x - 5), min(board_size, x + 6)
        left, right = max(0, y - 5), min(board_size, y + 6)

        horizontal = to_str(state[x, left: right])
        vertical = to_str(state[up: down, y])

        lr_diag = to_str(state.diagonal(y - x))
        rl_diag = to_str(reverse_state.diagonal(x + y - 14))

        my_record = defaultdict(int)
        my_record = self._check_lines(my_record, horizontal, vertical, lr_diag, rl_diag)

        my_mark = sum(map(lambda situ: situations[situ] * my_record[situ], my_record.keys()))

        state[move[0]][move[1]] = 0

        return my_mark

    def _heuristic_test_3(self, state, move, player):  # 5 * 5 test
        state[move[0]][move[1]] = player

        # if move == (10, 5):
        #     print(state)

        mark = 0
        situations = self.situations

        if player == 2:
            state = state.copy()
            state[state == 1] = 3
            state[state == 2] = 1
            state[state == 3] = 2

        to_str = lambda array: ''.join(map(str, array))

        reverse_state = state[::-1]

        x, y = move
        board_size = len(state)
        up, down = max(0, x - 5), min(board_size, x + 6)
        left, right = max(0, y - 5), min(board_size, y + 6)

        horizontal = to_str(state[x, left: right])
        vertical = to_str(state[up: down, y])

        lr_diag = to_str(state.diagonal(y - x))
        rl_diag = to_str(reverse_state.diagonal(x + y - 4))

        my_record = defaultdict(int)
        my_record = self._check_lines(my_record, horizontal, vertical, lr_diag, rl_diag)

        my_mark = sum(map(lambda situ: situations[situ] * my_record[situ], my_record.keys()))

        state[move[0]][move[1]] = 0

        return my_mark

    def heuristic_evaluate(self, state: np.ndarray, move, player):
        mark = 0  # the returned value of heuristic definition

        # if move == (9, 3):
        #     print('here')
        #     print(state)

        next_board = state.copy()

        # if move == (10, 5):
        #     print(next_board)

        # Tirstly, testing the 4 directions line in 9*9 board
        opponent = 2 if player == 1 else 1
        mark += self._heuristic_test_2(next_board, move, player) + self._heuristic_test_2(next_board, move, opponent)

        # Then, testing the 8 grids and 8 groups in 5*5 board
        # mark += self._heuristic_test_2(next_board, move, player)

        return mark

    def has_neighboor(self, state, move):
        board_size = len(state)

        x, y = move
        up, down = max(0, x - 2), min(board_size, x + 3)
        left, right = max(0, y - 2), min(board_size, y + 3)
        partial_board = state[up: down, left: right]  # patial of the total board

        return 1 in partial_board or 2 in partial_board

    def actions(self, state, player):  # generate the available move, then use heuristic evaluation to sort

        valid_move = (state == 0).nonzero()
        valid_move = list(zip(valid_move[0], valid_move[1]))

        center = len(state) // 2

        if 1 not in state and 2 not in state:
            return [(random.randint(center - 3, center + 4), random.randint(center - 3, center + 4))]

        valid_move = [move for move in valid_move if self.has_neighboor(state, move)]

        valid_move = sorted(valid_move, key=lambda m: self.heuristic_evaluate(state, m, player), reverse=True)

        # print((8, 12) in valid_move)

        return valid_move[:max(20, (len(valid_move) // 2) + 1)]

    def result(self, state, player, action):
        next_state = state.copy()
        next_state[action[0]][action[1]] = player
        return next_state

    def terminal_test_great(self, state: np.ndarray, action):
        continuity_test = lambda string: '11111' in string or '22222' in string

        x, y = action

        board_size = len(state)

        horizontal = ''.join(map(str, state[x, max(0, y - 4): min(board_size, y + 5)]))
        vertical = ''.join(map(str, state[max(0, x - 4): min(board_size, x + 5), y]))
        lr_diag = ''.join(map(str, state.diagonal(y - x)))
        rl_diag = ''.join(map(str, state[:, ::-1].diagonal(14 - x - y)))
        if continuity_test(horizontal) or continuity_test(vertical) or continuity_test(lr_diag) or continuity_test(rl_diag):
            return True

        return False

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

            num = line.count('1')
            length = len(line)
            if num >= 3 and length >= 7:
                if exist('11111'):
                    record['win5'] += 1
                    continue
                if exist('011110'):
                    record['alive4'] += 1
                    continue
                if exist('211110', '011112', '*11110', '01111*'):
                    record['continue-rush4'] += 1
                    continue
                if exist('11101', '10111', '11011'):
                    record['jump-rush4'] += 1
                if exist('001110', '011100', '011010', '010110'):
                    record['alive3'] += 1
                    continue
                if exist('211100', '001112', '*11100', '00111*'):
                    record['continue-sleep3'] += 1
                    continue
                if exist('211010', '010112', '210110', '011012', '*11010', '01011*', '*10110', '01101*'):
                    record['jump-sleep3'] += 1
                    continue
                if exist('11001', '10011', '10101'):
                    record['special_sleep3'] += 1
                    continue
                if exist('2011102', '*011102', '201110*', '*01110*'):
                    record['fake-alive3'] += 1
                    continue
            elif num >= 2 and length >= 7:
                if exist('001100', '011000', '000110', '001010', '010100', '010010'):
                    record['alive2'] += 1
                    continue
                if exist('211000', '000112', '10001', '2010102', '210100', '001012', '2011002', '2001102', '210010', '010012', '*10010', '01001*',
                         '*11000', '00011*', '*01010*', '201010*', '*010102', '*10100', '00101*', '*011002', '200110*', '201100*', '*001102'):
                    record['sleep2'] += 1
                    continue
            else:
                if exist('010'):
                    record['alive1'] += 1
                    continue
                if exist('210', '012', '*10', '01*'):
                    record['sleep1'] += 1

        return record

    def evaluate(self, state: np.ndarray):
        situations = self.situations

        to_str = lambda array: ''.join(map(str, array))
        size = len(state)

        # the opponent state 1 --> 2, 2 --> 1
        opponent_state = state.copy()
        opponent_state[opponent_state == 1] = 3
        opponent_state[opponent_state == 2] = 1
        opponent_state[opponent_state == 3] = 2
        opponent_reverse_state = opponent_state[::-1]

        my_record = defaultdict(int)
        opponent_record = defaultdict(int)
        reverse_state = state[::-1]

        my_horizontal, my_vertical = [to_str(x) for x in state], [to_str(state[..., x]) for x in range(size)]
        my_lr_diag, my_rl_diag = [], []

        op_horizontal, op_vertical = [to_str(x) for x in opponent_state], [to_str(opponent_state[..., x]) for x in range(size)]
        op_lr_diag, op_rl_diag = [], []
        for i in range(-size, size):
            my_lr_diag.append(to_str(state.diagonal(i)))
            my_rl_diag.append(to_str(reverse_state.diagonal(i)))

            op_lr_diag.append(to_str(opponent_state.diagonal(i)))
            op_rl_diag.append(to_str(opponent_reverse_state.diagonal(i)))

        my_record = self._check_lines(my_record, *my_horizontal, *my_vertical, *my_lr_diag, *my_rl_diag)
        opponent_record = self._check_lines(opponent_record, *op_horizontal, *op_vertical, *op_lr_diag, *op_rl_diag)

        my_mark = sum(map(lambda situ: situations[situ] * my_record[situ], my_record.keys()))
        opponent_mark = sum(map(lambda situ: situations[situ] * opponent_record[situ], opponent_record.keys()))

        return (1.01 * my_mark - opponent_mark) if self.color == 1 else (1.01 * opponent_mark - my_mark)


def alphabeta_search(state, game: Game, player, d=4, cutoff_test=None, eval_fn=None):
    def max_value(state, player, last_move, alpha, beta, depth):
        if cutoff_test(state, last_move, depth):
            value = eval_fn(state)
            return value
        v = -infinity
        opponent = 1 if player != 1 else 2
        actions = game.actions(state, player)
        for a in actions:
            v = max(v, min_value(game.result(state, player, a), opponent, a, alpha, beta, depth + 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, player, last_move, alpha, beta, depth):
        if cutoff_test(state, last_move, depth):
            return eval_fn(state)
        v = infinity
        opponent = 1 if player != 1 else 2
        actions = game.actions(state, player)
        for a in actions:
            v = min(v, max_value(game.result(state, player, a), opponent, a, alpha, beta, depth + 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    cutoff_test = cutoff_test or (lambda state, action, depth: depth > d or game.terminal_test_great(state, action))
    eval_fn = eval_fn or (lambda state: game.evaluate(state))
    best_score = -infinity
    beta = infinity
    opponent = 1 if player != 1 else 2
    actions = game.actions(state, player)
    for a in actions:
        v = min_value(game.result(state, player, a), opponent, a, best_score, beta, 1)
        if v > best_score:
            best_score = v
            yield a


class AI(object):
    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        # the max time you should use, your algorithm’s run time must not exceed the time limit.
        self.time_out = time_out
        self.candidate_list = []
        self.board_history = []

    def go(self, chessboard):
        # Clear candidate_list
        self.candidate_list.clear()
        # ==================================================================
        # To write your algorithm here

        # tic = time()

        game = Game(self.color)

        chessboard[chessboard == -1] = 2

        color = 1 if self.color == 1 else 2
        for a in alphabeta_search(chessboard, game, color, 1):
            self.candidate_list.append(a)
        #
        # print(time() - tic, 's')

        # ==============Find new pos========================================
        # Make sure that the position of your decision in chess board is empty.
        # If not, return error.
        # assert chessboard[new_pos[0], new_pos[1]] == 0
        # Add your decision into candidate_list, Records the chess board
        # self.candidate_list.append(new_pos)