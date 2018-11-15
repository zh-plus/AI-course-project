import random
from collections import defaultdict

import numpy as np

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)
infinity = np.inf


class Game:
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

        self.actions_time = 0
        self.actions_num = 0
        self.evaluate_time = 0
        self.evaluate_num = 0
        self.kill_actions_time = 0
        self.kill_actions_num = 0
        self.eval_base_time = 0
        self.eval_base_num = 0

    def _heuristic_test_2(self, state, move, player):  # 5 * 5 test
        state[move[0]][move[1]] = player

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

        # horizontal = to_str(state[x, ...])
        # vertical = to_str(state[..., y])

        lr_diag = to_str(state.diagonal(y - x))
        rl_diag = to_str(reverse_state.diagonal(x + y - 14))

        my_record = defaultdict(int)
        my_record = self._check_lines(my_record, horizontal, vertical, lr_diag, rl_diag)

        my_mark = sum(map(lambda situ: situations[situ] * my_record[situ], my_record.keys()))

        state[move[0]][move[1]] = 0

        return my_mark

    def heuristic_evaluate(self, state: np.ndarray, move, player):
        mark = 0  # the returned value of heuristic definition

        next_board = state.copy()

        # Firstly, testing the 4 directions line in 9*9 board
        opponent = 2 if player == 1 else 1
        mark += self._heuristic_test_2(next_board, move, player) + self._heuristic_test_2(next_board, move, opponent)

        return mark

    def has_neighbor(self, state, move):
        board_size = len(state)

        x, y = move
        up, down = max(0, x - 2), min(board_size, x + 3)
        left, right = max(0, y - 2), min(board_size, y + 3)
        partial_board = state[up: down, left: right]  # partial of the total board

        return 1 in partial_board or 2 in partial_board

    def actions(self, state, player):  # generate the available move, then use heuristic evaluation to sort
        valid_move = (state == 0).nonzero()
        valid_move = list(zip(valid_move[0], valid_move[1]))

        # center = len(state) // 2

        if 1 not in state and 2 not in state:
            return [(7, 7)]
            # return [(random.randint(center - 3, center + 4), random.randint(center - 3, center + 4))]

        valid_move = [move for move in valid_move if self.has_neighbor(state, move)]

        valid_move = sorted(valid_move, key=lambda m: self.heuristic_evaluate(state, m, player), reverse=True)

        return valid_move[:max(15, (len(valid_move) // 3) + 1)]

    def max_kill_actions(self, state, player):  # max层米字算杀: 进攻
        if player == 2:
            state = state.copy()
            state[state == 1] = 3
            state[state == 2] = 1
            state[state == 3] = 2

        to_str = lambda array: ''.join(map(str, array))

        valid_move = (state == 0).nonzero()
        valid_move = list(zip(valid_move[0], valid_move[1]))
        valid_move = [move for move in valid_move if self.has_neighbor(state, move)]

        reverse_state = state[::-1]

        kill_move = []
        for move in valid_move:
            x, y = move
            board_size = len(state)
            up, down = max(0, x - 5), min(board_size, x + 6)
            left, right = max(0, y - 5), min(board_size, y + 6)

            horizontal = to_str(state[x, left: right])
            vertical = to_str(state[up: down, y])

            lr_diag = to_str(state.diagonal(y - x))
            rl_diag = to_str(reverse_state.diagonal(x + y - 14))

            exist = lambda *substrings: self._check_substr(horizontal, *substrings) and self._check_substr(vertical, *substrings) and \
                                        self._check_substr(lr_diag, *substrings) and self._check_substr(rl_diag, *substrings)

            if exist('11110', '01111', '11101', '10111', '11011', '01110'):
                kill_move.append(move)

        return kill_move

    def min_kill_actions(self, state):  # min层米字算杀: 进攻+防守
        to_str = lambda array: ''.join(map(str, array))

        valid_move = (state == 0).nonzero()
        valid_move = list(zip(valid_move[0], valid_move[1]))
        valid_move = [move for move in valid_move if self.has_neighbor(state, move)]

        reverse_state = state[::-1]

        kill_move = []
        for move in valid_move:
            x, y = move
            board_size = len(state)
            up, down = max(0, x - 5), min(board_size, x + 6)
            left, right = max(0, y - 5), min(board_size, y + 6)

            horizontal = to_str(state[x, left: right])
            vertical = to_str(state[up: down, y])

            lr_diag = to_str(state.diagonal(y - x))
            rl_diag = to_str(reverse_state.diagonal(x + y - 14))

            exist = lambda *substrings: self._check_substr(horizontal, *substrings) and self._check_substr(vertical, *substrings) and \
                                        self._check_substr(lr_diag, *substrings) and self._check_substr(rl_diag, *substrings)

            if exist('11110', '01111', '11101', '10111', '11011', '01110', '22220', '02222', '22202', '20222', '22022', '02220'):
                kill_move.append(move)

        return kill_move

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

    def get_lines_score(self, state, move):
        to_str = lambda array: ''.join(map(str, array))
        situations = self.situations
        x, y = move

        reverse_state = state[::-1]

        # the opponent state 1 --> 2, 2 --> 1
        opponent_state = state.copy()
        opponent_state[opponent_state == 1] = 3
        opponent_state[opponent_state == 2] = 1
        opponent_state[opponent_state == 3] = 2
        opponent_reverse_state = opponent_state[::-1]

        # calculate the origin lines score (my_mark - 1.1 * opponent_mark) if self.color == 1 else (opponent_mark - 1.1 * my_mark)
        my_horizontal = to_str(state[x, ...])
        my_vertical = to_str(state[..., y])
        my_lr_diag = to_str(state.diagonal(y - x))
        my_rl_diag = to_str(reverse_state.diagonal(x + y - 14))

        op_horizontal = to_str(opponent_state[x, ...])
        op_vertical = to_str(opponent_state[..., y])
        op_lr_diag = to_str(opponent_state.diagonal(y - x))
        op_rl_diag = to_str(opponent_reverse_state.diagonal(x + y - 14))

        my_record = defaultdict(int)
        op_record = defaultdict(int)
        self._check_lines(my_record, my_horizontal, my_vertical, my_lr_diag, my_rl_diag)
        self._check_lines(op_record, op_horizontal, op_vertical, op_lr_diag, op_rl_diag)
        my_mark = sum(map(lambda situ: situations[situ] * my_record[situ], my_record.keys()))
        op_mark = sum(map(lambda situ: situations[situ] * op_record[situ], op_record.keys()))
        score = (my_mark - 1.1 * op_mark) if self.color == 1 else (op_mark - 1.1 * my_mark)

        return score

    def evaluate(self, state: np.ndarray, move, player, base_score):
        this_state: np.ndarray = state.copy()
        x, y = move
        board_size = len(state)
        opponent = 2 if player == 1 else 1

        origin_score = self.get_lines_score(this_state, move)

        surrounding_state = this_state[max(0, x - 1): min(board_size, x + 2), max(0, y - 1): min(board_size, y + 2)]
        player_num = np.sum(surrounding_state == player)
        opponent_num = np.sum(surrounding_state == opponent)
        num_coe = 0.015 * player_num

        this_state[move[0], move[1]] = player

        my_score = self.get_lines_score(this_state, move)

        return (base_score - origin_score + my_score) * (1 + num_coe)

    def eval_base(self, state):
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

        return (my_mark - 1.1 * opponent_mark) if self.color == 1 else (opponent_mark - 1.1 * my_mark)


def alphabeta_search(state, game: Game, player, d=4, kd=7, cutoff_test=None):
    def max_value(state, player, last_move, alpha, beta, depth, base_score):
        opponent = 1 if player != 1 else 2
        v = -infinity

        if cutoff_test(state, last_move, depth):
            if depth < kd and not game.terminal_test_great(state, last_move):  # 算杀
                actions = game.max_kill_actions(state, player)
                if actions:
                    for a in actions:
                        a_base_score = game.evaluate(state, a, player, base_score)
                        v = max(v, min_value(game.result(state, player, a), opponent, a, alpha, beta, depth + 1, a_base_score))
                        if v > beta:
                            return v
                        alpha = max(alpha, v)
                    return v
                else:
                    return base_score
            else:
                return base_score

        actions = game.actions(state, player)
        for a in actions:
            a_base_score = game.evaluate(state, a, player, base_score)
            v = max(v, min_value(game.result(state, player, a), opponent, a, alpha, beta, depth + 1, a_base_score))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, player, last_move, alpha, beta, depth, base_score):
        opponent = 1 if player != 1 else 2
        v = infinity

        if cutoff_test(state, last_move, depth):
            if depth < kd and not game.terminal_test_great(state, last_move):  # 算杀
                actions = game.min_kill_actions(state)
                if actions:
                    for a in actions:
                        a_base_score = game.evaluate(state, a, player, base_score)
                        v = min(v, max_value(game.result(state, player, a), opponent, a, alpha, beta, depth + 1, a_base_score))
                        if v <= alpha:
                            return v
                        beta = min(beta, v)
                    return v
                else:
                    return base_score
            else:
                return base_score

        actions = game.actions(state, player)
        for a in actions:
            a_base_score = game.evaluate(state, a, player, base_score)
            v = min(v, max_value(game.result(state, player, a), opponent, a, alpha, beta, depth + 1, a_base_score))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    cutoff_test = cutoff_test or (lambda state, action, depth: depth > d or game.terminal_test_great(state, action))
    best_score = -infinity
    beta = infinity
    opponent = 1 if player != 1 else 2
    base_score = game.eval_base(state)
    actions = game.actions(state, player)
    for a in actions:
        a_base_score = game.evaluate(state, a, player, base_score)
        v = min_value(game.result(state, player, a), opponent, a, best_score, beta, 1, a_base_score)
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
        game = Game(self.color)

        chessboard[chessboard == -1] = 2

        color = 1 if self.color == 1 else 2
        for a in alphabeta_search(chessboard, game, color, 1, 11):
            self.candidate_list.append(a)


if __name__ == '__main__':
    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput

    # from Gobang.Submit_pro import AI

    with PyCallGraph(output=GraphvizOutput()):
        c = lambda strings: list(map(int, strings))
        chessboard = np.zeros((15, 15), int)

        log = open('chess_log.txt')
        for line in log.readlines():
            array = c(line.split(','))
            chessboard[array[0], array[1]] = array[2]

        color = COLOR_WHITE

        test_ai = AI(15, color, 5)
        test_ai.go(chessboard)

        print(test_ai.candidate_list[-1])
