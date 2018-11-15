from time import time

import random
import numpy as np

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
infinity = 2147483647

OP_WIN = ['22222']
OP_LIVE4 = ['022220']
OP_LIVE3 = ['02220', '020220', '022020']
OP_LIVE2 = ['0220', '02020', '020020']
OP_LIVE1 = ['020']
OP_BLOCK4 = ['122220', '022221', '[22220', '02222]', '22022', '22202', '20222']
OP_BLOCK3 = ['12220', '02221', '0222]', '[2220', '020221', '122020', '02022]', '[22020']
OP_BLOCK2 = ['0221', '1220', '022]', '[220']

name2situ = {
    'WIN': [op.replace('2', '3').replace('1', '2').replace('3', '1') for op in OP_WIN],
    'OP_WIN': OP_WIN,
    'LIVE4': [op.replace('2', '3').replace('1', '2').replace('3', '1') for op in OP_LIVE4],
    'OP_LIVE4': OP_LIVE4,
    'LIVE3': [op.replace('2', '3').replace('1', '2').replace('3', '1') for op in OP_LIVE3],
    'OP_LIVE3': OP_LIVE3,
    'LIVE2': [op.replace('2', '3').replace('1', '2').replace('3', '1') for op in OP_LIVE2],
    'OP_LIVE2': OP_LIVE2,
    'LIVE1': [op.replace('2', '3').replace('1', '2').replace('3', '1') for op in OP_LIVE1],
    'OP_LIVE1': OP_LIVE1,
    'BLOCK4': [op.replace('2', '3').replace('1', '2').replace('3', '1') for op in OP_BLOCK4],
    'OP_BLOCK4': OP_BLOCK4,
    'BLOCK3': [op.replace('2', '3').replace('1', '2').replace('3', '1') for op in OP_BLOCK3],
    'OP_BLOCK3': OP_BLOCK3,
    'BLOCK2': [op.replace('2', '3').replace('1', '2').replace('3', '1') for op in OP_BLOCK2],
    'OP_BLOCK2': OP_BLOCK2
}

name2point = {
    'WIN': 100000,
    'OP_WIN': -100000,
    'LIVE4': 20000,
    'OP_LIVE4': -20000,
    'LIVE3': 10000,
    'OP_LIVE3': -10000,
    'LIVE2': 1000,
    'OP_LIVE2': -1000,
    'LIVE1': 100,
    'OP_LIVE1': -100,
    'BLOCK4': 10000,
    'OP_BLOCK4': -10000,
    'BLOCK3': 1000,
    'OP_BLOCK3': -1000,
    'BLOCK2': 100,
    'OP_BLOCK2': -100
}


class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []

    def go(self, chessboard):
        self.candidate_list.clear()
        new_pos = gobang(chessboard, self.chessboard_size, self.color, self.time_out)
        assert chessboard[new_pos[0], new_pos[1]] == COLOR_NONE
        self.candidate_list.append(new_pos)


eval_time, eval_numer = 0, 0


def gobang(chessboard, size, color, time_out):
    eval_time = 0
    eval_numer = 0

    chessboard = np.where(chessboard == -1, 2, chessboard)
    color = 2 if color == -1 else 1
    opposite_color = 2 if color == 1 else 1
    coe = -1 if color == 2 else 1

    def max_value(state, alpha, beta, depth, last_step=None):
        if cutoff_test(state, depth, opposite_color, last_step):
            return eval_fn(state)

        v = -infinity
        for next_step in action(state):
            next_state = go(state, next_step, color)
            score = min_value(next_state, alpha, beta, depth - 1, next_step)
            v = max(v, score)

            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta, depth, last_step=None):
        if cutoff_test(state, depth, color, last_step):
            return eval_fn(state)

        v = infinity
        for next_step in action(state):
            next_state = go(state, next_step, opposite_color)
            score = max_value(next_state, alpha, beta, depth - 1, next_step)
            v = min(v, score)
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    def cutoff_test(state, depth, last_color=None, last_step=None):
        if last_color and last_step:
            rev_state = state[::-1]
            y, x = last_step
            h_line = state[y]
            v_line = state[..., x]
            dia_line = state.diagonal(x - y)
            rev_dia_line = rev_state.diagonal((size - 1 - x) - y)

            for line in [h_line, v_line, dia_line, rev_dia_line]:
                line = '[' + ''.join([str(num) for num in line]) + ']'
                if '[{color}{color}{color}{color}{color}]'.format(color=last_color) in line:
                    return True

        return depth == 0 or state.all()

    def eval_fn(state):
        tic = time()

        score = 0
        rev_state = state[::-1]
        for i in range(size):
            h_line = state[i]
            v_line = state[..., i]
            pos_dia_line = state.diagonal(i)
            pos_rev_dia_line = rev_state.diagonal(i)
            if i != 0:
                neg_dia_line = state.diagonal(-i)
                neg_rev_dia_line = rev_state.diagonal(-i)
                score += calculate(neg_dia_line) + calculate(neg_rev_dia_line)

            score += calculate(h_line) + calculate(v_line) + calculate(pos_dia_line) + calculate(pos_rev_dia_line)

        global eval_numer, eval_time
        eval_numer += 1
        eval_time += (time() - tic)

        return coe * score

    def calculate(line):
        if not line.any():
            return 0
        line = '[' + ''.join([str(num) for num in line]) + ']'
        score = 0
        for name, point in name2point.items():
            for situation in name2situ[name]:
                if situation in line:
                    score += point

        return score

    def action(state):
        if not state.any():
            yield [size // 2, size // 2]
        else:
            available_steps = np.where(state == COLOR_NONE)
            available_steps = list(zip(available_steps[0], available_steps[1]))
            random.shuffle(available_steps)

            for step in available_steps:
                if has_neighbor(state, step, 2):
                    yield step

    def go(state, step, next_color):
        next_state = state.copy()
        next_state[step[0], step[1]] = next_color
        return next_state

    def has_neighbor(state, step, limit):
        size = state.shape[0]
        x, y = step
        x_left = max(0, x - limit)
        x_right = min(size, x + limit)
        y_left = max(0, y - limit)
        y_right = min(size, y + limit)
        neighbor = state[x_left:x_right, y_left:y_right]
        return neighbor.any()

    best_score = -infinity
    beta = infinity
    best_step = None
    for next_step in action(chessboard):
        next_state = go(chessboard, next_step, color)
        v = min_value(next_state, best_score, beta, 1)
        if v > best_score:
            best_score = v
            best_step = next_step

    return best_step


if __name__ == '__main__':
    # chessboard: np.ndarray = np.random.choice((0, 1, -1), (15, 15), p=(0.5, 0.25, 0.25))
    # chessboard = np.zeros((15, 15), np.int)  # type: # np.ndarray
    # chessboard[7, 7] = -1
    # chessboard[7, 8] = 1
    # chessboard[7, 9] = -1
    # chessboard[6, 7] = 1
    # chessboard[8, 8] = -1

    chessboard = np.zeros((15, 15), dtype=np.int)
    chessboard[0, 0:2] = -1
    chessboard[0, 7] = -1
    chessboard[1, 1:4] = 1

    chessboard_size = 15
    color = -1
    time_out = 3

    test_ai = AI(chessboard_size, color, time_out)
    test_ai.go(chessboard)

    print(test_ai.candidate_list)

    print('eval_time:', eval_time)
    print('eval_number:', eval_numer)