import random
import time
from pprint import pprint
import sys
import copy

import numpy as np

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)


class Node:
    def __init__(self, role, isAI, board, parent=None, x=-1, y=-1):  # the parent of root node is always None
        self.role = role
        if x != -1:  # the first node do not have x y
            board[x][y] = role[isAI]

        self.board = board
        self.value = None
        self.alpha = -np.inf
        self.beta = np.inf
        self.isAI = isAI
        self.parent = parent
        self.children = []

    def _heuristic_test_1(self, board, move, chessman):  # 9 * 9 test
        mark = 0;

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

        opponent = 0 - chessman
        for line in lines:
            p = line.index(chessman)
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

    def heuristic_evaluate(self, move):
        board = copy.deepcopy(self.board)
        chessman = self.role[self.isAI]  # the represent of testing partern (-1 or 1)
        board[move[0]][move[1]] = chessman

        mark = 0  # the returned value of heuristic definition

        # Tirstly, testing the 4 directions line in 9*9 board
        mark += self._heuristic_test_1(board, move, chessman)

        # Then, testing the 8 grids and 8 groups in 5*5 board
        mark += self._heuristic_test_2(board, move, chessman)

        return mark

    def generate(self):  # generate the available move, then use heuristic evaluation to sort
        valid_move = (self.board == 0).nonzero()
        valid_move = list(zip(valid_move[0], valid_move[1]))
        valid_move = sorted(valid_move, key=self.heuristic_evaluate, reverse=True)

        return valid_move

    def propagete_alphabeta(self):  # update the alpha/beta value of parent and siblings
        if self.parent.isAI and self.value > self.parent.alpha:
            self.parent.alpha = self.value
            for child in self.parent.children:
                child.alpha = self.value
        elif not self.parent.isAI and self.value < self.parent.beta:
            self.parent.beta = self.value
            for child in self.parent.children:
                child.beta = self.value

    def update_value(self):
        self.value = max(self.children, key=lambda child: child.value if self.isAI else -child.value)

    def evaluate(self):  # evaluate function that define the condition fo board, only used in leaf node
        board = self.board

    def terminal_test(self):
        return False
        # TODO


class AI(object):
    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.role = {True: color, False: -1 if color == 1 else 1}
        # the max time you should use, your algorithm’s run time must not exceed the time limit.
        self.time_out = time_out
        self.candidate_list = []

    def alphabeta_search(self, node, depth):
        if depth == 3 or node.terminal_test():  # reach leaf nodes
            node.value = node.evaluate()
            node.propagete_alphabeta()
            return

        valid_move = node.generate()
        for x, y in valid_move:
            n = Node(self.role, not node.isAI, node.board, node, x, y)
            node.children.append(n)

        # TODO here
        child: Node
        for child in node.children:
            if child.alpha >= child.beta:
                self.alphabeta_search(n, depth + 1)

    def go(self, chessboard):
        # Clear candidate_list
        self.candidate_list.clear()
        # ==================================================================
        # To write your algorithm here
        # Here is the simplest sample:Random decision

        node = Node(self.role, False, chessboard)
        print(node.board)
        print(node.role[node.isAI])

        action = self.alphabeta_search(node, 0)

        # valid_move = node.generate()
        #
        # print(valid_move)
        # for move in valid_move:
        #     print(node.heuristic_evaluate(move))

        # pos_idx = random.randint(0, len(valid_move) - 1)
        # new_pos = valid_move[pos_idx]

        # ==============Find new pos========================================
        # Make sure that the position of your decision in chess board is empty.
        # If not, return error.
        # assert chessboard[new_pos[0], new_pos[1]] == 0
        # Add your decision into candidate_list, Records the chess board
        # self.candidate_list.append(new_pos)


if __name__ == '__main__':
    chessboard: np.ndarray = np.random.choice((0, 1, -1), (15, 15), p=(0.5, 0.25, 0.25))
    chessboard_size = 15
    color = 1
    time_out = 3

    test_ai = AI(chessboard_size, color, time_out)
    test_ai.go(chessboard)
