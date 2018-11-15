from Gobang.Submit_pro import AI as AI_pro
from Gobang.Submit_final import AI as AI_next
from Gobang.Submit_pro import Game
import numpy as np


pro_color = 1
next_color = -1

board = np.zeros((15, 15))
ai_pro = AI_pro(15, pro_color, 5)
ai_next = AI_next(15, next_color, 5)

game_pro = Game(pro_color)
game_next = Game(next_color)

while 1:
    ai_next.go(board)
    next_action = ai_next.candidate_list[-1]
    print('next_action:', next_action)
    if game_next.terminal_test_great(board, next_action):
        break
    board[next_action[0]][next_action[1]] = next_color
    print(board)

    ai_pro.go(board)
    pro_action = ai_pro.candidate_list[-1]
    print('pro_action:', pro_action)
    if game_next.terminal_test_great(board, pro_action):
        break
    board[pro_action[0]][pro_action[1]] = pro_color
    print(board)


print(board)
