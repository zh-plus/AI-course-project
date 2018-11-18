import numpy as np
import random

depot = 0

routes = [[(1, 2), (4, 7), (9, 3)], [(5, 4), (3, 1), (8, 2)], [(6, 4), (4, 1), (3, 2)]]

selected_arc_index = random.randrange(0, len(routes))  # start <= N < end
selected_arc = routes[selected_arc_index]
selected_task_index = random.randrange(0, len(selected_arc))  # start <= N < end
print(selected_arc, selected_task_index)

# calculate changed costs
u, v = selected_arc[selected_task_index]
pre_end = selected_arc[selected_task_index - 1][1] if selected_task_index != 0 else depot
next_start = selected_arc[selected_task_index + 1][0] if selected_task_index != len(selected_arc) - 1 else depot

print(u, v, pre_end, next_start)

selected_task = selected_arc.pop(selected_task_index)

routes.append([])
inserting_arc_index = random.randrange(0, len(routes))
inserting_arc = routes[inserting_arc_index]
inserting_position = random.randint(0, len(inserting_arc))  # start <= N <= end
print(inserting_arc, inserting_position)

inserting_arc.insert(inserting_position, selected_task)
print(routes)
