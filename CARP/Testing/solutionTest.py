class Solution:
    def __init__(self, routes, loads, costs, total_cost, valid=True, non_valid_num=None):
        self.routes = routes
        self.loads = loads
        self.costs = costs
        self.total_cost = total_cost
        self.valid = valid
        if self.valid:
            self.non_valid_generations = 0
        else:
            self.non_valid_generations = non_valid_num if non_valid_num else 1

    def __hash__(self):
        return hash(str(self.routes))

    def __eq__(self, other):
        return self.routes == other.routes


r = [[(1, 116), (116, 117), (117, 119), (2, 117), (114, 118), (86, 87), (85, 86), (84, 85), (82, 84), (80, 82), (79, 80)],
     [(113, 114), (112, 113), (107, 108), (108, 109), (107, 110), (110, 111), (110, 112), (107, 112), (105, 106), (104, 105), (67, 69)],
     [(106, 107), (102, 104), (66, 67), (67, 68), (69, 71), (71, 72), (72, 73), (44, 45), (34, 139), (33, 139), (34, 45), (43, 46)],
     [(124, 126), (126, 130), (78, 79), (77, 78), (46, 77), (43, 44), (44, 73), (37, 43), (36, 37), (38, 39), (48, 49)],
     [(63, 64), (64, 65), (62, 63), (62, 66), (95, 96), (96, 97), (97, 98), (55, 56), (55, 140), (54, 55), (49, 140), (11, 12), (12, 13), (13, 14),
      (27, 28)], [(36, 38), (39, 40), (11, 33), (11, 27), (28, 29), (28, 30), (30, 32), (25, 27), (24, 25), (20, 22), (6, 8)],
     [(8, 9), (8, 11), (20, 24), (5, 6)]]

loads = [1, 2, 3]
costs = [1, 2, 3]
total_cost = sum(costs)
s1 = Solution(r, loads, costs, total_cost)

s2 = Solution(r, [1, 2], [1, 2], 3)
se = set()
se.add(s1)
se.add(s2)
s3 = Solution([1, 2, 3, [2, 3]], loads, costs, total_cost)
se.add(s3)
print(list(se))