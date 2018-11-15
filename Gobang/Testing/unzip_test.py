l1 = [1, 2, 3]
l2 = [3, 4, 5]


def add(*num):
    return sum(num)

print(add(*l1, *l2))
