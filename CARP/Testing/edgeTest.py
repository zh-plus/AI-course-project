class Edge:
    def __init__(self, u, v, cost, demand):
        self.u = u
        self.v = v
        self.cost = cost
        self.demand = demand

    def __hash__(self):
        return hash((self.u, self.v)) + hash((self.v, self.u))

    def __eq__(self, other):
        t = (self.u, self.v)
        return t == (other.u, other.v) or t == (other.v, other.u)


e1 = Edge(6, 2, 10, 0)
e2 = Edge(2, 6, 10, 0)
e3 = Edge(1, 5, 10, 0)
e4 = Edge(5, 1, 10, 0)

print(hash(e1))
print(hash(e2))
print(hash(e3))
print(e4 == e3)