class test:
    def __init__(self):
        self.l = [
            1, 2, 3, self.l[0]
        ]

    def __str__(self):
        return str(self.l)

print(test())