import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint


class SMO:
    def __init__(self, X, Y, C, kernel, alphas, b, errors):
        self.X = X
        self.Y = Y
        self.C = C
        self.kernel = kernel
        self.alphas = alphas
        self.b = b
        self.errors = errors
        self.obj = []
        self.m = len(X)

        self.E = [] #TODO thinking about function E

    @staticmethod
    def linear_kernel(u, v, b=1):
        return np.dot(u, v.T) + b

    @staticmethod
    def gaussian_kernel(u, v, sigma=1):
        return np.exp(- np.linalg.norm(u[:, np.newaxis] - v[np.newaxis, :], axis=2) / (2 * sigma ** 2))

    def KKT(self, i):
        y_g = self.g(i) * self.Y[i]
        alpha = self.alphas[i]
        conditions = {
            alpha == 0 or y_g >= 1,
            0 < alpha < self.C or y_g == 1,
            alpha > self.C or y_g <= 1
        }

        return True in conditions

    def g(self, i):
        x = self.X[i]
        result = self.b
        for j in range(self.m):
            result += self.alphas[j] * self.Y * self.kernel(self.X[j], x)
        return result

    def E(self, i):
        return self.g(i) - self.Y[i]

    def get_low_high(self, i1, i2):
        alpha1 = self.alphas[i1]
        alpha2 = self.alphas[i2]
        C = self.C

        if self.Y[i1] == self.Y[i2]:
            L = max(0, alpha1 + alpha2 - C)
            H = min(C, alpha1 + alpha2)
        else:
            L = max(0, alpha2 - alpha1)
            H = min(C, C + alpha2 - alpha1)

        return L, H

    # @staticmethod
    # def objective_fn(alphas, y, kernel, X):
    #     return np.sum(alphas) - 0.5 * np.sum()

    def select_alpha(self):
        # heuristic selection
        indexes = [i for i in range(self.m) if 0 < self.alphas[i] < self.C]
        non_satisfy = set(range(self.m)) - set(indexes)
        indexes.extend(non_satisfy)

        for i in indexes:
            if self.KKT(i):
                continue

            E1 = self.E(i)
            most = min if E1 >= 0 else max
            j = most(range(self.m), key=lambda x: self.E(x))

            return i, j

    def step(self):
        pass



if __name__ == '__main__':
    x_len, y_len = 2, 5
    x = np.random.rand(x_len, 1)
    y = np.random.rand(y_len, 1)
    #
    # z = SMO.gaussian_kernel(x, y)
    # print(z.shape)

    z = x[:, np.newaxis] - y[np.newaxis, :]
    pprint(z)

    pprint(x[:, np.newaxis])
    pprint(y[:, np.newaxis])

    pprint(x[:, np.newaxis].shape)
    pprint(y[:, np.newaxis].shape)
