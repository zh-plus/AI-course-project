import random

import numpy as np
from pprint import pprint

from time import perf_counter


class SMO:
    def __init__(self, X, Y, C, tolerance, kernel, alphas, b):
        self.X = X
        self.Y = Y
        self.C = C
        self.tolerance = tolerance
        self.kernel = kernel
        self.alphas = alphas
        self.b = b
        self.w = self.calc_w(self.alphas, self.Y, self.X)
        self.m = len(X)
        self.K = self.ini_kernel()
        # self.cached_E = np.zeros(self.m)

    def ini_kernel(self):
        K = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(self.m):
                K[i][j] = self.kernel(self.X[i], self.X[j])
        return K

    @staticmethod
    def linear_kernel(u, v, b=2):
        result = np.dot(u, v) + b
        # result = np.mat(u).transpose() * np.mat(v)
        return result

    @staticmethod
    def gaussian_kernel(u, v, sigma=1):
        result = np.exp(- np.linalg.norm(u[:, np.newaxis] - v[np.newaxis, :]) / (2 * sigma ** 2))
        return result

    @staticmethod
    def kernel_quadratic(x1, x2):
        return np.square(np.dot(x1, x2.T))

    def h(self, i):
        return np.sign(np.dot(self.w, self.X[i]) + self.b).astype(int)

        # result = self.b
        # for j in range(self.m):
        #     k = self.K[i, j]
        #     result += self.alphas[j] * self.Y[j] * k
        # return result

    # def h(self, i):
    #     x = self.X[i]
    #     w = self.w
    #     return np.sign(np.dot(w.T, x.T) + self.b).astype(int)

    def E(self, i):
        return self.h(i) - self.Y[i]

    def calc_w(self, alpha, y, X):
        return np.dot(X.T, np.multiply(alpha, y))

    def get_low_high(self, i, j):
        alpha1 = self.alphas[i]
        alpha2 = self.alphas[j]
        C = self.C

        if self.Y[i] == self.Y[j]:
            L = max(0, alpha1 + alpha2 - C)
            H = min(C, alpha1 + alpha2)
        else:
            L = max(0, alpha2 - alpha1)
            H = min(C, C + alpha2 - alpha1)

        return L, H

    # def step1(self, i1, i2):
    #     def calc_b(X, y, w):
    #         b_tmp = y - np.dot(w.T, X.T)
    #         return np.mean(b_tmp)
    #
    #     def calc_w(alpha, y, X):
    #         return np.dot(X.T, np.multiply(alpha, y))
    #
    #     # Prediction
    #     def h(X, w, b):
    #         return np.sign(np.dot(w.T, X.T) + b).astype(int)
    #
    #     # Prediction error
    #     def E(x_k, y_k, w, b):
    #         return h(x_k, w, b) - y_k
    #
    #     alphas = self.alphas
    #     kernels = self.K
    #
    #     x_i, x_j, y_i, y_j = self.X[i1], self.X[i2], self.Y[i1], self.Y[i2]
    #     k_ij = kernels[i1, i1] + kernels[i2, i2] - 2 * kernels[i1, i2]
    #     if k_ij == 0:
    #         return
    #     alpha_prime_j, alpha_prime_i = alphas[i2], alphas[i1]
    #     (L, H) = self.get_low_high(i1, i2)
    #
    #     # Compute model parameters
    #     self.w = calc_w(alphas, self.Y, self.X)
    #     self.b = calc_b(self.X, self.Y, self.w)
    #
    #     # Compute E_i, E_j
    #     E_i = E(x_i, y_i, self.w, self.b)
    #     E_j = E(x_j, y_j, self.w, self.b)
    #
    #     # Set new alpha values
    #     alphas[i2] = alpha_prime_j + float(y_j * (E_i - E_j)) / k_ij
    #     alphas[i2] = max(alphas[i2], L)
    #     alphas[i2] = min(alphas[i2], H)
    #
    #     alphas[i1] = alpha_prime_i + y_i * y_j * (alpha_prime_j - alphas[i2])

    def step(self, i1, i2):
        alpha1 = self.alphas[i1]
        alpha2 = self.alphas[i2]
        y1 = self.Y[i1]
        y2 = self.Y[i2]

        eta = self.K[i1, i1] + self.K[i2, i2] - 2 * self.K[i1, i2]
        if eta == 0:
            return 0

        L, H = self.get_low_high(i1, i2)
        if L == H:
            return 0

        E1 = self.E(i1)
        E2 = self.E(i2)

        new_alpha2 = alpha2 + y2 * (E1 - E2) / eta
        # clip to the boundary
        if new_alpha2 < L:
            new_alpha2 = L
        elif new_alpha2 > H:
            new_alpha2 = H

        alpha2_diff = new_alpha2 - alpha2
        if abs(alpha2_diff) < self.tolerance:
            return 0

        new_alpha1 = alpha1 - y1 * y2 * alpha2_diff

        self.w += (new_alpha1 - alpha1) * y1 * self.X[i1] + alpha2_diff * y2 * self.X[i2]
        self.b = np.mean(self.Y - np.dot(self.w.T, self.X.T))

        self.alphas[i1] = new_alpha1
        self.alphas[i2] = new_alpha2

        # for i in [x for x in range(self.m) if 0 < self.alphas[x] < self.C]:
        #     self.cached_E[i] = self.E(i)

        return 1

    def violate_KKT(self, r2, alpha):
        return ((r2 < -self.tolerance) and (alpha < self.C)) or ((r2 > self.tolerance) and (alpha > 0))

    def run(self, SVM, max_iter=10):
        iter_num = 0
        # print(f'first iter: acc: {SVM.test()}')
        while True:
            # print(iter_num)
            # start = perf_counter()
            alpha_prev = np.copy(self.alphas)

            for j in range(self.m):
                i = self.get_rand_i(j)
                # step_start = perf_counter()
                self.step(i, j)
                # print(f'step time: {perf_counter() - step_start} s')

            # # Check convergence
            diff = np.linalg.norm(self.alphas - alpha_prev)
            if diff < self.tolerance:
                break
            # acc = SVM.test()
            # print(f'iter: {iter_num}: {perf_counter() - start} s')
            # print(f'iter: {iter_num}: {perf_counter() - start} s, diff: {diff}')
            # print(f'iter: {iter_num}: {perf_counter() - start} s, acc: {acc}')

            if iter_num >= max_iter:
                break

            iter_num += 1

    def get_rand_i(self, j):
        i = j

        # while i == j:
        #     i = random.randrange(0, self.m)

        num = 0
        while 0 <= self.alphas[i] <= self.C:
            i = random.randrange(0, self.m)
            num += 1
            if i == j:
                continue
            if num >= 10:
                break

        return i


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
