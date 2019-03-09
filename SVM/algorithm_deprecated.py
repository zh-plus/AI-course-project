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
        self.cached_E = np.zeros(self.m)

    def ini_kernel(self):
        K = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(self.m):
                K[i][j] = self.kernel(self.X[i], self.X[j])
        return K

    @staticmethod
    def linear_kernel(u, v, b=1):
        result = np.dot(u, v) + b
        # result = np.mat(u).transpose() * np.mat(v)
        return result

    @staticmethod
    def gaussian_kernel(u, v, sigma=1):
        result = np.exp(- np.linalg.norm(u[:, np.newaxis] - v[np.newaxis, :]) / (2 * sigma ** 2))
        return result

    def h(self, i):
        return np.sign(np.dot(self.w, self.X[i]) + self.b).astype(int)

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

    def step1(self, i1, i2):
        alpha1 = self.alphas[i1]
        alpha2 = self.alphas[i2]
        y1 = self.Y[i1]
        y2 = self.Y[i2]
        y1y2 = y1 * y2

        L, H = self.get_low_high(i1, i2)
        if L == H:
            return 0

        if 0 < alpha2 < self.C:
            E1 = self.cached_E[i1]
        else:
            E1 = self.E(i1)
        E2 = self.cached_E[i2]

        eta = self.K[i1, i1] + self.K[i2, i2] - 2 * self.K[i1, i2]
        if eta > 0:
            new_alpha2 = alpha2 + y2 * (E1 - E2) / eta
            # clip to the boundary
            if new_alpha2 < L:
                new_alpha2 = L
            elif new_alpha2 > H:
                new_alpha2 = H
        else:
            c1 = eta / 2
            c2 = y2 * (E1 - E2) - eta * alpha2
            L_obj = L * (c1 * L + c2)
            H_obj = H * (c1 * H + c2)
            if L_obj < H_obj - self.tolerance:
                new_alpha2 = H
            elif L_obj > H_obj + self.tolerance:
                new_alpha2 = L
            else:
                new_alpha2 = alpha2

        alpha2_diff = new_alpha2 - alpha2
        if abs(alpha2_diff) < self.tolerance:
            return 0

        new_alpha1 = alpha1 - y1y2 * alpha2_diff
        new_b1 = self.b - E1 - y1 * (new_alpha1 - alpha1) * self.K[i1, i1] - y2 * (new_alpha2 - alpha2) * self.K[i1, i2]
        new_b2 = self.b - E2 - y1 * (new_alpha1 - alpha1) * self.K[i1, i2] - y2 * (new_alpha2 - alpha2) * self.K[i2, i2]

        if 0 < new_alpha1 < self.C:
            self.b = new_b1
        elif 0 < new_alpha2 < self.C:
            self.b = new_b2
        else:
            self.b = (new_b1 + new_b2) / 2

        self.alphas[i1] = new_alpha1
        self.alphas[i2] = new_alpha2

        for i in [x for x in range(self.m) if 0 < self.alphas[x] < self.C]:
            self.cached_E[i] = self.E(i)

        return 1

    def violate_KKT(self, r2, alpha):
        return ((r2 < -self.tolerance) and (alpha < self.C)) or ((r2 > self.tolerance) and (alpha > 0))

    def examine_example(self, i2):
        y2 = self.Y[i2]
        alpha2 = self.alphas[i2]
        # if 0 < alpha2 < self.C:
        #     E2 = self.cached_E[i2]
        # else:
        #     E2 = self.E(i2)
        #     self.cached_E[i2] = E2
        E2 = self.E(i2)

        E_y = E2 * y2
        if self.violate_KKT(E_y, alpha2):
            # heuristic 1
            max_E_diff = 0
            i1 = -1
            for x in range(self.m):
                if 0 < self.alphas[x] < self.C:
                    if x == i2:
                        continue
                    E1 = self.E(x)
                    E_diff = abs(E2 - E1)
                    if E_diff > max_E_diff:
                        max_E_diff = E_diff
                        i1 = x

            if i1 >= 0:
                if self.step(i1, i2):
                    return 1

            # heuristic 2
            random_indexes = np.random.permutation(self.m)
            for i1 in random_indexes:
                if 0 < self.alphas[i1] < self.C:
                    if i1 == i2:
                        continue
                    if self.step(i1, i2):
                        return 1

            # heuristic 3
            random_indexes = np.random.permutation(self.m)
            for i1 in random_indexes:
                if i1 == i2:
                    continue
                if self.step(i1, i2):
                    return 1

        # no suitable i1, i2
        return 0

    def run1(self, svm, max_iter=10):
        examine_all = True
        iter_num = 0
        iter_times = 0
        while iter_num < max_iter:
            iter_times += 1
            alpha_prev = np.copy(self.alphas)
            start = perf_counter()

            num_changed = 0
            if examine_all:
                for i2 in range(self.m):
                    # print(f'iter: {iter_num}, candidate: {i2}')
                    num_changed += self.examine_example(i2)
            else:
                for i2 in range(self.m):
                    if 0 < self.alphas[i2] < self.C:
                        # print(f'iter: {iter_num}, candidate: {i2}')
                        num_changed += self.examine_example(i2)

            # print(f'iter: {iter_num}: {perf_counter() - start} s, num_changed: {num_changed}')

            # Check convergence
            diff = np.linalg.norm(self.alphas - alpha_prev)
            print(f'iter: {iter_num}: {perf_counter() - start} s, diff: {diff}, num_changed: {num_changed}, itertimes: {iter_times}')
            if diff < self.tolerance or num_changed <= iter_times // 10:
                examine_all = True
                iter_num += 1
                iter_times = 0
                continue

            if num_changed == 0:
                iter_num += 1
                iter_times = 0
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

    def run(self, SVM, max_iter=10):
        iter_num = 0
        while iter_num < max_iter:
            alpha_prev = np.copy(self.alphas)
            start = perf_counter()

            num_changed = 0
            for i2 in range(self.m):
                # print(f'iter: {iter_num}, candidate: {i2}')
                num_changed += self.examine_example(i2)

            # print(f'iter: {iter_num}: {perf_counter() - start} s, num_changed: {num_changed}')

            # Check convergence
            diff = np.linalg.norm(self.alphas - alpha_prev)
            acc = SVM.test()
            print(f'iter: {iter_num}: {perf_counter() - start} s, diff: {diff}, num_changed: {num_changed}, acc: {acc}')
            if diff < self.tolerance * 10:
                break

            iter_num += 1

    def get_rand_i(self, j):
        i = j
        while i == j:
            i = random.randrange(0, self.m)

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
