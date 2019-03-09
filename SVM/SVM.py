from data_process import read_data, split_data
from algorithm_new import SMO
from multiprocessing import Pool

import numpy as np


class SVM:
    def __init__(self, train_data_path, output_test_path, max_iter=50, max_time=10, C=9, tolerance=0.0001, kernel=SMO.linear_kernel):
        self.data = read_data(train_data_path)
        self.output_test_data = read_data(output_test_path)

        # TODO change to submit format
        self.training_data, self.testing_data = split_data(self.data)
        self.train_X, self.train_Y = self.training_data[:, :-1], np.squeeze(self.training_data[:, -1:])
        self.test_X, self.test_Y = self.testing_data[:, :-1], np.squeeze(self.testing_data[:, -1:])

        # print(self.train_X.shape, self.train_Y.shape)

        # self.alphas = np.random.randn(len(self.train_X))
        self.alphas = np.zeros(len(self.train_X))
        self.b = 0.0
        self.m = len(self.train_X)

        self.max_iter = max_iter
        self.max_time = max_time
        self.kernel = kernel
        self.C = C
        self.tolerance = tolerance

    def train(self):
        smo = SMO(self.train_X, self.train_Y, self.C, self.tolerance, self.kernel, self.alphas, self.b)

        smo.run(self, self.max_iter)

    def predict(self, data):
        result = self.b
        for i in range(self.m):
            result += self.alphas[i] * self.train_Y[i] * self.kernel(data, self.train_X[i])
        return 1.0 if result > 0 else -1.0

    def test(self):
        correct = 0
        for data, label in zip(self.test_X, self.test_Y):
            if self.predict(data) == label:
                correct += 1
        return correct / len(self.test_X)

    def output_test(self):
        results = []
        for data in self.output_test_data:
            results.append(str(int(self.predict(data))))

        output = '\n'.join(results)
        print(output)


# def ensemble_predict(svms, data):
#     results = [svm.predict(data) for svm in svms]
#
#     # print(results)
#     c = Counter(results).most_common(1)
#     result = c[0][0]
#     return result
#
#
# def ensemble_test(svms):
#     test_X = svms[0].test_X
#     test_Y = svms[0].test_Y
#
#     correct = 0
#     for data, label in zip(test_X, test_Y):
#         if ensemble_predict(svms, data) == label:
#             correct += 1
#     return correct / len(test_X)


def train_and_test(train_path, test_path):
    svm = SVM(train_data_path=train_path, output_test_path=test_path)
    svm.train()
    acc = svm.test()
    return [acc, svm]


if __name__ == '__main__':
    import sys

    # start = perf_counter()

    if len(sys.argv) == 1:
        sys.argv = ['SVM.py', 'train_data.txt', 'test_data.txt', '-t', '20']

    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]

    with Pool(16) as p:
        future_results = [p.apply_async(train_and_test, args=(train_data_path, test_data_path,)) for i in range(8)]
        results = np.array([f.get() for f in future_results])

    # print(max(results, key=lambda x: x[0]))

    best_svm = max(results, key=lambda x: x[0])[1]

    best_svm.output_test()

    # svm = SVM()
    # svm.train()
    # svm.output_test()
    # svms = results[:, 1]
    # ensemble_acc = ensemble_test(svms)
    # print('ensemble acc:', ensemble_acc)

    # elapse = perf_counter() - start
    # print(elapse)

    # from pycallgraph import PyCallGraph
    # from pycallgraph.output import GraphvizOutput
    # with PyCallGraph(GraphvizOutput()):
    #     svm.train()

    # svm.train()

    # test_acc = svm.test()
    # print(test_acc)

    # imp.time_remain -= elapse

    # result = svm.predict('CELF')
    # for x in result:
    #     print(x)
    # print(imp.last_ise)
    #
    # print(perf_counter() - start, 's')

    sys.exit(0)