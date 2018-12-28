from data_process import read_data, split_data


class SVM:
    def __init__(self):
        self.data = read_data()

        # TODO change to submit format
        self.training_data, self.testing_data = split_data(self.data)






if __name__ == '__main__':
    import sys

    # start = perf_counter()

    # if len(sys.argv) == 1:
    #     sys.argv = ['IMP.py', '-i', 'network.txt', '-k', '5', '-m', 'LT', '-t', '60']

    svm = SVM()
    # elapse = perf_counter() - start
    # imp.time_remain -= elapse

    # result = svm.predict('CELF')
    # for x in result:
    #     print(x)
    # print(imp.last_ise)
    #
    # print(perf_counter() - start, 's')

    sys.exit(0)
