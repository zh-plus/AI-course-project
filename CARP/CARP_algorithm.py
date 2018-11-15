from CARP_info import CARPInfo


class CARPAlgorithm:
    def __init__(self, info):
        """

        :type info: CARPInfo
        """
        self.info = info
        self.min_dist = info.min_dist
        self.population = self.path_scanning()

    def path_scanning(self):
        free = self.info.tasks.copy()
        pass

    def step(self):
        pass
