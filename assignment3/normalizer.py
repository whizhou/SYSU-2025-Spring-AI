import numpy as np

class LinearNormalizer:
    def __init__(self, data):
        self.data = data
        # self.min = np.min(data, axis=0)
        # self.max = np.max(data, axis=0)
        self.offset = np.mean(data, axis=0)
        self.scale = 1 / (np.std(data, axis=0) + 1e-8)
        # self.scale = 1 / (self.max - self.min)

    def normalize(self, data=None):
        return (data - self.offset) * self.scale

    def denormalize(self, normalized_data):
        return normalized_data / self.scale + self.offset