import numpy as np


class ReLU:
    def __init__(self):
        pass

    def __call__(self, z):
        a = np.maximum(0, z)
        return a


class Sigmoid:
    def __init__(self):
        pass

    def __call__(self, z):
        a = 1 / (1 + np.exp(-z))
        return a


class Origin:
    def __init__(self):
        pass

    def __call__(self, z):
        return np.array(z)


class Hardlim:
    def __init__(self):
        pass

    def __call__(self, z):
        z[z >= 0] = 1
        z[z < 0] = -1
        return z
