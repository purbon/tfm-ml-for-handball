import numpy as np


class Vector:

    def __init__(self):
        pass

    def diff_norm(self, x, y):
        a = [y[0] - x[0], y[1] - x[1]]
        return self.norm(a)

    def norm(self, x):
        return np.linalg.norm(x)

    def cos(self, a, b=None):
        if b is None:
            b = [a[0], 0]
        return np.dot(a, b) / (self.norm(a) * self.norm(b))

    def angle(self, a, b=None):
        cosine_value = self.cos(a=a, b=b)
        return np.arccos(cosine_value) * (180/np.pi)