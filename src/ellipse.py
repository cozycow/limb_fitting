import numpy as np


class Ellipse(list):
    def __init__(self, *items):
        super().__init__(items)

    @classmethod
    def from_matrix(cls, A):
        coeff = A[0, 0], A[0, 1] + A[1, 0], A[1, 1], A[0, 2] + A[2, 0], A[1, 2] + A[2, 1], A[2, 2]
        return cls(*coeff)

    @property
    def matrix(self):
        A, B, C, D, E, F = self
        return np.array([[A, B / 2, D / 2],
                         [B / 2, C, E / 2],
                         [D / 2, E / 2, F]])

    @property
    def center(self):
        A, B, C, D, E, F = self
        Q = B ** 2 - 4 * A * C
        return (2 * C * D - B * E) / Q, (2 * A * E - B * D) / Q

    @property
    def major(self, minor=False):
        A, B, C, D, E, F = self

        Q = B ** 2 - 4 * A * C
        P = np.sqrt((A - C) ** 2 + B ** 2)
        R = 2 * (A * E ** 2 + C * D ** 2 - B * D * E + Q * F)
        return np.max([- np.sqrt(R * (A + C + P)) / Q, - np.sqrt(R * (A + C - P)) / Q], axis=0)

    @property
    def minor(self):
        A, B, C, D, E, F = self

        Q = B ** 2 - 4 * A * C
        P = np.sqrt((A - C) ** 2 + B ** 2)
        R = 2 * (A * E ** 2 + C * D ** 2 - B * D * E + Q * F)
        return np.min([- np.sqrt(R * (A + C + P)) / Q, - np.sqrt(R * (A + C - P)) / Q], axis=0)

    @property
    def radius(self):
        return (self.minor + self.major) / 2

    @property
    def theta(self):
        A, B, C, D, E, F = self
        theta = np.arctan2(-B, C - A) / 2 ### Need to check
        return theta

    @property
    def eccentricity(self):
        a, b = self.major, self.minor
        return np.sqrt(1 - (b / a) ** 2)

    @property
    def flatness(self):
        a, b = self.major, self.minor
        return 1 - b / a

    def patch(self, **kwargs):
        from matplotlib import patches
        return patches.Ellipse(self.center, 2 * self.major, 2 * self.minor,
                               angle=self.theta * 180 / np.pi, fill=False, **kwargs)


def fit_ellipse(x, y):
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x), np.std(y)

    M = np.array([[1 / sx, 0, -mx / sx],
                  [0, 1 / sy, -my / sy],
                  [0, 0, 1]])
    x_, y_, _ = M @ np.array([x, y, np.ones_like(x)])

    A = np.array([x_ ** 2, x_ * y_, y_ ** 2, x_, y_])
    B = np.sum(A, axis=1)

    ellipse = Ellipse(*(B @ np.linalg.inv(A @ A.T)), -1)
    Q = M.T @ ellipse.matrix @ M
    return Ellipse.from_matrix(Q)
