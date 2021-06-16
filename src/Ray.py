import numpy as np


class Ray(object):

    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = np.divide(direction, np.linalg.norm(direction))  # normiert

    def __repr__(self):
        return 'Ray(%s,%s)' % (repr(self.origin), repr(self.direction))

    def pointAtParameter(self, t):
        # Point on ray with parameter t
        return np.add(self.origin, np.multiply(self.direction, t))
