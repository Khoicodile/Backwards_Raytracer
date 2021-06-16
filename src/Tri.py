import numpy as np

from Ray import Ray


class Tri(object):

    def __init__(self, a, b, c, color=None):
        self.a = a
        self.b = b
        self.c = c

        self.shinyness = 100
        self.shadowness = 0.3
        self.reflection = 4
        self.reflectionFactor = 0.5

        self.ka = 0.5  # ka < kd+ks
        self.kd = 0.5  # kd + ks < 1
        self.ks = 0.5

        self.u = np.subtract(self.b, self.a)
        self.v = np.subtract(self.c, self.a)
        if color is not None:
            self.color = color
        else:
            self.color = np.array([255, 255, 0])

    def __repr__(self, *args, **kwargs):
        return 'Triangle(%s,%s,%s' % (repr(self.a), repr(self.b), repr(self.c))

    def intersectionParameter(self, ray):
        w = np.subtract(ray.origin, self.a)
        dv = np.cross(ray.direction, self.v)
        dvu = np.dot(dv, self.u)
        if dvu == 0.0:
            return None
        wu = np.cross(w, self.u)
        r = np.divide(np.dot(dv, w), dvu)
        s = np.divide(np.dot(wu, ray.direction), dvu)
        if 0 <= r <= 1 and 0 <= s and s <= 1 and r + s <= 1:
            return np.divide(np.dot(wu, self.v), dvu)


    def normalAt(self):
        return np.divide(np.cross(self.u, self.v), np.linalg.norm(np.cross(self.u, self.v)))

    def getReflection(self):
        return self.reflection

    def getColor(self, ray, light, objectlist):
        # Phong
        intersectionPoint = ray.pointAtParameter(self.intersectionParameter(ray))
        lightWay = np.subtract(light[0], intersectionPoint)
        lightWay = np.divide(lightWay, np.linalg.norm(lightWay))

        lr = lightWay - 2 * np.dot(intersectionPoint, lightWay)
        lr = np.divide(lr, np.linalg.norm(lr))

        ambient = self.color * self.ka
        diffus = light[1] * self.kd * np.dot(lightWay, self.normalAt())
        spekular = light[1] * self.ks * (np.linalg.norm(np.dot(lr, -ray.direction)) ** self.shinyness)

        Cout = ambient + diffus + spekular

        # Shadow calculation
        light = Ray(intersectionPoint, light[0])
        light.origin = light.origin - (1 / 10000 * (light.origin - light.direction))

        for object in objectlist:
            point = object.intersectionParameter(light)
            if point and point > 0:
                Cout = Cout * self.shadowness
                break
        return Cout

    def getReflectionColor(self, ray, light, objectlist, reflection):
        # Recursive reflection calculation
        if reflection == 0:
            color = self.getColor(ray, light, objectlist)
        else:
            intersectionPoint = ray.pointAtParameter(self.intersectionParameter(ray))
            normal = self.normalAt()
            color = self.getColor(ray, light, objectlist)
            newColor = None
            newRay = Ray(intersectionPoint, ray.direction - (2 * np.dot(normal, ray.direction) * normal))
            for object in objectlist:
                newIntersection = object.intersectionParameter(newRay)
                if newIntersection and newIntersection >= 0:
                    newColor = object.getReflectionColor(newRay, light, objectlist, reflection - 1)
                    break
            if newColor is not None:
                color = color + newColor * self.reflectionFactor
        return color
