import numpy as np

from Ray import Ray


class Sphere(object):

    def __init__(self, center, radius, color=None):
        self.center = center
        self.radius = radius

        self.shinyness = 100
        self.shadowness = 0.7
        self.reflection = 4
        self.reflectionFactor = 0.5

        self.ka = 0.5  # ka < kd+ks
        self.kd = 0.5  # kd + ks < 1
        self.ks = 0.5
        if color is not None:
            self.color = color
        else:
            self.color = np.array([255, 0, 0])

    def __repr__(self):
        return 'Sphere(%s,%s)' % (repr(self.center), self.radius)

    def intersectionParameter(self, ray):
        co = np.subtract(self.center, ray.origin)
        v = np.dot(co, ray.direction)
        discriminant = np.add(np.subtract(np.multiply(v, v), np.dot(co, co)),
                              np.multiply(self.radius, self.radius))  # discriminant ist the sqrt thing, herein pq
        if discriminant < 0:
            return None
        else:
            return np.subtract(v, np.sqrt(discriminant))

    def normalAt(self, p):
        return np.divide(np.subtract(p, self.center), np.linalg.norm(np.subtract(p, self.center)))

    def getReflection(self):
        return self.reflection

    def getColor(self, ray, light, objectlist):
        # Phong lightning calculation
        intersectionPoint = ray.pointAtParameter(self.intersectionParameter(ray))
        lightWay = np.subtract(light[0], intersectionPoint)
        lightWay = np.divide(lightWay, np.linalg.norm(lightWay))

        lr = lightWay - (2 * np.dot(intersectionPoint, lightWay))
        lr = np.divide(lr, np.linalg.norm(lr))

        ambient = self.color * self.ka
        diffus = light[1] * self.kd * np.dot(lightWay, self.normalAt(intersectionPoint))
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
            return color
        else:
            intersectionPoint = ray.pointAtParameter(self.intersectionParameter(ray))
            normal = self.normalAt(intersectionPoint)
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
