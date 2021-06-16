import numpy as np
from Ray import Ray


class ChestPlane(object):
    
    def __init__(self, point, normal, color=None):
        self.point = point
        self.normal = np.divide(normal, np.linalg.norm(normal))
        
        self.shinyness = 100
        self.shadowness = 0.3
        self.reflection = 0.5
        self.reflectionFactor = 0.5
        
        self.ka = 1  # ka < kd+ks
        self.kd = 0.5  # kd + ks < 1
        self.ks = 0.5
        
        self.color1 = (0, 0, 0)
        self.color2 = (255, 255, 255)
        self.checkSize = 2
        
    def intersectionParameter(self, ray):
        op = ray.origin - self.point
        a = np.dot(op, self.normal)
        b = np.dot(ray.direction, self.normal)
        if b < 0:
            return -a / b
        else:
            return None
        
    def __repr__(self):
        return 'Plane(%s,%s)' % (repr(self.point), repr(self.normal))
    
    def normalAt(self):
        return self.normal
    
    def getReflection(self):
        return self.reflection
    
    def getColor(self, ray, light, objectlist):
        # Colour calculation for chest plane
        intersectionPoint = ray.pointAtParameter(self.intersectionParameter(ray))
        
        v = intersectionPoint * (1.0 / self.checkSize)
        if (int(abs(v[0]) + 0.5) + int(abs(v[1]) + 0.5) + int(abs(v[2]) + 0.5)) % 2:
            color = self.color1
        else:
            color = self.color2
        
        # Phong lightning calculation
        lightWay = np.subtract(light[0], intersectionPoint)
        lightWay = np.divide(lightWay, np.linalg.norm(lightWay))
        
        lr = lightWay - 2 * np.dot(intersectionPoint, lightWay)
        lr = np.divide(lr, np.linalg.norm(lr))
        
        ambient = color * self.ka
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
        if reflection == 0 or self.reflection == 0:
            color = self.getColor(ray, light, objectlist)
            return color
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
