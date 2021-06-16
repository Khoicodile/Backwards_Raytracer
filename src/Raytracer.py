import numpy as np
from PIL import Image
from test.libregrtest.save_env import multiprocessing


from Sphere import Sphere
from Ray import Ray
from Plane import Plane
from Tri import Tri
from ChestPlane import ChestPlane

# Image resolution
W_RES = 1000
H_RES = 1000

EYE = np.array([0, 0, -5])
# [0]=position & [1]=colour
LIGHT = [np.array([1, 1, 1]), np.array([300, 300, 300])]

C = np.array([0, 0, 0])
UP = np.array([0, -1, 0])
F = np.divide(np.subtract(C, EYE), np.linalg.norm(np.subtract(C, EYE)))
S = np.divide(np.cross(F, UP), np.linalg.norm(np.cross(F, UP)))
U = np.cross(S, F)

FOV = 45
ALPHA = np.deg2rad(FOV) / 2.0
VIEWHEIGHT = 2 * np.tan(ALPHA)
VIEWIDTH = VIEWHEIGHT * W_RES / H_RES  # aspect ratio = 1

P_WIDTH = VIEWIDTH / (W_RES - 1)
P_HEIGHT = VIEWHEIGHT / (H_RES - 1)

BACKGROUND_COLOR = (0, 0, 0)
PROCESSES = 4


def arrayToTuple(color):
    # np.Array in tupel for rgb calculation
    tuple = ()
    for element in color:
        z = int(element) 
        tuple = tuple + (int(element),)
        color = tuple
    return color

            
def calcRay(x, y):
    # Calculate ray vector for x & y
    xcomp = np.multiply(S, (x * P_WIDTH - VIEWIDTH / 2))
    ycomp = np.multiply(U, (y * P_HEIGHT - VIEWHEIGHT / 2))
    return Ray(EYE, np.add(F, np.add(xcomp, ycomp)))


def doRaytracing(queue, start, end, objectlist, index, img, Bcolor):
    # The main algorithmn
    i = index
    for x in range(start, end):
        for y in range(H_RES):
            ray = calcRay(x, y)
            maxdist = float('inf')
            color = Bcolor
            for object in objectlist:
                hitdist = object.intersectionParameter(ray)
                if hitdist:
                    if hitdist < maxdist:
                        maxdist = hitdist
                        color = object.getReflectionColor(ray, LIGHT, objectlist, object.getReflection())
                        color = arrayToTuple(color)  # Wandelt Array in Tupel um
            img.putpixel((x - start, y), color)
            dict = {}
    dict[i] = img
    queue.put(dict)


def initImage1(processes):
    objectlist = []
    objectlist.append(Sphere(np.array([0, 1, 10]), 1, np.array([0, 0, 255])))
    objectlist.append(Sphere(np.array([-1.2, -1, 10]), 1, np.array([0, 255, 0])))
    objectlist.append(Sphere(np.array([1.2, -1, 10]), 1, np.array([255, 0, 0])))
    objectlist.append(Plane(np.array([0, -2.3, 4.5]), np.array([0, 550, -0.1])))
    objectlist.append(Tri(np.array([0, 2, 20]), np.array([2.5, -2, 20]), np.array([-2.5, -2, 20])))
    
    img = Image.new("RGB", (int(W_RES / processes), H_RES))    
    queue = multiprocessing.Queue()
    chunks = int(W_RES / processes)

    procs = []
    dict = {}
    
    for index in range(processes):
        proc = multiprocessing.Process(target=doRaytracing, args=(queue, chunks * index, chunks * (index + 1), objectlist, index, img, BACKGROUND_COLOR))
        procs.append(proc)
        proc.start()
    
    for i in range(processes):
        dict.update(queue.get())
        
    for i in procs:
        i.join()
    
    imgComplete = Image.new("RGB", (W_RES, H_RES))
    x_offset = 0
    
    for x in range(processes):
        imgComplete.paste(dict[x], (x_offset, 0))
        x_offset += int(W_RES / processes)
    
    imgComplete.save("WithoutChestPattern.jpg")
    imgComplete.show()


def initImage2(processes):
    objectlist = []

    objectlist.append(Sphere(np.array([0, 1, 10]), 1, np.array([0, 0, 255])))
    objectlist.append(Sphere(np.array([-1.2, -1, 10]), 1, np.array([0, 255, 0])))
    objectlist.append(Sphere(np.array([1.2, -1, 10]), 1, np.array([255, 0, 0])))
    objectlist.append(ChestPlane(np.array([0, -2.3, 4.5]), np.array([0, 550, -0.1])))
    objectlist.append(Tri(np.array([0, 2, 20]), np.array([2.5, -2, 20]), np.array([-2.5, -2, 20])))
    
    img = Image.new("RGB", (int(W_RES / processes), H_RES))    
    queue = multiprocessing.Queue()
    chunks = int(W_RES / processes)
    procs = []
    dict = {}
    
    for index in range(processes):
        proc = multiprocessing.Process(target=doRaytracing, args=(queue, chunks * index, chunks * (index + 1), objectlist, index, img, BACKGROUND_COLOR))
        procs.append(proc)
        proc.start()
    
    for i in range(processes):
        dict.update(queue.get())
        
    for i in procs:
        i.join()
    
    imgComplete = Image.new("RGB", (W_RES, H_RES))
    x_offset = 0
    
    for x in range(processes):
        imgComplete.paste(dict[x], (x_offset, 0))
        x_offset += int(W_RES / processes)
    
    imgComplete.save("WithChestPattern.jpg")
    imgComplete.show()

if __name__ == '__main__':
    initImage1(PROCESSES)
    initImage2(PROCESSES)
    

