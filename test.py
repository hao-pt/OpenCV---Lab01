import numpy as np
import cv2

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


a = np.array([range(-2, 3)]).reshape(5, 1)
print(a)
b = np.tile(a, (1, 5))

c = np.array([range(-2, 3)])
d = np.tile(c, (5, 1))
print((np.power(d, 2) + np.power(b, 2))/2)


print(gaussian_kernel(5, 1.4))

x = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
y = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
d = np.array([[1, 4, 3, 4], [5, 4, 5, 6], [10, 7, 8, 9], [23, 5, 9, -2]])
#print(d)
print(d[x + 1, y + 2])
print(np.sum(d[x + 1, y + 2] == 4) > 0)