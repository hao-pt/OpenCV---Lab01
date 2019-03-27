import numpy as np
import cv2
import myImage
import filter as flt
import math


#Read image & display it
img = myImage.readImage('girl.png')

#Grayscale image
grayImg = myImage.grayScale(img)
myImage.writeImage('Grayscale image', grayImg)

# Detect edges
filter = flt.CFilter()
# filter.gaussianGenerator(5, 1.4)
# xImg, yImg, xyImg = filter.detectByPrewitt(grayImg)
# #Show edges image
# myImage.writeImage('Vertical image', xImg)
# myImage.writeImage('Horizontal image', yImg)
# myImage.writeImage('Edge detection', xyImg)

# Canny edge detector
filter.gaussianGenerator(5, 1)
cannyImg = filter.detectByCanny(grayImg)
myImage.writeImage('Canny edge detector', cannyImg)

# # Detect edges by LoG
# outImg = filter.detectByLaplacian(grayImg)
# myImage.writeImage('Edge detection - Laplacian of Gaussian', outImg)

# meanImg = filter.smoothenImage(grayImg, 'mean')
# myImage.writeImage('Blur image - mean filter', meanImg)

# print(filter.gaussianGenerator(7, 1.5))
# gaussImg = filter.smoothenImage(grayImg, 'gauss')
# myImage.writeImage('Blur image - gaussian filter', gaussImg)

# xImg = filter.edgeDetector(grayImg, 'sobel')
# myImage.writeImage('xy_image', xImg)
# #cv2.imwrite('xImage.png', xImg)

# a = np.array([[289, 370, 3], [-4, 240, 6], [30, 90, 500]], np.int)
# b = np.array([2, 3, 4], np.uint8)
# c = a * b
# print(c.dtype, c)

cv2.waitKey(0)
# Destroy all windows
cv2.destroyAllWindows()
