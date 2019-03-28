import numpy as np
import cv2
import myImage
import filter as flt
import math


#Read image & display it
img = myImage.readImage('cat.jpg')

#Grayscale image
grayImg = myImage.grayScale(img)
myImage.writeImage('Grayscale image', grayImg)

# Detect edges
# Init CFilter() obj
filter = flt.CFilter()
# Generate gaussian kernel
filter.gaussianGenerator(5, 1)

# # Detect edge by Prewitt filter
# xImg, yImg, xyImg = filter.detectByPrewitt(grayImg)
# # Show edges image
# myImage.writeImage('Prewitt X', xImg)
# myImage.writeImage('Prewitt Y', yImg)
# myImage.writeImage('Prewitt XY', xyImg)

# # Detect edges by Sobel filter
# xImg, yImg, xyImg = filter.detectBySobel(grayImg)
# # Show edges image
# myImage.writeImage('Sobel X', xImg)
# myImage.writeImage('Sobel Y', yImg)
# myImage.writeImage('Sobel XY', xyImg)

# # Detect edges by Laplacian of gaussian filter
# logImg = filter.detectByLaplacian(grayImg)
# # Show edges image
# myImage.writeImage('LoG edge detector', logImg)

# Canny edge detector
filter.gaussianGenerator(5, 1)
blurImg, sobelXY, surpressImg, thresholdingImg, cannyImg = filter.detectByCanny(grayImg)
myImage.writeImage('Gauss blurring', blurImg)
myImage.writeImage('Sobel XY', sobelXY)
myImage.writeImage('Non-maximum surpression', surpressImg)
myImage.writeImage('Thresholding image', thresholdingImg)
myImage.writeImage('Canny edge detector', cannyImg)


cv2.waitKey(0)
# Destroy all windows
cv2.destroyAllWindows()
