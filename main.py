import numpy as np
import cv2
import myImage
import filter as flt

#Read image & display it
img = myImage.readImage('lena.png')
myImage.writeImage('Original image', img)

#Grayscale image
grayImg = myImage.grayScale(img)
myImage.writeImage('Grayscale image', grayImg)

#Detect edges
filter = flt.CFilter()
xImg, yImg, xyImg = filter.edgeDetector(grayImg, 'sobel')
#Show edges image
myImage.writeImage('Vertical image', xImg)
myImage.writeImage('Horizontal image', xImg)
myImage.writeImage('Magnitude image', xImg)

cv2.waitKey(0)
#Destroy all windows
cv2.destroyAllWindows()

