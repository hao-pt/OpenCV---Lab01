#Include libraries
import numpy as np
import cv2

#Read image
img = cv2.imread('gray_lena.png')

#Convert image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Display image
cv2.namedWindow('Original image', cv2.WINDOW_AUTOSIZE)
cv2.imshow('Original image',img)

cv2.waitKey(0)

#Destroy all windows
cv2.destroyAllWindows()

