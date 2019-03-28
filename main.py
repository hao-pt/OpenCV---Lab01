import numpy as np
import cv2
import argparse
import myImage
import filter as flt
import math


#Instatiate ArgumentParser() obj and parse argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True, help = "Path to input image")
ap.add_argument("-c", "--code", required= True, help = "Code action")
# ap.add_argument("")
args = vars(ap.parse_args())

# Main driver
def main(args):
    #Read image & display it
    img = myImage.readImage(args['input'])

    #Grayscale image
    grayImg = myImage.grayScale(img)
    myImage.writeImage('Grayscale image', grayImg)

    # Detect edges
    # Init CFilter() obj
    filter = flt.CFilter()
    # Generate gaussian kernel
    filter.gaussianGenerator(5, 1)

    # Get code action
    code = int(args['code'])

    if code == 1:
        # Detect edges by Sobel filter
        xImg, yImg, xyImg = filter.detectBySobel(grayImg)
        # Show edges image
        myImage.writeImage('Sobel X', xImg)
        myImage.writeImage('Sobel Y', yImg)
        myImage.writeImage('Sobel XY', xyImg)
    elif code == 2:
        # Detect edge by Prewitt filter
        xImg, yImg, xyImg = filter.detectByPrewitt(grayImg)
        # Show edges image
        myImage.writeImage('Prewitt X', xImg)
        myImage.writeImage('Prewitt Y', yImg)
        myImage.writeImage('Prewitt XY', xyImg)
    elif code == 3:
        # Detect edges by Laplacian of gaussian filter
        logImg = filter.detectByLaplacian(grayImg)
        # Show edges image
        myImage.writeImage('LoG edge detector', logImg)
    elif code == 4:
        # Canny edge detector
        filter.gaussianGenerator(5, 1)
        blurImg, sobelXY, surpressImg, thresholdingImg, cannyImg = filter.detectByCanny(grayImg)
        myImage.writeImage('Gauss blurring', blurImg)
        myImage.writeImage('Sobel XY', sobelXY)
        myImage.writeImage('Non-maximum surpression', surpressImg)
        myImage.writeImage('Thresholding image', thresholdingImg)
        myImage.writeImage('Canny edge detector', cannyImg)
    else:
        print('There are just 4 function from 1 to 4. Please enter code again!')
        return 0

    cv2.waitKey(0)
    # Destroy all windows
    cv2.destroyAllWindows()
    return 1

if __name__ == '__main__':
    main(args)
