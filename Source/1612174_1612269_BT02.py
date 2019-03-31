#Built-in libs
import numpy as np
import cv2
import argparse
import math
from matplotlib import pyplot as plt

#Built-out libs
import myImage
import filter as flt


#Instatiate ArgumentParser() obj and parse argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True, help = "Path to input image")
ap.add_argument("-c", "--code", required= True, help = "Code action")
args = vars(ap.parse_args())

# Main driver
def main(args):
    #Measure time
    e1 = cv2.getTickCount()

    #Read image & display it
    img = myImage.readImage(args['input'])

    #Grayscale image
    grayImg = myImage.grayScale(img)
    # myImage.writeImage('Grayscale image', grayImg)

    plt.figure(1)
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122)
    plt.imshow(grayImg, cmap='gray', interpolation = 'bicubic')
    plt.title('Gray-scale image'), plt.xticks([]), plt.yticks([])

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
        # # Use cv2 method
        # myImage.writeImage('Sobel X', xImg)
        # myImage.writeImage('Sobel Y', yImg)
        # myImage.writeImage('Sobel XY', xyImg)

        plt.figure(2)
        plt.subplot(131)
        plt.imshow(xImg, cmap='gray', interpolation='bicubic')
        plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
        plt.subplot(132)
        plt.imshow(yImg, cmap='gray', interpolation = 'bicubic')
        plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
        plt.subplot(133)
        plt.imshow(xyImg, cmap='gray', interpolation = 'bicubic')
        plt.title('Sobel XY'), plt.xticks([]), plt.yticks([])

    elif code == 2:
        # Detect edge by Prewitt filter
        xImg, yImg, xyImg = filter.detectByPrewitt(grayImg)
        # Show edges image
        # myImage.writeImage('Prewitt X', xImg)
        # myImage.writeImage('Prewitt Y', yImg)
        # myImage.writeImage('Prewitt XY', xyImg)

        plt.figure(2)
        plt.subplot(131)
        plt.imshow(xImg, cmap='gray', interpolation='bicubic')
        plt.title('Prewitt X'), plt.xticks([]), plt.yticks([])
        plt.subplot(132)
        plt.imshow(yImg, cmap='gray', interpolation = 'bicubic')
        plt.title('Prewitt Y'), plt.xticks([]), plt.yticks([])
        plt.subplot(133)
        plt.imshow(xyImg, cmap='gray', interpolation = 'bicubic')
        plt.title('Prewitt XY'), plt.xticks([]), plt.yticks([])

    elif code == 3:
        # Detect edges by Laplacian of gaussian filter
        logImg = filter.detectByLaplacian(grayImg)
        # Show edges image
        # myImage.writeImage('LoG edge detector', logImg)
        
        plt.figure(2)
        plt.imshow(logImg, cmap='gray', interpolation='bicubic')
        plt.title('Negative laplacian'), plt.xticks([]), plt.yticks([])

    elif code == 4:
        # Canny edge detector
        filter.gaussianGenerator(5, 1)
        blurImg, sobelXY, surpressImg, thresholdingImg, cannyImg = filter.detectByCanny(grayImg)
        # myImage.writeImage('Gauss blurring', blurImg)
        # myImage.writeImage('Sobel XY', sobelXY)
        # myImage.writeImage('Non-maximum surpression', surpressImg)
        # myImage.writeImage('Thresholding image', thresholdingImg)
        # myImage.writeImage('Canny edge detector', cannyImg)

        plt.figure(2)
        plt.subplot(121)
        plt.imshow(blurImg, cmap='gray', interpolation='bicubic')
        plt.title('Gaussian blur'), plt.xticks([]), plt.yticks([])
        plt.subplot(122)
        plt.imshow(sobelXY, cmap='gray', interpolation='bicubic')
        plt.title('Sobel XY'), plt.xticks([]), plt.yticks([])
        
        plt.figure(3)
        plt.subplot(121)
        plt.imshow(surpressImg, cmap='gray', interpolation='bicubic')
        plt.title('Non-maximum surpression'), plt.xticks([]), plt.yticks([])
        plt.subplot(122)
        plt.imshow(thresholdingImg, cmap='gray', interpolation='bicubic')
        plt.title('Thresholding'), plt.xticks([]), plt.yticks([])

        plt.figure(4)
        plt.imshow(cannyImg, cmap='gray', interpolation='bicubic')
        plt.title('Canny edge detector'), plt.xticks([]), plt.yticks([])

    else:
        print('There are just 4 function from 1 to 4. Please enter code again!')
        return 0

    
    e2 = cv2.getTickCount()
    time = (e2 - e1)/cv2.getTickFrequency()
    print('Time: %.2f(s)' %(time))

    plt.show()

    # cv2.waitKey(0)
    # # Destroy all windows
    # cv2.destroyAllWindows()

    
    return 1

if __name__ == '__main__':
    main(args)
    
