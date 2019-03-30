#Built-in lib
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt 
import argparse

#Built-out libs
import myImage

#Instatiate ArgumentParser() obj and parse argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True, help = "Path to input image")
ap.add_argument("-c", "--code", required= True, help = "Code action")
args = vars(ap.parse_args())

# Find threshold base on ratio of two threshold
# In this case, we pick sigma = 0.33 base on experience when testing often give stable result
def thresholdSeeking(img, sigma = 0.33):
    # Get median (or can get mean instead)
    med = np.median(img)
    
    #Find lowThreshold and highThreshold base on sigma
    highThreshold = math.ceil(med * (1 + sigma))
    lowThreshold = math.ceil(med * (1 - sigma))

    return lowThreshold, highThreshold

def main(args):
    # Measure time
    e1 = cv2.getTickCount()

    #Input image
    img = myImage.readImage(args['input'])

    #Gray-scale
    img = myImage.grayScale(img)

    # Get code active
    code = int(args['code'])
    
    if code == 1:
        # Sobel edge detector
        # Blur image first
        blurImg = cv2.GaussianBlur(img, (5, 5), 0)
        sobelX = cv2.Sobel(blurImg, cv2.CV_64F , 1, 0, ksize = 3)
        sobelY = cv2.Sobel(blurImg, cv2.CV_64F, 0, 1, ksize = 3)
        # Compute magnitude
        sobelXY,_ = cv2.cartToPolar(sobelX, sobelY)

        # Convert image back to abs CV_8U
        sobelX = cv2.convertScaleAbs(sobelX)
        sobelY = cv2.convertScaleAbs(sobelY)
        sobelXY = cv2.convertScaleAbs(sobelXY)

        # # Show image
        # myImage.writeImage('Sobel X', sobelX)
        # myImage.writeImage('Sobel Y', sobelY)
        # myImage.writeImage('Sobel XY', sobelXY)

        plt.figure(2)
        plt.subplot(131)
        plt.imshow(sobelX, cmap='gray', interpolation='bicubic')
        plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
        plt.subplot(132)
        plt.imshow(sobelY, cmap='gray', interpolation = 'bicubic')
        plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
        plt.subplot(133)
        plt.imshow(sobelXY, cmap='gray', interpolation = 'bicubic')
        plt.title('Sobel XY'), plt.xticks([]), plt.yticks([])
        

    elif code == 2:
        # Prewitt edge detector
        blurImg = cv2.GaussianBlur(img, (5, 5), 0)

        #Init kernel
        kernelX = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        kernelY = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])

        prewittX = cv2.filter2D(blurImg, cv2.CV_64F, kernelX)
        prewittY = cv2.filter2D(blurImg, cv2.CV_64F, kernelY)
        # Compute magnitude
        prewittXY,_ = cv2.cartToPolar(prewittX, prewittY)

        # Convert image back to abs CV_8U
        prewittX = cv2.convertScaleAbs(prewittX)
        prewittY = cv2.convertScaleAbs(prewittY)
        prewittXY = cv2.convertScaleAbs(prewittXY)

        # # Show edges image
        # myImage.writeImage('Prewitt X', prewittX)
        # myImage.writeImage('Prewitt Y', prewittY)
        # myImage.writeImage('Prewitt XY', prewittXY)

        plt.figure(2)
        plt.subplot(131)
        plt.imshow(prewittX, cmap='gray', interpolation='bicubic')
        plt.title('Prewitt X'), plt.xticks([]), plt.yticks([])
        plt.subplot(132)
        plt.imshow(prewittY, cmap='gray', interpolation = 'bicubic')
        plt.title('Prewitt Y'), plt.xticks([]), plt.yticks([])
        plt.subplot(133)
        plt.imshow(prewittXY, cmap='gray', interpolation = 'bicubic')
        plt.title('Prewitt XY'), plt.xticks([]), plt.yticks([])
        


    elif code == 3:
        # Laplacian edge detector
        blurImg = cv2.GaussianBlur(img, (3, 3), 0)
        laplacian = cv2.Laplacian(blurImg, cv2.CV_64F)
        # Convert img back to CV_U8
        laplacian = cv2.convertScaleAbs(laplacian)
        # myImage.writeImage('Negative laplacian derivatives', laplacian)

        plt.figure(2)
        plt.imshow(laplacian, cmap='gray', interpolation='bicubic')
        plt.title('Negative laplacian'), plt.xticks([]), plt.yticks([])

    
    elif code == 4:
        # Canny edge detector
        lowThreshold, highThreshold = thresholdSeeking(img)
        
        edges = cv2.Canny(img, lowThreshold, highThreshold)
        # myImage.writeImage('Edge detection by OpenCv', edges)

        plt.figure(2)
        plt.imshow(edges, cmap='gray', interpolation='bicubic')
        plt.title('Canny edge detection'), plt.xticks([]), plt.yticks([])


    e2 = cv2.getTickCount()
    time = (e2 - e1)/cv2.getTickFrequency()
    print('Time: %.2f(s)' %(time))

    plt.show()

    cv2.waitKey(0)
    # Destroy all windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(args)
