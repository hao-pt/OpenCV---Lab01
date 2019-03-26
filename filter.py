#Built-in lib
import numpy as np
import cv2
from skimage import exposure, img_as_ubyte, img_as_float64, img_as_uint
import math
#Built-out lib
import convolution as myconv

def gaussianXFunction(size, sigma):
        #Kernel radius
        kernelRadius = size // 2
        #x will range in [-kernelRadius, kernelRadius]
        x = np.array([range(-kernelRadius, kernelRadius + 1)], np.float64).reshape(size, 1)
        #Calculate gaussian kernel X by gaussian function
        twoSquareSigma = 2 * (sigma**2)
        x_2 = x*x
        gaussX = (1/ math.sqrt(math.pi*twoSquareSigma)) * np.exp(-x_2/twoSquareSigma)
        return gaussX

class CFilter:
    def __init__(self):
        #Gx: vertical
        #Gy: horizontal
        self.prewittKernel = {'Gx': np.array([[-1, 0, 1],
                                        [-1, 0, 1],
                                        [-1, 0, 1]], np.int),
                            'Gy' : np.array([[-1, -1, -1],
                            [0, 0, 0],
                            [1, 1, 1]], np.int)}

        self.sobelKernel = {'Gx': np.array([[-1, 0, 1]
                                ,[-2, 0, 2],
                                [-1, 0, 1]], np.int), 
                        'Gy': np.array([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], np.int)}
        
        self.weightedAvgKernel = np.array([[1, 2, 1],
                                            [2, 4, 2],
                                            [1, 2, 1]], np.float64) * 1.0/16
        self.meanKernel = np.ones((3, 3), np.int) * 1.0/9
        self.gaussKernelX = self.gaussKernelY = 0
    
    def smoothenImage(self, img, kernelName):
        conv = myconv.CMyConvolution()
        if kernelName == 'mean':
            conv.setKernel(self.meanKernel)
        elif kernelName == 'weighted avg':
            conv.setKernel(self.weightedAvgKernel)
        elif kernelName == 'gauss':
            #Convolve 2 times to reduce time complexity
            conv.setKernel(self.gaussKernelY)
            blurImg = conv.convolution(img)
            conv.setKernel(self.gaussKernelX)
            blurImg = conv.convolution(blurImg)
            return blurImg
            
        blurImg = conv.convolution(img)
        return blurImg

    def gaussianGenerator(self, size, sigma):
        #Separate gaussian into 2 direction: vertical & horizontal
        #Especially, both vertical & horizontal filter are symmetric
        #GaussX
        gaussX = gaussianXFunction(size, sigma)
        #GaussY is a transpose matrix of gaussX because they are symmetric
        gaussY = gaussX.transpose()

        #Normalize gaussian kernel by averaging
        sumX = np.sum(gaussX)
        sumY = np.sum(gaussY)
        #Round it 4 decimals
        gaussX = np.round(gaussX/sumX, 4)
        gaussY = np.round(gaussY/sumY, 4)
        self.gaussKernelX = gaussX
        self.gaussKernelY = gaussY

        return gaussX, gaussY

    def edgeDetector(self, img, kernelName):
        #Declare CMyConvolution() object
        conv = myconv.CMyConvolution()
        #Smoothen image to have blur the noise and the detail of edges
        img = self.smoothenImage(img, 'gauss')
        if kernelName == 'sobel':
            #Convole with vertical kernel
            conv.setKernel(self.sobelKernel['Gx'])
            verticalImage = conv.convolution(img)
            #Convole with horizontal kernel
            conv.setKernel(self.sobelKernel['Gy'])
            horizontalImage = conv.convolution(img)
            #Combine 2 vertical & horizontal image together to get magnitude of gradient at each point
            #|G| = sqrt(Gx^2 + Gy^2)
            magnitudeImg = np.zeros(img.shape, dtype = np.float64) #avoid out of range [0, 255]
            magnitudeImg = np.sqrt(np.power(verticalImage.astype(np.float64), 2) + np.power(horizontalImage.astype(np.float64), 2))
            
            #Normalize the output image to be in range [0, 255] accurately_
            #_when it's presented in float dtype [0, 1] called 'shrinking image'
            magnitudeImg = exposure.rescale_intensity(magnitudeImg, in_range=(0, 255))
            #Convert dtype of image back to uint8
            magnitudeImg = img_as_ubyte(magnitudeImg)
            return (verticalImage, horizontalImage, magnitudeImg)
            

        elif kernelName == 'prewitt':
            #Convole with vertical kernel
            conv.setKernel(self.prewittKernel['Gx'])
            verticalImage = conv.convolution(img)
            #Convole with horizontal kernel
            conv.setKernel(self.prewittKernel['Gy'])
            horizontalImage = conv.convolution(img)
            #Combine 2 vertical & horizontal image together to get magnitude of gradient at each point
            #|G| = sqrt(Gx^2 + Gy^2)
            #Combine 2 vertical & horizontal image together to get magnitude of gradient at each point
            magnitudeImg = np.zeros(img.shape, dtype = np.float64) #avoid out of range [0, 255]
            magnitudeImg = np.sqrt(np.power(verticalImage.astype(np.float64), 2) + np.power(horizontalImage.astype(np.float64), 2))
            
            #Normalize the output image to be in range [0, 255] accurately_
            #_when it's presented in float dtype [0, 1] called 'shrinking image'
            magnitudeImg = exposure.rescale_intensity(magnitudeImg, in_range=(0, 255))
            #Convert dtype of image back to uint8
            magnitudeImg = img_as_ubyte(magnitudeImg)
            return (verticalImage, horizontalImage, magnitudeImg)
        

        





