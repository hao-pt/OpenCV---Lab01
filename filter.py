#Built-in lib
import numpy as np
import cv2
from skimage.exposure import rescale_intensity
#Built-out lib
import convolution as myconv

class CFilter:
    def __init__(self):
        #Gx: vertical
        #Gy: horizontal
        self.prewittKernel = {'Gx': np.array([[-1, 0, 1],
                                        [-1, 0, 1],
                                        [-1, 0, 1]], np.int8),
                            'Gy' : np.array([[-1, -1, -1],
                            [0, 0, 0],
                            [1, 1, 1]], np.int8)}

        self.sobelKernel = {'Gx': np.array([[-1, 0, 1]
                                ,[-2, 0, 2],
                                [-1, 0, 1]], np.int8), 
                        'Gy': np.array([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], np.int8)}
    
    def edgeDetector(self, img, kernelName):
        conv = myconv.CMyConvolution()
        if kernelName == 'sobel':
            #Convole with vertical kernel
            conv.setKernel(self.sobelKernel['Gx'])
            verticalImage = conv.convolution(img)
            #Convole with horizontal kernel
            conv.setKernel(self.sobelKernel['Gy'])
            horizontalImage = conv.convolution(img)
            #Combine 2 vertical & horizontal image together to get magnitude of gradient at each point
            magnitudeImg = np.zeros(img.shape, dtype = np.float64) #avoid out of range [0, 255]
            magnitudeImg = np.sqrt(np.power(verticalImage, 2) + np.power(horizontalImage, 2))

        elif kernelName == 'prewitt':
            #Convole with vertical kernel
            conv.setKernel(self.prewittKernel['Gx'])
            verticalImage = conv.convolution(img)
            #Convole with horizontal kernel
            conv.setKernel(self.prewittKernel['Gy'])
            horizontalImage = conv.convolution(img)
            #Combine 2 vertical & horizontal image together to get magnitude of gradient at each point
            #|G| = sqrt(Gx^2 + Gy^2)
            magnitudeImg = np.zeros(img.shape, dtype = np.float64) #avoid out of range [0, 255]
            magnitudeImg = np.sqrt(np.power(verticalImage, 2) + np.power(horizontalImage, 2))

        #Normalize magnitudeImg to be in range [0, 255]
        magnitudeImg = rescale_intensity(magnitudeImg, in_range=(0, 255))
        #Convert dtype of image back to uint8
        magnitudeImg = (magnitudeImg * 255).astype(np.uint8)
        return [verticalImage, horizontalImage, magnitudeImg]

        





