import numpy as np
import cv2

class CMyConvolution:
    #Ham khoi tao
    def __init__(self):
        #Height & width of kernel
        self.kH, self.kW = 0
        #Init kernel by 3x3 matrix with 0-element
        self.kernel = np.zeros((3, 3))
    #Gan kernel voi 1 mask cho truoc
    def setKernel(self, mask):
        self.kW, self.kH
    def convolution(self, img):
        #Padding de output image keep the same size as input image
        #Tinh padding them vao
        p = (self.kW - 1) // 2
        #Padding
        paddingImg = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_REPLICATE)

        #Size of img
        iH, iW = img.shape

        #Output image
        outImg = np.zeros(img.shape, dtype = np.float64)

        #Scan image without padding. Cause center of kernel slide in each pixel of image
        for y in range(p, iH):
            for x in range(p, iW):
                for
                #Take sliding matrix that have same size as kernel
                paddingImg[iH + ]  