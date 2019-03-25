import numpy as np
import cv2
from skimage.exposure import rescale_intensity

class CMyConvolution:
    #Ham khoi tao
    def __init__(self):
        #Height & width of kernel
        self.kH = self.kW = 0
        #Init kernel by 3x3 matrix with 0-element
        self.kernel = np.zeros((3, 3))

    #Gan kernel voi 1 mask cho truoc
    def setKernel(self, mask):
        self.kW, self.kH = mask.shape
        self.kernel = mask
    
    #Flip filter
    def flipFilter(self):
        return self.kernel[::-1, ::-1]

    def convolution(self, img):
        #Padding de output image keep the same size as input image
        #Tinh padding them vao
        p = (self.kW - 1) // 2
        #Padding
        paddingImg = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_REPLICATE)

        #Size of img
        iH, iW = img.shape

        #Output image
        #Use dtype = np.float64 because avoiding out of range [0, 255] when convoling
        outImg = np.zeros(img.shape, dtype = np.float64)

        #flip filter then multiply element-wise
        flipKernel = self.flipFilter()

        #Scan image without padding. Cause center of kernel slide in each pixel of image
        for y in range(p, iH):
            for x in range(p, iW):
                #Extract the roi (Region of interesting) that have same size as kernel
                roi = paddingImg[y - p: y + p + 1, x-p: x + p + 1]

                #Element-wise multiplication of roi & kernel then get the sum of it 
                # to get the convolve output
                k = (roi * flipKernel).sum()

                #Assign this convole output to pixel (y, x) of output image
                #Note: without padding
                #Use method: .itemset of numpy to speed up modify pixel in image 
                outImg.itemset((y - p, x - p), k)
        
        #Normalize the output image to be in range [0, 255]
        outImg = rescale_intensity(outImg, in_range=(0, 255))
        #Convert output image's dtype back to uint8. Because its dtype still float64
        outImg = (outImg * 255).astype(np.uint8)

        return outImg




        
        


