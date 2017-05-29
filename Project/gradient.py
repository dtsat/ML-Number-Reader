import numpy
import cv2
from base import DescriptorBase 
import math

class Gradient(DescriptorBase): 
    def __init__(self, data=None):
        """
        Constructor.

        Parameters:
            data (optional) -- numpy.array of 225
        """
        if data is None:
            super(Gradient, self).__init__(size=144)
        else:
            assert len(data) == 144
            super(Gradient, self).__init__(data)
    
class GradientFeatureExtractor: 
    def __init__(self):
        """
        Constructor
        """
        pass

    def __del__(self):
        """
        Destructor
        """
        pass

    def __str__(self):
        """
        toString(), like in Java
        """
        return "GradientFeatureExtractor"

    def __resize(self):
        self.resized = cv2.resize(self.image,(42, 42))
 
    def ___gradient(self): 
        #entire table filled with 0s 42 x 42
        self.entire = numpy.zeros((42,42)) 

        self.resized = cv2.resize(self.image,(42, 42))


        # #sobel x  
        # sobelx64f = cv2.Sobel(self.resized,cv2.CV_32F,1,0,ksize=3)

        # #sobel y
        # sobely64f = cv2.Sobel(self.resized,cv2.CV_32F,0,1,ksize=3)

        kernelx = numpy.array([0, 0, 0, -1, 0, 1, 0, 0, 0])
        kernely = numpy.array([0, -1, 0, 0, 0, 0, 0, 1, 0])

        dx = cv2.filter2D(self.resized, -1, kernelx)
        dy = cv2.filter2D(self.resized, -1, kernely)


        self.magnitude = numpy.zeros((42,42))
        self.direction = numpy.zeros((42,42))
  
        # apply the magnitude formula and direction
        for x in range(42):
            for y in range(42): 
                self.magnitude[x][y] = math.sqrt(math.pow(dx[x][y], 2) + math.pow(dy[x][y], 2))
                self.direction[x][y] = math.atan2(dy[x][y], dx[x][y])


        descriptor = []
        for x in range(0, 42, 7):
            for y in range(0, 42, 7):
                histogram = self.__histogram(x, y)
                descriptor = descriptor + histogram
        return numpy.array(descriptor)


    def __rad_to_bin(self, rad):
        """
        Maps [-pi, pi] --> [0, 4]
        """
        if rad < 0:
            return math.floor((-rad / math.pi) * 2)
        else:
            return 1 + math.floor((rad / math.pi) * 2)


    def __histogram(self, row, col):
        histogram = [0, 0, 0, 0]
        for x in range(row, row+7):
            for y in range(col, col+7):
                histogram[self.__rad_to_bin(self.direction[x][y])] += self.magnitude[x][y]
        return histogram
                
 

    def run(self, image):
        """
        Parameters:
            image -- OpenCV image
        """

        self.image = image

        if image.shape != (42, 42):
            self.__resize()
        return Gradient(self.___gradient())
