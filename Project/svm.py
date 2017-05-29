#!/usr/bin/env python3

# OpenCV library
import cv2

# NumPy
import numpy

# Python math lib
import math

# Science Kit SVM classifier
from sklearn.svm import SVC

# Implemented common functions in the base class so we can re-use them for other descriptors
# documented in other papers
from base import DescriptorBase

from importer import Importer

#
class SVMDescriptor(DescriptorBase):
    """
    Descriptor documented in
    `Application of Support Vector Machines for Recognition of Handwritten Arabic/Persian Digits`
    paper.
    """
    def __init__(self, data=None):
        """
        Constructor.

        Parameters:
            data (optional) -- numpy.array of size 64
        """
        if data is None:
            super(SVMDescriptor, self).__init__(size=64)
        else:
            assert len(data) == 64
            super(SVMDescriptor, self).__init__(data)

    def __sub__(self, other):
        """
        Sum of squared differences (SSD)
        Returns a scalar.

        Example:
        Given <a, b> and <c, d> vectors,
        result = sqrt(|a*a - c*c|) + sqrt(|b*b - d*d|)

        This can be generalized into any n-dimensional vector. In our case, we have a 64-dimensional vector.
        """
        return self.sum_squared_difference(other)

    def __mul__(self, other):
        """
        Inner product.
        Works the same as the dot product except in n-dimensional vectors like this descriptor.
        Returns a scalar.
        """
        return self.inner_product(other)

    def __rmul__(self, other):
        """
        Scalar multiplication of the descriptor.

        Identical to vector multiplication:
            <a, b, c> * d = <a*d, b*d, c*d>

        Returns a descriptor.
        """
        result = Descriptor(self.data)
        for i in range(64):
            result.data[i] *= other
        
        return result

    def __str__(self):
        return repr(self.data)

class SVMFeatureExtractor:
    """
    Feature extractor documented in 
    `Application of Support Vector Machines for Recognition of Handwritten Arabic/Persian Digits`
    paper.
    """
    def __init__(self):
        """
        Constructor

        Parameters:
            image -- numpy.array created by OpenCV (cv2.imread), grayscale only
        """
        self.__white = 0.99 # To be adjusted

    def __str__(self):
        return "SVMFeatureExtractor"

    def __resize(self):
        """
        Resizes the image to 64x64 as per specification.
        """
        if self.image.shape != (64, 64):
            self.resized = cv2.resize(self.image, (64, 64))
        else:
            self.resized = self.image

    def __left(self):
        """
        Left function
        """
        result = numpy.zeros(64)
        for y in range(64):
            count = 0
            for x in range(64):
                pixel = self.resized[y][x]
                if pixel < self.__white:
                    break
                else:
                    count += 1
            result[y] = count
        return result

    def __right(self):
        """
        Right function
        """
        result = numpy.zeros(64)
        for y in range(64):
            count = 0
            for x in reversed(range(64)):
                pixel = self.resized[y][x]
                if pixel < self.__white:
                    break
                else:
                    count += 1
            result[y] = count
        return result

    def __top(self):
        """
        Top function
        """
        result = numpy.zeros(64)
        for x in range(64):
            count = 0
            for y in range(64):
                pixel = self.resized[y][x]
                if pixel < self.__white:
                    break
                else:
                    count += 1
            result[x] = count
        return result

    def __bottom(self):
        """
        Bottom function
        """
        result = numpy.zeros(64)
        for x in range(64):
            count = 0
            for y in reversed(range(64)):
                pixel = self.resized[y][x]
                if pixel < self.__white:
                    break
                else:
                    count += 1
            result[x] = count
        return result


    def __derivatives(self):
        """
        Computes the derivatives for each function using OpenCV
        """
        top = self.__top()
        left = self.__left()
        right = self.__right()
        bottom = self.__bottom()

        kernel = numpy.array([-1, 0, 1]) # Derivatives kernel

        # Compute derivatives for each of the 4 functions
        dtop = numpy.array([x[0] for x in cv2.filter2D(top, -1, kernel)])
        dleft = numpy.array([x[0] for x in cv2.filter2D(left, -1, kernel)])
        dright =  numpy.array([x[0] for x in cv2.filter2D(right, -1, kernel)])
        dbottom = numpy.array([x[0] for x in cv2.filter2D(bottom, -1, kernel)])

        return dtop, dright, dbottom, dleft

    
    def __descriptor(self):
        """
        Creates a descriptor by sampling 1/4 value in each derivative.
        """
        dtop, dright, dbottom, dleft = self.__derivatives()
        descriptor = []
        # 1/4 sampling rate
        for x in range(0, 64, 4):
            descriptor.append(dtop[x])
            descriptor.append(dright[x])
            descriptor.append(dbottom[x])
            descriptor.append(dleft[x])
        return numpy.array(descriptor)


    def run(self, image):
        """
        Public method for this class.

        Usage:
            extractor = SVMFeatureExtractor(image)
            descriptor = extractor.run()
        """
        self.image = image
        self.__resize()
        return SVMDescriptor(self.__descriptor())




















