#!/usr/bin/env python3

# OpenCV library
import cv2

# NumPy
import numpy

# Python math lib
import math

class DescriptorBase:
    """
    We can implement various functions that some descriptors have in common in here.

    One good use case of this inheritence is different distance functions for kn-means classifier.
    We can use Eucledian distance or SSD or something else. Implement it here
    and override the __sub__ function in your child class and call the method from this class.
    """
    def __init__(self, data=None, *args, **kwargs):
        if data is not None:
            self.data = data
        elif 'size' in kwargs:
            self.data = numpy.zeros(kwargs['size'])
        else:
            raise Exception("Either provide data or provide size of the descriptor as a kwarg.")

    def __mul__(self, other):
        raise Exception("Please override this function in the child class.")

    def __sub__(self, other):
        raise Exception("Please override this function in the child class.")

    def __str__(self):
        return repr(self.data)

    def __repr__(self):
        return repr(self.data)

    def sum_squared_difference(self, other):
        """
        Sum of squares differences.

        Example:
            sum_squared_difference(<a, b>, <c, d>) = sqrt(|a^2 - c^2|) + sqrt(|b^2 - d^2|)

        Returns a scalar.
        """
        # Make sure we are operating on descriptors of the same size
        assert len(self.data) == len(other.data)

        result = 0.0
        for i in range(len(self.data)):
            result += math.sqrt(math.fabs(math.pow(self.data[i], 2) - math.pow(other.data[i], 2)))

        return result
    
    def inner_product(self, other):
        """
        Inner product (generalization of dot product of vectors in n-dimensional space).

        Returns a scalar.
        """
        # Make sure we are operating on descriptors of the same size
        assert len(self.data) == len(other.data)

        result = 0.0
        for i in range(len(self.data)):
            result += self.data[i] * other.data[i]
        
        return result
