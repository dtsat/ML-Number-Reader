#!/usr/bin/env python3

from kmeans import KMeansDescriptor, KMeansFeatureExtractor
import unittest
import cv2

class kmeansTest(unittest.TestCase):
    
    def setUp(self):
        # Load the image with only one channel: grayscale
        self.image = cv2.imread("image3.png", cv2.IMREAD_GRAYSCALE)

        # Normalize the grayscale 0 - 255 intensity range to 0.0 to 1.0 floating point range
        # for extra precision
        self.image = cv2.normalize(self.image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

        self.image2 = cv2.imread("image2.png", cv2.IMREAD_GRAYSCALE)
        self.image2 = cv2.normalize(self.image2.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    def test_descriptor(self):
        extractor = KMeansFeatureExtractor()
        descriptor = extractor.run(self.image)
        print(descriptor)

if __name__ == '__main__':
    unittest.main()