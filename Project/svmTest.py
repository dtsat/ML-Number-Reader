"""
Run this to test the features extractors in features.py
"""
# Python unit testing
import unittest

# OpenCV
import cv2

# Features extractor
from svm import SVMFeatureExtractor

class featuresTest(unittest.TestCase):
    """
    Tests meaningful methods in features extractors implemented in features.py
    """
    def setUp(self):
        """
        Loads the image.
        """
        # Load the image with only one channel: grayscale
        self.image = cv2.imread("image3.png", cv2.IMREAD_GRAYSCALE)

        # Normalize the grayscale 0 - 255 intensity range to 0.0 to 1.0 floating point range
        # for extra precision
        self.image = cv2.normalize(self.image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

        self.image2 = cv2.imread("image2.png", cv2.IMREAD_GRAYSCALE)
        self.image2 = cv2.normalize(self.image2.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    def test_sub_descriptors(self):
        """
        Tests subsctraction of descriptors.
        """
        # Extractor
        extractor = SVMFeatureExtractor()

        # Run it
        descriptor = extractor.run(self.image)

        # Run it again with another image
        descriptor2 = extractor.run(self.image2)

        print("Difference of two descriptors: ", descriptor2 - descriptor)

        # Descriptor minus another descriptor should not give 0
        self.assertTrue(descriptor2 - descriptor != 0.0)

        # Print the descriptor
        print("Descriptor: \r\n", descriptor)

    def test_mul_descriptors(self):
        """
        Tests multiplication of descriptors (inner product).
        """
        # Extractor
        extractor = SVMFeatureExtractor()

        # Run it
        descriptor = extractor.run(self.image)

        # Copy
        copy = descriptor

        # Multiply
        result = copy * descriptor

        # Print the result
        print("Multiplication of two descriptors: ", result)

    def test_rmul_descriptor_scalar(self):
        """
        Tests scalar multiplication.
        """
        pass
   

if __name__ == '__main__':
    unittest.main()