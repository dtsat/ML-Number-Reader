import unittest
from base import DescriptorBase
import numpy
import math

class baseTest(unittest.TestCase):

    def test_init(self):
        descriptor = DescriptorBase(size=25)
        self.assertTrue(len(descriptor.data) == 25)

    def test_mul_forbidden(self):
        try:
            descriptor = DescriptorBase(size=10)
            descriptor * descriptor
            self.fail('Multiply operator should not work in base class.')
        except:
            pass
    
    def test_sub_forbidden(self):
        try:
            descriptor = DescriptorBase(size=10)
            descriptor - descriptor
            self.fail('Subtract operator should not work in base class.')
        except:
            pass

    def test_sum_squared_difference(self):
        a1 = numpy.array([1, 2, 3])
        a2 = numpy.array([1, 2, 2])

        d1 = DescriptorBase(a1)
        d2 = DescriptorBase(a2)

        dif = d1.sum_squared_difference(d2)
        self.assertTrue(dif == (math.sqrt(5)))

    def test_inner_product(self):
        a1 = numpy.array([1, 2, 3])
        a2 = numpy.array([1, 2, 2])

        d1 = DescriptorBase(a1)
        d2 = DescriptorBase(a2)

        dif = d1.inner_product(d2)
        self.assertTrue(dif == (1 * 1 + 2 * 2 + 3 * 2))

if __name__ == '__main__':
    unittest.main()