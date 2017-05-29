#!/usr/bin/env python3
from importer import Importer
from svm import SVMFeatureExtractor
from base import DescriptorBase
import cv2
import math
import numpy

class KMeansDescriptor(DescriptorBase):
	def __init__(self, data=None):
		"""
		Constructor.

		Parameters:
			data (optional) -- numpy.array of size 64
		"""
		if data is None:
			super(KMeansDescriptor, self).__init__(size=144)
		else:
			assert len(data) == 144
			super(KMeansDescriptor, self).__init__(data)


class KMeansFeatureExtractor:

	def __init__(self):
		pass

	def __str__(self):
		return "KMeansFeatureExtractor"

	def __resize(self):
		self.resized = cv2.resize(self.image, (42, 42))

	def __gradient(self):
		kernelY = numpy.array([0, -1, 0, 0, 0, 0, 0, 1, 0])
		kernelX = numpy.array([0, 0, 0, -1, 0, 1, 0, 0, 0])

		# Compute Y gradient
		self.yGradient = cv2.filter2D(self.resized, -1, kernelY)
		self.xGradient = cv2.filter2D(self.resized, -1, kernelX)

	def __orientation_and_magnitude(self):
		self.o = numpy.zeros((42, 42))
		self.m = numpy.zeros((42, 42))
		for y in range(42):
			for x in range(42):
				# Do not divide by 0
				if (self.yGradient[x][y] == 0):
					continue
				self.m[x][y] = math.sqrt(math.pow(self.xGradient[x][y], 2) + math.pow(self.yGradient[x][y], 2))
				self.o[x][y] = math.atan2(self.yGradient[x][y], self.xGradient[x][y])
	

	def __rad_to_bucket(self, rad):
		if (rad < 0):
			return math.floor((-rad / math.pi) * 2)
		else:
			return 2 + math.floor((rad / math.pi) * 2)

	def __histogram(self, row, col):
		histogram = [0, 0, 0, 0]
		for y in range(col, col+7):
			for x in range(row, row+7):
				bucket = self.__rad_to_bucket(self.o[x][y])
				histogram[bucket] += self.m[x][y]
		return histogram

	def __descriptor(self):
		descriptor = []
		for y in range(0, 42, 7):
			for x in range(0, 42, 7):
				histogram = self.__histogram(x, y)
				descriptor = descriptor + histogram
		return numpy.array(descriptor)


	def run(self, image):
		self.image = image
		if image.shape != (42, 42):
			self.__resize()
		self.__gradient()
		self.__orientation_and_magnitude()
		return KMeansDescriptor(self.__descriptor())