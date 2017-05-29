#!/usr/bin/env python3

#OpenCV
import cv2
import os
import random


class Importer:

	def __init__(self, directory, crossValidate=True, offset=10):
		"""
		Paramters:
			-- directory: Where the images are located
		"""
		self.directory = directory
		self.learn = []
		self.test = []
		self.validate = []
		#used to retrieve images name of cluster center
		self.grabImagesName = []
		self.__load(crossValidate, offset)

	def __cross_validate_sets(self, files, offset=10):
		"""
		Load learn, test, and validate sets randomly.
		"""
		print("Importing files using cross-validation with offset: ", offset, '\n')
		
		fileCount = len(files)
		
		# Divide into learn, validate and test sets
		learn = len(files) / 100 * 64
		test = len(files) / 100 * 20
		validate = len(files) / 100 * 16 # Validation set = size of test set (20% each)
		counter = 0
		for index in range(len(files)):
			idx = (index + offset) % fileCount
			file = files[idx]
			# full path
			path = self.directory + '/' + file
			# Load image with OpenCV
			image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

			# Test that data was loaded successfully
			if image.data is None:
				print("Error reading image")
			else:
				# Put into a set
				if counter < learn:
					self.learn.append(image)
				elif counter < learn + validate:
					self.validate.append(image)
				else:
					self.test.append(image)
			counter += 1
		
		
	# Two underscores = private method
	def __load(self, crossValidate=False, offset=10):
		"""
		Loads the images using OpenCV (grayscale).
		Supports loading images in random order and in sequential order.
		
		Parameters:
			-- crossValidate : randomly select images for each set
		"""
		# List the files
		files = os.listdir(self.directory)

		# Randomly sample images
		if (crossValidate):
			self.__cross_validate_sets(files, offset)
			return

		# Sample files sequentially

		# Divide into learn, validate and test sets
		learn = len(files) / 100 * 64
		test = len(files) / 100 * 20
		validate = len(files) / 100 * 16 # Validation set = size of test set (20% each)
		
		counter = 0
		for file in files:
			# full path
			path = self.directory + '/' + file

			#Used to retrieve image name on cluster centers
			images = {}
			images[file] = counter
			self.grabImagesName.append(images)

			# Load image with OpenCV
			image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

			# Test that data was loaded successfully
			if image.data is None:
				print("Error reading image")
			else:
				# Put into a set
				if counter < learn:
					self.learn.append(image)
				elif counter < learn + validate:
					self.validate.append(image)
				else:
					self.test.append(image)
			counter += 1