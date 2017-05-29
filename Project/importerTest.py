#!/usr/bin/env python3

from importer import Importer

import unittest

class importerTest(unittest.TestCase):
	def setUp(self):
		self.directory = 'images/1'

	def test_import_images(self):
		importer = Importer(self.directory)
		self.assertTrue(len(importer.learn) > 0)
		self.assertTrue(len(importer.test) > 0 and len(importer.test) < len(importer.learn))
		self.assertTrue(len(importer.validate) > 0)
		print("Learn set: ", len(importer.learn))
		print("Validate set: ", len(importer.validate))
		print("Test set: ", len(importer.test))

if __name__ == '__main__':
	unittest.main()