#!/usr/bin/env python3

# Operating System functions
import os

# OpenCV : Library to read images and videos that comes with a lot of AI algorithms
import cv2

# Library to create math plots and graphs. Can be used to show images
from matplotlib import pyplot as plt

from classifier import Classifier

# 
def show_image(image):
    """
    Useful for debugging. Opens the image in a tool.
    """
    if not image.data:
        print("You are trying to show an image that has no data.")
        exit()
    img = cv2.normalize(image.astype('float'), None, 0, 255, cv2.NORM_MINMAX)
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.show()

# 
def load_image(image):
    """
    Use this function to open images. It should return a multidimensional array of pixels.
    """
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    if not img.data:
        print("Could not open image %s. Maybe the path is wrong? \r\n Mac/Linux: ./image.jpg \r\nWindows: .\image.jpg")
    else:
        out = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        return out


def run():
    print("Welcome to the AI Project by Quentin, Dave, David, Vanessa, and Lev.")
    numImages = input("How many image sets you'd like to use? (min: 3, max 10): ")
    if int(numImages) < 3 or int(numImages) > 10:
        print("No! Won't do that. Bye!")
        exit()
    crossValidate = input("Do you want to use cross-validation for this run? If you select no, we will load the images sequentially.(Y/n): ")
    if crossValidate not in ['Y','y']:
        c = Classifier(int(numImages), False)
    else:
        c = Classifier(int(numImages), True)
    print("Done!")

run()
