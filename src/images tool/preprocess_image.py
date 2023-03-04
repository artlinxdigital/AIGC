'''
Artlinx - Step 2: Gather Data and Create Training Sets 
Artlinx is an AI-powered generative art tool that aims to enhance artist involvement in the creative process. This README explains how Artlinx demonstrates the second step of building an AIGC tool, which involves gathering data and creating training sets.

Prerequisites
Python 3.x
OpenCV (cv2) library
NumPy library
Getting Started
To get started, you will need to gather a significant amount of data to train the machine learning algorithm. The data should be diverse and cover various art styles and techniques. You can collect data from public art databases, galleries, or ask artists to contribute their work.

Once you have collected and organized the image data in separate directories, you can use the preprocess_image() function in gather_data.py to read and preprocess the images by resizing, converting to RGB format, and normalizing the pixel values.
'''

import os 
import cv2
import numpy as np

# Define function to read and preprocess image data
def preprocess_image(image_path, size=(256, 256)):
    # Load image and resize to desired dimensions
    img = cv2.imread(image_path)
    img = cv2.resize(img, size)
    # Convert from BGR to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize pixel values to range [0, 1]
    img = img / 255.0
    return img