"""
To preprocess images for training. That way the preprocessing doesn't take up so much of the CPU.
"""

import numpy as np
from skimage import io, filters
from tensorflow import keras
import os

def gather_files(directory):
    # get all files from a directory and their path for iteration
    files = os.listdir(directory)
    return files
# train directory

directory = 'D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/train/render-og/'

for path in gather_files(directory):
    print(path[:-4])
    img = keras.preprocessing.image.load_img(directory+path, color_mode='grayscale')
    img = keras.preprocessing.image.img_to_array(img)
    img = filters.sobel(img)
    np.save(f'D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/train/render-processed/{path[:-4]}',img)

#test directory
directory = 'D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/test/render-og/'

for path in gather_files(directory):
    print(path[:-4])
    img = keras.preprocessing.image.load_img(directory+path, color_mode='grayscale')
    img = keras.preprocessing.image.img_to_array(img)
    img = filters.sobel(img)
    np.save(f'D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/test/render-processed/{path[:-4]}',img)