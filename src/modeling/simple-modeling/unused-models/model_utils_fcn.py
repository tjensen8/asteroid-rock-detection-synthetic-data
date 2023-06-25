"""
Python file for utility functions used across model implementations.
"""
from tensorflow import keras
import tensorflow as tf
import numpy as np
import os 

from sklearn.preprocessing import MinMaxScaler
# no flipping, no additional edge detection layers, only greyscale, binary detection

#%% Make sequence to load batches of the data
# https://keras.io/examples/vision/oxford_pets_image_segmentation/
class artificial_lunar_landscapes(keras.utils.Sequence):
    """
    Iterates over the data in given batches for consumption by model.
    """

    def __init__(self, batch_size, image_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.image_size = image_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        length = len(self.target_img_paths) // self.batch_size
        return length
    
    def __getitem__(self, idx):
        """
        Returns tuple (input, target) that matches with the given batch size.
        """
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        # make x to be filled out with image data to train on
        x = np.zeros((self.batch_size,) + self.image_size)
        #bring in tensor of image, match it to provided size, and fill out our x for training
        ######## X
        for j, path in enumerate(batch_input_img_paths):
            img = keras.preprocessing.image.load_img(path)
            img = keras.preprocessing.image.img_to_array(img)
            img = keras.preprocessing.image.smart_resize(img, (480, 704)) #resize to fit vgg 16 multiple of 32 requirement
            img = img/255.
            x[j] = img
        
        ###### Y
        y = np.zeros((self.batch_size,) + self.image_size)
        for j, path in enumerate(batch_target_img_paths):
            img = keras.preprocessing.image.load_img(path)
            img = keras.preprocessing.image.img_to_array(img)
            img = keras.preprocessing.image.smart_resize(img, (480, 704)) #resize to fit vgg 16 multople of 32 requirement
            # set anything relatively bright to 255, anything else is dark
            img[img>=25] = 255
            img[img<25] = 0

            img[:,:,0] = 0
            #merge layers into single array
            #single_array = img[:,:,1] + img[:,:,2]
            single_array = img
            # make layers maximum and minimum (in case of over 255)
            single_array[single_array>0] = 255
            single_array[single_array==0] = 0

            #change 255 to 1 in order to normalize image pixel values
            #single_array[single_array>=255] = 1
            single_array = single_array/255.
            #add the needed dimension
            #single_array = np.expand_dims(single_array, axis=-1)
                        
            y[j] = single_array
        return x,y

def generate_image_paths(
    input_dir='D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/train/render-processed-numpy', 
    target_dir='D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/train/mask-processed-numpy'
    ):

    # prepare paths of input images and target segmentation masks

    #renders directory
    input_dir = input_dir
    #masks directory
    target_dir = target_dir

    #sort images so they will be processed together correctly
    input_img_paths = sorted(
        [os.path.join(input_dir, fname) for fname in os.listdir(input_dir)]
    )

    target_img_paths = sorted(
        [os.path.join(target_dir, fname) for fname in os.listdir(target_dir)]
    )

    print("Number of images: ", len(input_img_paths))
    print("Number of targets: ", len(target_img_paths))

    return input_dir, target_dir, input_img_paths, target_img_paths