"""
Python file for utility functions used across model implementations. ACTUALLY WORKS
"""
from tensorflow import keras
import tensorflow as tf
import numpy as np
import os 
from skimage import io, filters

from sklearn.preprocessing import MinMaxScaler
import preprocess_generator
# no flipping, no additional edge detection layers, only greyscale, binary detection

#%% Make sequence to load batches of the data
# https://keras.io/examples/vision/oxford_pets_image_segmentation/
class artificial_lunar_landscapes(keras.utils.Sequence):
    """
    Iterates over the data in given batches for consumption by model.
    """

    def __init__(self, batch_size, image_size, image_processed_size, input_img_paths, target_img_paths, input_img_processed_paths):
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_processed_size = image_processed_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.input_img_processed_paths = input_img_processed_paths

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
        batch_input_img_processed_paths = self.input_img_processed_paths[i : i + self.batch_size]
        # make x to be filled out with image data to train on
        x = np.zeros((self.batch_size,) + self.image_size)
        #bring in tensor of image, match it to provided size, and fill out our x for training
        ######## X
        
        for j, path in enumerate(batch_input_img_paths):
            img = keras.preprocessing.image.load_img(path, color_mode='grayscale')
            img = keras.preprocessing.image.img_to_array(img)
            img = img.copy().astype('uint8')
            
            #replace sky with mean value
            min_pixel = img.min()
            mean_pixel = np.round(img.mean(),0)
            img[img <min_pixel+5] = mean_pixel
            
            #additional filter processing
            #img_processed = filters.sobel(img[:,:,0])
            #img_processed = np.expand_dims(img_processed, axis=-1)

            img = img/255.
            #img_processed = img_processed/255.
            
            #add image to list for deep learning algorithm
            x[j] = img
            

        x_processed = np.zeros((self.batch_size,) + self.image_processed_size)
        for j, path in enumerate(batch_input_img_processed_paths):
            img_processed = np.load(path)
            img_processed = img_processed/255.
            x_processed[j] = img_processed
        
        ###### Y
        y = np.zeros((self.batch_size,) + self.image_size)
        for j, path in enumerate(batch_target_img_paths):
            img = keras.preprocessing.image.load_img(path)
            img = keras.preprocessing.image.img_to_array(img)
            img = img.copy().astype('uint8')
            
            # set anything relatively bright to 255, anything else is dark
            img[img>=25] = 255
            img[img<25] = 0
            
            #merge layers into single array
            single_array = img[:,:,1] + img[:,:,2]
            # make layers maximum and minimum (in case of over 255)
            single_array[single_array>0] = 255
            single_array[single_array==0] = 0

            #change 255 to 1 in order to normalize image pixel values
            #single_array[single_array>=255] = 1
            single_array = single_array/255.
            #add the needed dimension
            single_array = np.expand_dims(single_array, axis=-1)
            #print("Y ARRAY SHAPE: ",single_array.shape)
            y[j] = single_array
        return [x,x_processed],y

def generate_image_paths(
    processed_dir='D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/train/render-processed',
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

    process_img_paths = sorted(
        [os.path.join(processed_dir, fname) for fname in os.listdir(processed_dir)]
    )

    print("Number of images: ", len(input_img_paths))
    print("Number of processed: ", len(process_img_paths))
    print("Number of targets: ", len(target_img_paths))
    

    return input_dir, target_dir, input_img_paths, target_img_paths, process_img_paths