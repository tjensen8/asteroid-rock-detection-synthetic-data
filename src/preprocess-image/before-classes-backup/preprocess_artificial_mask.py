# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 15:36:43 2021

@author: Taylor Jensen

Apply preprocessing generator to all synthetic images. 
The breakout of all of the processed synthetic images and conversion to tensors 
    is done in train-test-validation-split dataset
"""

""" Move original artificial rock images to folder 
- Copy and past images 
"""

import os
from skimage import transform
import cv2

import tensorflow.keras as keras
import tensorflow.keras.preprocessing.image as k_image

import numpy as np
import matplotlib.pyplot as plt

#input directories
artificial_image_dir = 'D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/kaggle-rock/images/'
artificial_image_masks_dir = artificial_image_dir+'ground/'

#output directories
output_dir = 'D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/processed-images/'



def process_mask(mask_image_array):
    """
    Takes a 3 channel image array masks and combines it into a unified mask that identifies
    all rocks, regardless if they are labeled 'large' or 'small' in the original dataset.
    
    Also attempts to fill in any errors in the masks to make the masks consistent.
    

    Parameters
    ----------
    image_mask_array : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """    
    
    #items farther away are 'darker', this is making all rocks equally bright in their channels
    for channel in range(mask_image_array.shape[2]):
        for i in range(mask_image_array.shape[0]):
            for j in range(mask_image_array.shape[1]):
                if mask_image_array[i,j, channel] > 0:
                    mask_image_array[i,j, channel] == 255
                if mask_image_array[i,j, channel] < 0:
                    mask_image_array[i,j, channel] == 0
    #merge into single array
    single_array = mask_image_array[:,:,1] + mask_image_array[:,:,2]
    
    # make sure any pixels that are over 255 are set to 255
    for i in range(single_array.shape[0]):
        for j in range(single_array.shape[1]):
            if single_array[i,j] > 255:
                single_array[i,j] == 255
  
    # apply fillter to rocks to close holes in rocks
    kernel = np.ones((1,1), np.uint8)
    closing = cv2.morphologyEx(single_array, cv2.MORPH_CLOSE, kernel)
    
    image_mask_array = closing
    
    return image_mask_array


def breakout_mask_images():
    
    mask_image_files = os.listdir(artificial_image_masks_dir)
    
    print(f"Total Mask Images: {len(mask_image_files)}")
    for image in mask_image_files:
        print(image)
        mask_image_dir = artificial_image_masks_dir+image
        mask_image = k_image.load_img(mask_image_dir)
        mask_image_array = np.array(mask_image)
        mask_image_array = process_mask(mask_image_array)

        #save processed image
        plt.imsave(output_dir+'synthetic-all/images/'+image[:-4]+'-processed.png', mask_image_array, cmap='Greys')
        # save out individual tensor
        np.save(output_dir+f'synthetic-all/tensor/mask-tensors-individual/{image[:-4]}-tensor.npy', mask_image_array)
        
        #rotate image 3 times and add to our list as new images
        flip_count = 3
        for i in range(1,flip_count+1):
            mask_rotated = transform.rotate(image=mask_image_array, angle=90*i, resize=False)
            #save processed image
            plt.imsave(output_dir+'synthetic-all/images/'+image[:-4]+f'processed-rotate{i}.png', mask_rotated, cmap='Greys')
            #save flipped tensor
            np.save(output_dir+f'synthetic-all/tensor/mask-tensors-individual/{image[:-4]}-flip{i}-tensor.npy', mask_rotated)
            
        print(f'Processed')
    

breakout_mask_images()

    