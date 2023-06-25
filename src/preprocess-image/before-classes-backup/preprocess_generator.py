# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 15:36:43 2021

@author: Taylor Jensen

Script for image-agnostic processing and filtering.
Will be used for the bennu images and for the artificial renders, masks.

"""
import numpy as np
from skimage import io, filters, feature, measure, exposure, transform
from skimage.morphology import skeletonize, thin, reconstruction, disk

import tensorflow.keras as keras
import tensorflow.keras.preprocessing.image as k_image

import matplotlib.pyplot as plt

import os
import random
# preprocess images and add to new directory

# directory for image data synced with NAS
# D:\TJPersonalCloud\Programming\msds-thesis-data

#preprocessing to do
""" For each image in the mask and redered image 
1. process both the mask and the rendered image at the same time 

Masks
2. use the ground datasets for masks -- done
3. mask: make blue/green masks (large/small rocks) the same class (green - small rocks) -- done
4. mask: make red mask (sky) a part of the black class (ground) -- done
5. mask: fill empty spots in rocks (look at the reference notebook in kaggle for this) -- done
6. make the mask a single channel of rock or not rock -- done
7. save visualization of changes so it can be added to paper -- done
8. perform any random image flipping on the same images as the renders


Renders
2. perform any random image flipping
3. add estimated sensor noise/gaussian noise
4. blur images randomly
-- add filters as additional image input layers (start with 3)
5. perform edge detection
6. perform blob detection
7. perform skeletonization 
8. remove tiny objects via top hat filter
9. save visualization of different filters/actions for all images so it can be added to paper
10. take render and apply the neccesary filters to it and additive channels 

2. save images in their mathmatical representation as a single file

assign image to either train, test, or validation - randomly split?
"""

# need to do flipping and other pixel-placement adjustments to both mask and render image

### DO IMAGE FLIPPING AFTER ALL OTHER PROCESSING HAS BEEN COMPLETED 

def gather_files(directory):
    # get all files from a directory and their path for iteration
    files = os.listdir(directory)
    return files

def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def image_visual_processing(image_array):
    # for flips or anything that changes how the image looks and would effect a mask
    
    # replace sky with mean value
    min_pixel = image_array.min()
    mean_pixel = np.round(image_array.mean(),0)

    
    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            if image_array[i,j] < min_pixel+5:
                image_array[i,j] = mean_pixel
                image_array[i-1,j] = mean_pixel
                image_array[i,j-1] = mean_pixel
                image_array[i-2,j] = mean_pixel
                image_array[i,j-2] = mean_pixel
            else:
                next
  
    #add gaussian noise - approximate bad sensors or other
    image_array = sp_noise(image_array, 0.01) # 1% chance of noise in each pixel
    
    #add gentle image blur to adjust out the noise 
    image_array = filters.gaussian(image_array, sigma=(2,2), truncate=1, multichannel=None) #size of 2x2 blue, effect of blur trunecated a 1 standard deviation
    
    # plt.imshow(image_array, cmap='Greys')
    # plt.title('Preprocessed Image')
    # plt.show()
    
    return image_array

def image_filter_processing(image_array):
    # for filters and any additional image representations or image processing

    #edge - the sigma is used for the gaussian blur filter
    edge = feature.canny(image_array, sigma=2)
    
    # plt.imshow(edge)
    # plt.title('Canny Edge Detection, Sigma=2')
    # plt.show()
    
    
    # sobel edge detection
    edge_sorbel = filters.sobel(image_array)
    # plt.imshow(edge_sorbel)
    # plt.title('Sobel Edge Detection')
    # plt.show()
    
    # local binary pattern detection
    binary_pattern = feature.local_binary_pattern(image_array,2,1)
    # plt.imshow(binary_pattern)
    # plt.title('Local Binary Pattern Detection')
    # plt.show()

    
    # # gradient of image
    # gradient_img = filters.rank.gradient(image_array, disk(1))
    # plt.imshow(gradient_img)
    # plt.title('Image Gradient, Disk=1 radius')
    # plt.show()
    
    return edge, edge_sorbel, binary_pattern


def processing(image_dir):
    
    #image_dir = 'D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/'
    #gather_files(image_dir+'kaggle-rock/images/render')
    
    
    image = k_image.load_img(image_dir+'kaggle-rock/images/render/render9766.png', color_mode='grayscale')
    #image_mask = k_image.load_img(image_dir+'kaggle-rock/images/ground/ground0006.png')
    
    image_array = np.array(image)
    #image_mask_array = np.array(image_mask)
    
    #image_mask_array.shape
    #image_array.shape
    
    #original image
    #plt.imshow(image_array, cmap='Greys')
    #plt.title("Original Image")
    #plt.show()
    
    image_array = image_visual_processing(image_array)
    edge, edge_sorbel, binary_pattern = image_filter_processing(image_array)

    #normalizing of image required before final model processing
    
    return edge, edge_sorbel, binary_pattern



