# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 15:36:43 2021

@author: Taylor Jensen

Script for image-agnostic processing and filtering.

"""
import numpy as np
from skimage import io, filters, feature, measure, exposure, transform, util
from skimage.morphology import skeletonize, thin, reconstruction, disk

import tensorflow.keras as keras
import tensorflow.keras.preprocessing.image as k_image

import matplotlib.pyplot as plt
import cv2

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
#use if we want images to be shrunk
target_size = None


#%% Preprocessing Rendered Images
def gather_files(directory):
    # get all files from a directory and their path for iteration
    files = os.listdir(directory)
    return files

def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
#    output = np.zeros(image.shape,np.uint8)
#    thres = 1 - prob 
#    for i in range(image.shape[0]):
#        for j in range(image.shape[1]):
#            rdn = random.random()
#            if rdn < prob:
#                output[i][j] = 0
#            elif rdn > thres:
#                output[i][j] = 255
#            else:
#                output[i][j] = image[i][j]
    output = util.random_noise(image=image, mode='s&p',seed=0, amount=prob)
    return output

def image_visual_processing(image_array):
    # for flips or anything that changes how the image looks and would effect a mask
    
    # replace sky with mean value
    min_pixel = image_array.min()
    mean_pixel = np.round(image_array.mean(),0)

    
#    for i in range(image_array.shape[0]):
#        for j in range(image_array.shape[1]):
#            if image_array[i,j] < min_pixel+5:
#                image_array[i,j] = mean_pixel
#                image_array[i-1,j] = mean_pixel
#                image_array[i,j-1] = mean_pixel
#                image_array[i-2,j] = mean_pixel
#                image_array[i,j-2] = mean_pixel
#            else:
#                next
    image_array[image_array <min_pixel+5] = mean_pixel

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
    #edge = feature.canny(image_array, sigma=2)   
    
    # sobel edge detection
    edge_sorbel = filters.sobel(image_array)
    
    # local binary pattern detection
    #binary_pattern = feature.local_binary_pattern(image_array,2,1)
    
    return edge_sorbel#, binary_pattern #edge


def process_artificial_images(img_dir, output_dir, render_dir, numpy_dir, flip_flag=True):
    """
    This is the actual preprocessing of a given directory.
    """
    print("Start Render List")
    render_image_files = os.listdir(img_dir)
    print("Render List Complete")
    for img in render_image_files:
        print(img)
        temp_image = k_image.load_img(img_dir+img, color_mode='grayscale', target_size=target_size)
        temp_image = np.array(temp_image)
        
        #save original image to new directory
        plt.imsave(output_dir+render_dir+img[6:10]+f'original-image.png', temp_image)
        np.save(output_dir+numpy_dir+f'{img[6:10]}-original-tensor.npy', img_rotated)

        #process image
        temp_image_visual_processed = image_visual_processing(temp_image)
        #save partial processed image for review
        plt.imsave(output_dir+render_dir+img[6:10]+f'processed-visuals.png', temp_image_visual_processed)
        
        new_layers = image_filter_processing(temp_image_visual_processed)
        
        #empty array to fill out with additional filters, etc.
        img_processed = np.zeros((480,720,len(new_layers)+1))
        
        #add new channels to image
        img_processed[:,:,0] = temp_image
        for new_channel in range(1,len(new_layers)+1):
            #print(new_channel," Channel Added")
            img_processed[:,:,new_channel] = new_layers[new_channel-1]
            
            #save image example
            plt.imsave(output_dir+render_dir+img[6:10]+f'processed-channel-{new_channel}.png', new_layers[new_channel-1])
        
        #save out processed individual image tensor 
        np.save(output_dir+numpy_dir+f'{img[6:10]}-processed-tensor.npy', img_processed)

        if flip_flag == True:
        # need to flip images 
            flip_count = 3
            for i in range(1,flip_count+1):
                img_rotated = transform.rotate(image=img_processed, angle=90*i, resize=False)
                # save out flipped tensor
                np.save(output_dir+numpy_dir+f'{img[6:10]}-flip{i}-processed-tensor.npy', img_rotated)

#%% For Mask Processing
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
#    for channel in range(mask_image_array.shape[2]):
#        for i in range(mask_image_array.shape[0]):
#            for j in range(mask_image_array.shape[1]):
#                if mask_image_array[i,j, channel] > 0:
#                    mask_image_array[i,j, channel] == 255
#                if mask_image_array[i,j, channel] < 0:
#                    mask_image_array[i,j, channel] == 0
    mask_image_array[mask_image_array>=25] = 255
    mask_image_array[mask_image_array<25] = 0
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


def breakout_mask_images(artificial_image_masks_dir, output_dir, render_dir, numpy_dir, flip_flag=True):
    """
    Breaks out the different classifications of the original masks, removes sky, and makes rocks all the same mask.
    """
    
    mask_image_files = os.listdir(artificial_image_masks_dir)
    
    print(f"Total Mask Images: {len(mask_image_files)}")
    for image in mask_image_files:
        print(image)
        mask_image_dir = artificial_image_masks_dir+image
        mask_image = k_image.load_img(mask_image_dir, target_size=target_size)
        mask_image_array = np.array(mask_image)
        
        #save original image
        plt.imsave(output_dir+render_dir+image[6:10]+'-original-image.png', mask_image_array, cmap='Greys')
        np.save(output_dir+numpy_dir+f'{image[6:10]}-original-tensor.npy', mask_image_array)
        
        #process image
        mask_image_array = process_mask(mask_image_array)

        #save processed image
        plt.imsave(output_dir+render_dir+image[6:10]+'-processed.png', mask_image_array, cmap='Greys')
        # save out individual tensor
        np.save(output_dir+numpy_dir+f'{image[6:10]}-tensor.npy', mask_image_array)
        
        #rotate image 3 times and add to our list as new images
        if flip_flag == True:
            flip_count = 3
            for i in range(1,flip_count+1):
                mask_rotated = transform.rotate(image=mask_image_array, angle=90*i, resize=False)
                #save processed image
                plt.imsave(output_dir+render_dir+image[6:10]+f'processed-rotate{i}.png', mask_rotated, cmap='Greys')
                #save flipped tensor
                np.save(output_dir+numpy_dir+f"{image[6:10]}-flip{i}-tensor.npy", mask_rotated)

