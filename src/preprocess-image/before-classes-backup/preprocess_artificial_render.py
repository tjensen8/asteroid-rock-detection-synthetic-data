# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 19:28:07 2021

@author: Taylor Jensen
for preprocessing of masks - before breakout of the files
"""
import os
from skimage import transform
import cv2

import tensorflow.keras as keras
import tensorflow.keras.preprocessing.image as k_image

import numpy as np
import matplotlib.pyplot as plt

from preprocess_generator import image_visual_processing, image_filter_processing

#input directories
artificial_image_dir = 'D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/kaggle-rock/images/'
artificial_image_renders_dir = artificial_image_dir+'render/'
artificial_image_masks_dir = artificial_image_dir+'ground/'

#output directories
output_dir = 'D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/processed-images/'
#output_dir_test = 'D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/processed-images/test'
#output_dir_train = 'D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/processed-images/train'
#output_dir_validation = 'D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/processed-images/validation'

#train_count = 6836
#test_count = 1953
#validate_count = 976


def process_artificial_images():
    render_image_files = os.listdir(artificial_image_renders_dir)
    
    for img in render_image_files:
        print(img)
        temp_image = k_image.load_img(artificial_image_renders_dir+img, color_mode='grayscale')
        temp_image = np.array(temp_image)
        temp_image_visual_processed = image_visual_processing(temp_image)
        #save partial processed image for review
        plt.imsave(output_dir+'synthetic-all/renders/'+img[:-4]+f'processed-visuals.png', temp_image_visual_processed)
        
        new_layers = image_filter_processing(temp_image_visual_processed)
        
        #empty array to fill out with additional filters, etc.
        img_processed = np.zeros((480,720,len(new_layers)+1))
        
        #add new channels to image
        img_processed[:,:,0] = temp_image
        for new_channel in range(1,len(new_layers)+1):
            #print(new_channel," Channel Added")
            img_processed[:,:,new_channel] = new_layers[new_channel-1]
            
            #save image example
            plt.imsave(output_dir+'synthetic-all/renders/'+img[:-4]+f'processed-channel-{new_channel}.png', new_layers[new_channel-1])
        
        #save out individual tensor 
        np.save(output_dir+f'synthetic-all/tensor/render-tensors-individual/{img[:-4]}-tensor.npy', img_processed)

    # need to flip images 
        flip_count = 3
        for i in range(1,flip_count+1):
            img_rotated = transform.rotate(image=img_processed, angle=90*i, resize=False)
            # save out flipped tensor
            np.save(output_dir+f'synthetic-all/tensor/render-tensors-individual/{img[:-4]}-flip{i}-tensor.npy', img_rotated)

process_artificial_images()