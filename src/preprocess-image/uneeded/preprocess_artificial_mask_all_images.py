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

import preprocess_generator

#input directories
artificial_image_dir = 'D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/kaggle-rock/images/'
artificial_image_masks_dir = artificial_image_dir+'ground/'

#output directories
output_dir = 'D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/processed-images/synthetic-all'
render_subdir = 'masks/'
numpy_subdir = 'tensor/'
    
preprocess_generator.breakout_mask_images(
    artificial_image_masks_dir=artificial_image_masks_dir, 
    output_dir=output_dir, 
    render_dir=render_subdir,
    numpy_dir=numpy_subdir,
    flip_flag=True)

    