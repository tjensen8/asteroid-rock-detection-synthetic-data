import os
from skimage import transform
import cv2

import tensorflow.keras as keras
import tensorflow.keras.preprocessing.image as k_image

import numpy as np
import matplotlib.pyplot as plt

import preprocess_generator
from preprocess_generator import process_artificial_images

#%% Renderings
#input directory of rendered images
artificial_image_dir = 'D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/kaggle-rock/images/render/'

#output directories
output_dir = 'D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/processed-images/'
# specific subdirectories for the output
render_subdir = 'synthetic-all/renders/'
numpy_subdir = 'synthetic-all/tensor/render-tensors-individual/'

#process entire rendered images
process_artificial_images(
    img_dir=artificial_image_dir, 
    output_dir=output_dir, 
    render_dir=render_subdir, 
    numpy_dir=numpy_subdir,
    flip_flag=True
    )

#%% Masks
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