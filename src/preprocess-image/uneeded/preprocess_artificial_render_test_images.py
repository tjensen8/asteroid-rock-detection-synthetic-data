# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 19:28:07 2021

@author: Taylor Jensen
for preprocessing of renderings - applied to test dataset
"""
from preprocess_generator import process_artificial_images

#input directory of rendered images
artificial_image_dir = 'D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/test/render-og/'

#output directories
output_dir = 'D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/test/'
# specific subdirectories for the output
render_subdir = 'render-processed-images/'
numpy_subdir = 'render-processed-numpy/'

#process test images
process_artificial_images(
    img_dir=artificial_image_dir, 
    output_dir=output_dir, 
    render_dir=render_subdir, 
    numpy_dir=numpy_subdir,
    flip_flag=False
    )