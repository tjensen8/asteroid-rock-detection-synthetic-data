# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 19:28:07 2021

@author: Taylor Jensen
for preprocessing of renderings - before breakout of the files
"""
from preprocess_generator import process_artificial_images

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