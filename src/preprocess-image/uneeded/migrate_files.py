# -*- coding: utf-8 -*-
"""
@author: Taylor Jensen
Identify and move the train, test, and validation files to their respective directories.
"""
from identify_file_splits import Namesplit
import os
import tensorflow.keras.preprocessing.image as k_image
# INCOMPLETE 

def migrate_to_directories(data, in_dir, out_base_dir='./'):
    """
    Moves the files that have been split to their respective directories, given a base directory.
    Directories are ./train, ./test, and ./validation - defaulted to the current directory.
    """
    all_image_names = os.listdir(in_dir)
    #migrate and preprocess
    for key in data.keys():
        if 'train' in key:
            images_to_iter = data[key]
            for image in images_to_iter:
                image_to_migrate = k_image.load_img(in_dir+image)
                
    
        
if __name__ == '__main__':
    
    #directories to find images to split
    input_images_dir = 'D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/kaggle-rock/images'
    mask_dir = input_images_dir+'/ground/'
    render_dir = input_images_dir+'/render/'

    #split X and Y files into training lists
    data = Namesplit(mask_dir, render_dir)
    #save the results of the lists
    data_dict = data.save_lists(
        'D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/processed-images/image-assignments/',
        separate_files = True
        )

    #move the lists to their specified directories
    out_base_dir = 'D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/processed-for-model-training/'
    migrate_to_directories(data=data_dict, in_dir=render_dir, out_base_dir=out_base_dir)