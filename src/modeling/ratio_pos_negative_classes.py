# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 21:34:27 2021

@author: Taylor Jensen
"""

import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

import matplotlib.pyplot as plt

def generate_image_paths(
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

    print("Number of images: ", len(input_img_paths))
    print("Number of targets: ", len(target_img_paths))

    return input_dir, target_dir, input_img_paths, target_img_paths




input_dir, target_dir, input_img_paths, target_img_paths = generate_image_paths(
    target_dir='D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/train/mask-og', 
    input_dir='D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/train/render-og'
)

#positive/negative
ratio = []
for f in target_img_paths:
    print(f)
    img = load_img(f)
    img = img_to_array(img)
    # set anything relatively bright to 255, anything else is dark
    img[img>=25] = 255
    img[img<25] = 0
    
    #merge layers into single array
    single_array = img[:,:,1] + img[:,:,2]
    # make layers maximum and minimum (in case of over 255)
    single_array[single_array>0] = 1
    single_array[single_array==0] = 0
    ratio.append(len(single_array[single_array==1])/len(single_array[single_array==0]))
    
print("The Averate Ratio of Positive To Negative Pixels is:")
print(np.mean(ratio))
    