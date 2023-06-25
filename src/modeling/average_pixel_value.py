# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 21:34:27 2021

@author: Taylor Jensen

Gather the min/max  pixel values across the training set because this brightness is what is fed to the ML model.
All images will have to be scaled to this.

"""
#%%
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

from sklearn.preprocessing import MinMaxScaler

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
#%%
#positive/negative
maximum_pixel_value = []
minimum_pixel_value = []
average_pixel_value = []
for f in input_img_paths:
    print(f)
    img = load_img(f, color_mode='grayscale')
    img = img_to_array(img)
    
    # add max, min to array
    maximum_pixel_value.append(np.max(img))
    minimum_pixel_value.append(np.min(img))
    average_pixel_value.append(np.mean(img))

    #fit the min and maximum for later scaling
    """
    f_row_shape = img[0].shape[0]
    scaler = MinMaxScaler
    for img_row in range(f_row_shape):
        scaler.partial_fit(X=f[0][img_row])
    """
#%%
print("The Average Maximum Pixel Value is:")
print(np.mean(maximum_pixel_value))

print("The Average Minimum Pixel Value:")
print(np.mean(minimum_pixel_value))

print("The Average of Average Pixel Value:")
print(np.mean(average_pixel_value))

# %%
plt.figure(figsize=[5,5])
plt.hist(maximum_pixel_value)
plt.title("Distribution of Maximum Pixel Values")
plt.xlabel("Value of Pixel")
plt.ylabel("Count of Occurances")
plt.show()

plt.figure(figsize=[5,5])
plt.hist(minimum_pixel_value)
plt.title("Distribution of Minimum Pixel Values")
plt.xlabel("Value of Pixel")
plt.ylabel("Count of Occurances")
plt.show

# %%
print("Overall Maximum Pixel Value Across Whole Training Set")
print(np.max(maximum_pixel_value))

print("Overall Minimum Pixel Value Across Whole Training Set")
print(np.min(minimum_pixel_value))


# %%
