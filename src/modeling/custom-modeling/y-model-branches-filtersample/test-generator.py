import pandas as pd
import numpy as np
from model_utils import artificial_lunar_landscapes, generate_image_paths

import matplotlib.pyplot as plt

batch_size = 1
image_size = (480,720,1)
image_processed_size = (480,720,1)

#training generator
input_dir, target_dir, input_img_paths, target_img_paths = generate_image_paths(
    target_dir='D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/train/mask-og', 
    input_dir='D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/train/render-og'
)
train_generator = artificial_lunar_landscapes(
    batch_size=batch_size,
    image_size=image_size,
    image_processed_size=image_processed_size,
    input_img_paths=input_img_paths,
    target_img_paths=target_img_paths
)


#test generator
input_dir, target_dir, input_img_paths, target_img_paths = generate_image_paths(
    target_dir='D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/test/mask-og', 
    input_dir='D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/test/render-og'
)
test_generator = artificial_lunar_landscapes(
    batch_size=batch_size,
    image_size=image_size,
    image_processed_size=image_processed_size,
    input_img_paths=input_img_paths,
    target_img_paths=target_img_paths
)

for i in train_generator:
    print(i[0][0].shape)
    plt.imshow(i[0][0][0,:,:,:])
    plt.show()
    print(i[0][1].shape)
    plt.imshow(i[0][1][0,:,:,:])
    plt.show()
    #print(i[1])