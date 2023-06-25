from tensorflow import keras
import tensorflow as tf
import numpy as np
import os 

from sklearn.preprocessing import MinMaxScaler

from model_utils import artificial_lunar_landscapes, generate_image_paths

import matplotlib.pyplot as plt


img = keras.preprocessing.image.load_img('D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/train/render-og/render0002.png', color_mode='grayscale')
img = keras.preprocessing.image.img_to_array(img)
img.shape
img = img/255
x[j] = img



image_size = (480, 720, 1)
#num_classes = 2 
batch_size = 1


#training generator
input_dir, target_dir, input_img_paths, target_img_paths = generate_image_paths(
    target_dir='D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/train/mask-og', 
    input_dir='D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/train/render-og'
)
train_generator = artificial_lunar_landscapes(
    batch_size=batch_size,
    image_size=image_size,
    input_img_paths=input_img_paths,
    target_img_paths=target_img_paths
)



for x,y in train_generator:
    print('x')
    plt.imshow(x[0])
    plt.title("x")
    plt.show()
    print('y')
    plt.imshow(y[0])
    plt.title("y")
    plt.show()
    