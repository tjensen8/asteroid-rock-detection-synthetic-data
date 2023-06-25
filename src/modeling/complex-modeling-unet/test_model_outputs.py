import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, array_to_img, save_img
import numpy as np
import os 

import matplotlib.pyplot as plt
import seaborn as sns

os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"]="2"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

image_size = np.load('D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/processed-images/synthetic-all/tensor/render-tensors-individual/render4884-tensor.npy').shape
num_classes = 1 
batch_size = 2

model = keras.models.load_model('D:/ProgrammingD/github/thesis/modeling/unet/checkpoints/model.01-0.27-0.49.h5')
model.summary()

#renders directory
input_dir = 'D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/train/render-processed-numpy' 
#masks directory
target_dir = 'D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/train/mask-processed-numpy' 

#
image_size = (480, 720)
num_classes = 1 
batch_size = 2

#sort images so they will be processed together correctly
input_img_paths = sorted(
    [os.path.join(input_dir, fname) for fname in os.listdir(input_dir)]
)

target_img_paths = sorted(
    [os.path.join(target_dir, fname) for fname in os.listdir(target_dir)]
)

print("Number of images: ", len(input_img_paths))
print("Number of targets: ", len(target_img_paths))



img = np.load('D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/processed-images/synthetic-all/tensor/render-tensors-individual/render0556-tensor.npy')
img.shape = (1,)+ img.shape
img = img / 255
sns.distplot(img)

prediction = model.predict(img)
sns.distplot(prediction[0])
plt.imshow(prediction[0])

save_img('test-img.png',array_to_img(prediction.shape[0]))


img = np.load('D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/processed-images/synthetic-all/tensor/mask-tensors-individual/ground0556-tensor.npy')
img[img>=0.5] = 1
img[img<0.5] = 0
plt.imshow(img)

sns.distplot(img)
