"""
Baseline unet model with weighted binary cross entropy.
"""
#import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
from tensorflow.random import set_seed
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, concatenate, Input, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.distribute import MirroredStrategy, HierarchicalCopyAllReduce
from tensorflow.keras import backend as K
import tensorflow
import numpy as np
from model_utils import artificial_lunar_landscapes, generate_image_paths
#from model_utils_y_model_copy import artificial_lunar_landscapes, generate_image_paths
import os
import gc
import time

set_seed(0)

os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"]="2"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

#set distributed training
strategy = MirroredStrategy(cross_device_ops=HierarchicalCopyAllReduce())
# https://github.com/huanglau/Keras-Weighted-Binary-Cross-Entropy/blob/master/DynCrossEntropy.py
def dyn_weighted_bincrossentropy(true, pred):
    """
    Calculates weighted binary cross entropy. The weights are determined dynamically
    by the balance of each category. This weight is calculated for each batch.
    
    The weights are calculted by determining the number of 'pos' and 'neg' classes 
    in the true labels, then dividing by the number of total predictions.
    
    For example if there is 1 pos class, and 99 neg class, then the weights are 1/100 and 99/100.
    These weights can be applied so false negatives are weighted 99/100, while false postives are weighted
    1/100. This prevents the classifier from labeling everything negative and getting 99% accuracy.
    
    This can be useful for unbalanced catagories.
    """
    # get the total number of inputs
    num_pred = keras.backend.sum(keras.backend.cast(pred < 0.5, true.dtype)) + keras.backend.sum(true)
    
    # get weight of values in 'pos' category
    zero_weight =  keras.backend.sum(true)/ num_pred +  keras.backend.epsilon() 
    
    # get weight of values in 'false' category
    one_weight = keras.backend.sum(keras.backend.cast(pred < 0.5, true.dtype)) / num_pred +  keras.backend.epsilon()

    # calculate the weight vector
    weights =  (1.0 - true) * zero_weight +  true * one_weight 
    
    # calculate the binary cross entropy
    bin_crossentropy = keras.backend.binary_crossentropy(true, pred)
    
    # apply the weights
    weighted_bin_crossentropy = weights * bin_crossentropy 

    return keras.backend.mean(weighted_bin_crossentropy)


def weighted_bincrossentropy(true, pred, weight_zero = 0.1, weight_one = 1):
    """
    Calculates weighted binary cross entropy. The weights are fixed.
        
    This can be useful for unbalanced catagories.
    
    Adjust the weights here depending on what is required.
    
    For example if there are 10x as many positive classes as negative classes,
        if you adjust weight_zero = 1.0, weight_one = 0.1, then false positives 
        will be penalize 10 times as much as false negatives.
    """
  
    # calculate the binary cross entropy
    bin_crossentropy = keras.backend.binary_crossentropy(true, pred)
    
    # apply the weights
    weights = true * weight_one + (1. - true) * weight_zero
    weighted_bin_crossentropy = weights * bin_crossentropy 

    return keras.backend.mean(weighted_bin_crossentropy)


def unet(input_size = (256,256,1)):
    with strategy.scope():
        inputs = Input(shape=input_size)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = -1)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = -1)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = -1)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = -1)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

        model = Model(inputs=inputs,outputs=conv10)
        print(model.summary())
        print("Compile Model")
        # https://keras.io/api/optimizers/adam/
        model.compile(optimizer = Adam(learning_rate= 0.0001), loss = weighted_bincrossentropy, metrics = ['accuracy', # OG learning rate from code source is 1e-4
            keras.metrics.MeanIoU(num_classes=2), keras.metrics.Recall(),keras.metrics.Precision()])
        print("Model Compiled")
    return model


image_size = (480, 720, 1)
#num_classes = 2 
batch_size = 2



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


#test generator
input_dir, target_dir, input_img_paths, target_img_paths = generate_image_paths(
    target_dir='D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/test/mask-og', 
    input_dir='D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/test/render-og'
)
test_generator = artificial_lunar_landscapes(
    batch_size=batch_size,
    image_size=image_size,
    input_img_paths=input_img_paths,
    target_img_paths=target_img_paths
)

# for tensorflow logging
root_logdir = os.path.join(os.curdir,"modeling","results","simple-modeling","unet-weighted-bce-trainval", "tflogs-normalize")

def run_logdir(root_logdir):
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

# for model checkpoints
root_checkpointdir = os.path.join(os.curdir,"modeling","results","simple-modeling","unet-weighted-bce-trainval", "checkpoints-normalize")

def run_checkpointdir(root_checkpointdir):
    run_id = 'model.{epoch:02d}-{loss:.2f}-{accuracy:.2f}.h5'
    return os.path.join(root_checkpointdir, run_id)

class MyCustomCallback(tensorflow.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        keras.backend.clear_session()
        gc.collect()

callbacks = [
keras.callbacks.ModelCheckpoint(run_checkpointdir(root_checkpointdir), mode='max', monitor='accuracy', save_freq='epoch', save_best_only=False),
keras.callbacks.TensorBoard(run_logdir(root_logdir), histogram_freq=1, write_images=True, update_freq='batch'),
MyCustomCallback()
]

model = unet(input_size=image_size)

model.fit(
    train_generator, 
    epochs=20, # 25 was originally selected - would take 48 hours
    callbacks=callbacks, 
    verbose=1,
    validation_data=(test_generator)) #validation results outputted after an epoch
