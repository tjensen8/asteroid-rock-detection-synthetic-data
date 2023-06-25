"""

Custom model approach with branches to handle original image and images that are preprocessed

"""
from tensorflow.random import set_seed
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, concatenate, Input, UpSampling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.distribute import MirroredStrategy, HierarchicalCopyAllReduce
import tensorflow as tf
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
with strategy.scope():
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
            if you adjust weight_zero = 1.0, weight_one = 0.1, then false positives # predict rock but actually no rock
            will be penalize 10 times as much as false negatives. # predict no rock but actually rock
        """
    
        # calculate the binary cross entropy
        bin_crossentropy = keras.backend.binary_crossentropy(true, pred)
        
        # apply the weights
        weights = true * weight_one + (1. - true) * weight_zero
        weighted_bin_crossentropy = weights * bin_crossentropy 

        return keras.backend.mean(weighted_bin_crossentropy)


    def dl_model(input_size = (256,256,1), input_processed_size=(256,256,1)):

        input_img = Input(shape=input_size)
        input_processed = Input(shape=input_processed_size)
        
        # regular image input
        conv1a = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_img)
        pool1a = MaxPooling2D(pool_size=(2, 2))(conv1a)
        conv2a = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1a)
        pool2a = MaxPooling2D(pool_size=(2, 2))(conv2a)
        conv3a = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2a)
        #pool3a = MaxPooling2D(pool_size=(2, 2))(conv3a)

        #preprocessed image input
        conv1b = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_processed)
        pool1b = MaxPooling2D(pool_size=(2, 2))(conv1b)
        conv2b = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1b)
        pool2b = MaxPooling2D(pool_size=(2, 2))(conv2b)
        conv3b = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2b)
        #pool3b = MaxPooling2D(pool_size=(2, 2))(conv3b)

        #compress high level info
        high_info = concatenate([conv1a, conv1b], axis=-1)
        high1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(high_info)
        
        #compress medium level info and upscale
        med_info = concatenate([conv2a, conv2b], axis=-1)
        med1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(med_info)
        med_up1 = Conv2DTranspose(filters=32, kernel_size=(2,2), strides=2, activation='relu')(med1)
        
        #compress low level info and upscale
        low_info = concatenate([conv3a, conv3b], axis=-1)
        low1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(low_info)
        low_up1 = Conv2DTranspose(filters=32, kernel_size=(2,2), strides=4, activation='relu')(low_info)

        #final compression
        final_info = concatenate([high1, med_up1, low_up1],axis=-1)
        pre_out = Conv2D(50,1,padding='same', activation='relu')(final_info)
        pre_out = Conv2D(10,1,padding='same', activation='relu')(final_info)
        out = Conv2D(1,1,padding='same', activation='sigmoid')(pre_out)
        model = Model(inputs=[input_img, input_processed],outputs=out)

        return model


    image_size = (480, 720, 1)
    input_processed_size = (480, 720, 1)
    #num_classes = 2 
    batch_size = 2


    #training generator
    input_dir, target_dir, input_img_paths, target_img_paths, process_img_paths = generate_image_paths(
        processed_dir='D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/train/render-processed',
        target_dir='D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/train/mask-og', 
        input_dir='D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/train/render-og'
    )
    train_generator = artificial_lunar_landscapes(
        batch_size=batch_size,
        image_size=image_size,
        image_processed_size = input_processed_size,
        input_img_paths=input_img_paths,
        target_img_paths=target_img_paths,
        input_img_processed_paths=process_img_paths
    )


    #test generator
    input_dir, target_dir, input_img_paths, target_img_paths, process_img_paths = generate_image_paths(
        processed_dir='D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/test/render-processed',
        target_dir='D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/test/mask-og', 
        input_dir='D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/test/render-og'
    )
    test_generator = artificial_lunar_landscapes(
        batch_size=batch_size,
        image_size=image_size,
        image_processed_size = input_processed_size,
        input_img_paths=input_img_paths,
        target_img_paths=target_img_paths,
        input_img_processed_paths=process_img_paths
    )

    # for tensorflow logging
    root_logdir = os.path.join(os.curdir,"modeling","results","custom-modeling","y-model-branches-deep", "tflogs")

    def run_logdir(root_logdir):
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(root_logdir, run_id)

    # for model checkpoints
    root_checkpointdir = os.path.join(os.curdir,"modeling","results","custom-modeling","y-model-branches-deep", "checkpoints","traintestrun")

    def run_checkpointdir(root_checkpointdir):
        run_id = 'model.{epoch:02d}-{loss:.2f}-{accuracy:.2f}.h5'
        return os.path.join(root_checkpointdir, run_id)



    model = dl_model(input_size=image_size, input_processed_size=input_processed_size)

    print(model.summary())
    print("Compile Model")
    # https://keras.io/api/optimizers/adam/
    model.compile(optimizer = Adam(learning_rate= 0.0001, beta_1=0.99), loss = weighted_bincrossentropy, metrics = ['accuracy', # OG learning rate from code source is 1e-4
        keras.metrics.MeanIoU(num_classes=2), keras.metrics.Recall(),keras.metrics.Precision()])
    print("Model Compiled")

    class MyCustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()

callbacks = [
keras.callbacks.ModelCheckpoint(run_checkpointdir(root_checkpointdir), mode='max', monitor='accuracy', save_freq='epoch', save_best_only=False),
keras.callbacks.TensorBoard(run_logdir(root_logdir), histogram_freq=1, write_images=False, update_freq='batch'),
MyCustomCallback()
]
model.fit(
    train_generator, 
    epochs=20, 
    callbacks=callbacks, 
    verbose=1)
    #validation_data=(test_generator)) #validation results outputted after an epoch
