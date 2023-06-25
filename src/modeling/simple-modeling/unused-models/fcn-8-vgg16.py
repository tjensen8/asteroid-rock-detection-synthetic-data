"""
Fully connected network - FCN 8 from:

https://github.com/divamgupta/image-segmentation-keras

overview here: https://towardsdatascience.com/review-fcn-semantic-segmentation-eb8c9b50d2d1

"""

from tensorflow.random import set_seed
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, concatenate, Input, UpSampling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.distribute import MirroredStrategy, HierarchicalCopyAllReduce

import numpy as np
from model_utils_fcn import artificial_lunar_landscapes, generate_image_paths
import os

from keras_segmentation.models.fcn import fcn_32_vgg


from keras.applications import vgg16
from keras.models import Model, Sequential
from keras.layers import Conv2D, Conv2DTranspose, Input, Cropping2D, add, Dropout, Reshape, Activation

import time

set_seed(0)

os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"]="2"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

IMAGE_ORDERING='channels_last'

#set distributed training
strategy = MirroredStrategy(cross_device_ops=HierarchicalCopyAllReduce())

with strategy.scope():

    def weighted_bincrossentropy(true, pred, weight_zero = 0.25, weight_one = 1):
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



    def FCN8_helper(nClasses, input_height, input_width):

        assert input_height % 32 == 0
        assert input_width % 32 == 0

        img_input = Input(shape=(input_height, input_width, 3))

        model = vgg16.VGG16(
            include_top=False,
            weights='imagenet', input_tensor=img_input,
            pooling=None,
            classes=1000)
        assert isinstance(model, Model)

        o = Conv2D(
            filters=4096,
            kernel_size=(
                7,
                7),
            padding="same",
            activation="relu",
            name="fc6")(
                model.output)
        o = Dropout(rate=0.5)(o)
        o = Conv2D(
            filters=4096,
            kernel_size=(
                1,
                1),
            padding="same",
            activation="relu",
            name="fc7")(o)
        o = Dropout(rate=0.5)(o)

        o = Conv2D(filters=nClasses, kernel_size=(1, 1), padding="same", activation="relu", kernel_initializer="he_normal",
                name="score_fr")(o)

        o = Conv2DTranspose(filters=nClasses, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation=None,
                            name="score2")(o)

        fcn8 = Model(inputs=img_input, outputs=o)
        # mymodel.summary()
        return fcn8


    def FCN8(nClasses, input_height, input_width):

        fcn8 = FCN8_helper(nClasses, input_height, input_width)

        # Conv to be applied on Pool4
        skip_con1 = Conv2D(nClasses, kernel_size=(1, 1), padding="same", activation=None, kernel_initializer="he_normal",
                        name="score_pool4")(fcn8.get_layer("block4_pool").output)
        Summed = add(inputs=[skip_con1, fcn8.output])

        x = Conv2DTranspose(nClasses, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation=None,
                            name="score4")(Summed)

        ###
        skip_con2 = Conv2D(nClasses, kernel_size=(1, 1), padding="same", activation=None, kernel_initializer="he_normal",
                        name="score_pool3")(fcn8.get_layer("block3_pool").output)
        Summed2 = add(inputs=[skip_con2, x])

        #####
        Up = Conv2DTranspose(nClasses, kernel_size=(8, 8), strides=(8, 8),
                            padding="valid", activation=None, name="upsample")(Summed2)

        Up = Reshape((-1, nClasses))(Up)
        Up = Activation("sigmoid")(Up)

        mymodel = Model(inputs=fcn8.input, outputs=Up)

        return mymodel

    image_size = (480, 704, 3)
    #num_classes = 2 
    batch_size = 3

    model = FCN8(nClasses=1, input_height=480,  input_width=704)

    model.compile(optimizer = Adam(learning_rate= 0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy', # OG learning rate from code source is 1e-4
        keras.metrics.MeanIoU(num_classes=2), keras.metrics.Recall(),keras.metrics.Precision()])
    print("Model Compiled")

    # for model checkpoints
    root_checkpointdir = os.path.join(os.curdir,"simple-modeling","fcn-8-vgg", "checkpoints")

    # for tensorflow logging
    root_logdir = os.path.join(os.curdir,"simple-modeling","fcn-8-vgg", "tflogs")

    def run_logdir(root_logdir):
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(root_logdir, run_id)

    # for model checkpoints
    root_checkpointdir = os.path.join(os.curdir,"simple-modeling","fcn-8-vgg", "checkpoints")

    def run_checkpointdir(root_checkpointdir):
        run_id = 'model.{epoch:02d}-{val_loss:.2f}-{mean_io_u:.2f}.h5'
        return os.path.join(root_checkpointdir, run_id)

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

    callbacks = [
    keras.callbacks.ModelCheckpoint(run_checkpointdir(root_checkpointdir), mode='max', monitor='mean_io_u', save_freq='epoch', save_best_only=True),
    keras.callbacks.TensorBoard(run_logdir(root_logdir), histogram_freq=1, write_images=True, update_freq='batch'),
    ]

    model.fit(
        train_generator, 
        epochs=20, # 25 was originally selected - would take 48 hours
        callbacks=callbacks, 
        verbose=2,
        validation_data=(test_generator)) #validation results outputted after an epoch

