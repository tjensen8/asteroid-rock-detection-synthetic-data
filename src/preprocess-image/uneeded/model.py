"""
Test file for a test image sementation model.
"""
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import os 

import matplotlib.pyplot as plt

from model_utils import artificial_lunar_landscapes, generate_image_paths
# create a custom generator to load the dataset in batches from the disk to the network for training
os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"]="2"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# evaluate mask r-cnn architecture -- too complex for what we are looking at. It covers image instance segmentation (like how many rocks and then identify rocks)
# we are looking for semantic segmentation - that is, ROCK or NOT ROCK as we only care about generating pixels where there are rocks, not really each individual rock

image_size = (480, 720)
num_classes = 1 
batch_size = 2


input_dir, target_dir, input_img_paths, target_img_paths = generate_image_paths()

# test iteration over the generator to see if it is working
data = artificial_lunar_landscapes(
    batch_size=2,
    image_size=image_size, 
    input_img_paths=input_img_paths,
    target_img_paths=target_img_paths
    )
#%% model

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (4,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="Adagrad", loss="BinaryCrossentropy")
    return model


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()


# Build model
model = get_model(image_size, num_classes)
model.summary()


#%% training 



callbacks = [
keras.callbacks.ModelCheckpoint("/checkpoints/checkpoint.h5")
]

train_gen = artificial_lunar_landscapes(
batch_size, image_size, input_img_paths[:10], target_img_paths[:10])
#val_gen = artificial_lunar_landscapes(batch_size, image_size, val_input_img_paths, val_target_img_paths)


# Train the model, doing validation at the end of each epoch.
epochs = 5
model.fit(train_gen, epochs=epochs, callbacks=callbacks)

#save model out
model.save('./modeling/saved-models/test_model')
#model = keras.models.load_model('test_model')