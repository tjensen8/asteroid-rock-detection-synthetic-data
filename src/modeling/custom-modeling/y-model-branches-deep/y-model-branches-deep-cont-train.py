"""

Custom model approach with branches to handle original image and images that are preprocessed

"""
from tensorflow.random import set_seed
from tensorflow import keras
from tensorflow.keras.models import Model, load_model, model_from_json
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

    class MyCustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()

    callbacks = [
    keras.callbacks.ModelCheckpoint(run_checkpointdir(root_checkpointdir), mode='max', monitor='accuracy', save_freq='epoch', save_best_only=False),
    keras.callbacks.TensorBoard(run_logdir(root_logdir), histogram_freq=1, write_images=False, update_freq='batch'),
    MyCustomCallback()
    ]
    #model.fit(
    #    train_generator, 
    #    epochs=20, 
    #    callbacks=callbacks, 
    #    verbose=1)
        #validation_data=(test_generator)) #validation results outputted after an epoch

    # for model checkpoints
    batch_size = 2
    image_size = (480,720,1)
    input_processed_size = (480,720,1)
    
    def do_modeling(epoch):
        #training generator
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

        model = load_model(f'{root_checkpointdir}/model-training-{epoch-1}.h5', custom_objects={'weighted_bincrossentropy': weighted_bincrossentropy})
        history = model.fit(
            train_generator,
            callbacks=callbacks,
            verbose=1,
            epochs=1)
            #validation_data=(test_generator))
        #model.evaluate_generator(test_generator, callbacks=callbacks)
        print("Saving Training Model")
        model.save(f'{root_checkpointdir}/model-training-{epoch}.h5')
        print("Saving Current Epoch Model")
        #model.save(f'{root_checkpointdir}/model-{epoch}.h5')
        
        del history
        del model
        del train_generator
        keras.backend.clear_session()
        for i in range(3): gc.collect()
        
        return False

    epochs=20
    start_epoch = 12
    for epoch in range(start_epoch, epochs):
        print(epoch)
        do_modeling(epoch)