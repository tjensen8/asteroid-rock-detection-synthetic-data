"""
Baseline unet model with weighted binary cross entropy. ADJUSTED LAYOUT. 
Removed initial training set in order to continue training after out of memory error on epoch 4. 
"""
#import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
from tensorflow.random import set_seed
from tensorflow import keras
from tensorflow.keras.models import Model, load_model, model_from_json
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, concatenate, Input, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.distribute import MirroredStrategy, HierarchicalCopyAllReduce
from tensorflow.keras import backend as K
import tensorflow as tf
import gc

tf.config.run_functions_eagerly(False)
import numpy as np
from model_utils import artificial_lunar_landscapes, generate_image_paths
#from model_utils_y_model_copy import artificial_lunar_landscapes, generate_image_paths
import os

import time

set_seed(0)

os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"]="2"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

#set distributed training
strategy = MirroredStrategy(cross_device_ops=HierarchicalCopyAllReduce())

with strategy.scope():
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

    def wbce( y_true, y_pred, weight1=1, weight0=0.1 ) :
        y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
        y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
        logloss = -(y_true * K.log(y_pred) * weight1 + (1 - y_true) * K.log(1 - y_pred) * weight0 )
        return K.mean( logloss, axis=-1)

    image_size = (480, 720, 1)
    #num_classes = 2 
    batch_size = 2



    # for tensorflow logging
    root_logdir = os.path.join(os.curdir,"modeling","results","simple-modeling","unet-weighted-bce-trainval", "tflogs")

    def run_logdir(root_logdir):
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(root_logdir, run_id)

    # for model checkpoints
    root_checkpointdir = os.path.join(os.curdir,"modeling","results","simple-modeling","unet-weighted-bce-trainval", "checkpoints")

    def run_checkpointdir(root_checkpointdir):
        run_id = 'model.{epoch:02d}-{loss:.2f}-{accuracy:.2f}.h5'
        return os.path.join(root_checkpointdir, run_id)



    class MyCustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            keras.backend.clear_session()
            gc.collect()
            

    callbacks = [
    keras.callbacks.ModelCheckpoint(run_checkpointdir(root_checkpointdir), mode='max', monitor='accuracy', save_freq='epoch', save_best_only=False),
    keras.callbacks.TensorBoard(run_logdir(root_logdir), histogram_freq=1, write_images=True, update_freq='batch'),
    MyCustomCallback()
    ]

    def do_modeling(epoch):
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
        """
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
        """
        model = load_model(f'{root_checkpointdir}/model-training-{epoch-1}.h5', custom_objects={'weighted_bincrossentropy': weighted_bincrossentropy})
        history = model.fit(
            train_generator,
            callbacks=callbacks,
            verbose=1,
            epochs=1,
            validation_data=(test_generator))
        #model.evaluate_generator(test_generator, callbacks=callbacks)
        print("Saving Training Model")
        model.save(f'{root_checkpointdir}/model-training-{epoch}.h5')
        print("Saving Current Epoch Model")
        model.save(f'{root_checkpointdir}/model-{epoch}.h5')
        
        del history
        del model
        del train_generator
        del test_generator
        keras.backend.clear_session()
        for i in range(3): gc.collect()
        
        return False
    
    
    epochs=20
    start_epoch = 16
    for epoch in range(start_epoch, epochs):
        print(epoch)
        do_modeling(epoch)



#model.fit(
#    train_generator, 
#    epochs=20, # 25 was originally selected - would take 48 hours
#    callbacks=callbacks, 
#    verbose=1,
#    validation_data=(test_generator)) #validation results outputted after an epoch
