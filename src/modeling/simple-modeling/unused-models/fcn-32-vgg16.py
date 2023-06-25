"""
Fully connected network - FCN 8 from:

https://github.com/divamgupta/image-segmentation-keras
"""

from tensorflow.random import set_seed
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, concatenate, Input, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.distribute import MirroredStrategy, HierarchicalCopyAllReduce

import numpy as np
from model_utils import artificial_lunar_landscapes, generate_image_paths
import os

import time

set_seed(0)

os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"]="2"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"