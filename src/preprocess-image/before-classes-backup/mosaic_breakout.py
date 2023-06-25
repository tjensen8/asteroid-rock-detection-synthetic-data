# script to process the mosaic into 720*480 photos
# and perform the same transformations as are applied to the synthetic images

import tensorflow.keras as keras
import tensorflow.keras.preprocessing.image as k_image
import PIL
import numpy as np
import matplotlib.pyplot as plt

def import_mosaic(image_dir, image_filename):
    """
    Imports the designated mosaic and converts to a numpy array for processing.

    Returns
    -------
    image: numpy array

    """
    print("Import Mosaic")
    PIL.Image.MAX_IMAGE_PIXELS = 10000000000000
    ### load mosaic into memory and break down into 720*480*3
    #import image
    image = PIL.Image.open(image_dir+image_filename)
    #image as numpy array for breakout
    print("Convert Mosaic to Array")
    image = np.array(image)
    print("Import and Conversion Complete")
    return image

def image_info(image, image_name, save_location):
    """
    Provides key information about the mosiac as well as the shape of the image.

    Returns
    -------
    image_size : tuple 
        Tuple of number pixels for x, y axis.

    """
    print("Calculate Image Size")
    image_size = image.shape
    print(f"Image Size Calculated at: {image_size}")
    
    fig = plt.figure(figsize=[5,5])
    plt.hist(image.ravel())
    plt.title(f'Distribution of Pixel Intensities: {image_name}')
    fig.savefig(save_location)
    
    return image_size

def breakout_mosaic():
    # import the mosaic from a particular directory
    mosaic_dir = 'D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/usga-bennu/'
    mosaic_filename = 'Bennu_global_FB34_FB56_ShapeV28_GndControl_MinnaertPhase30_PAN_8bit_half.jpg'
    image = import_mosaic(mosaic_dir, mosaic_filename)
    
    # save image information to the processed-images-meta directory
    image_info_save_location = 'D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/processed-images-meta/'
    image_info_save_filename = 'Distribution of Pixel Intensities Mosaic.png'
    image_size = image_info(image, 'Mosaic', image_info_save_location+image_info_save_filename)
    
    # save files to the process-images/beenu-validation directory
    mosiac_out_dir = 'D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/processed-images/bennu-validation/'
    
    # for each row of size 480
    for row in range(0,image_size[0],480):
        # go along all column chunks of 780
        for col in range(0, image_size[1], 720):
            #save part of the image 
            PIL.Image.fromarray(image[row:row+480, col:col+720]).save(mosiac_out_dir+f'{row}_{col}.jpg')
            


breakout_mosaic()