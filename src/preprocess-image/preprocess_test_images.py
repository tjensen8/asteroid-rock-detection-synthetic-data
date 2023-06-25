import preprocess_generator


#%% Renderings
#input directory of rendered images
artificial_image_dir = 'D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/test/render-og/'

#output directories
output_dir = 'D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/test/'
# specific subdirectories for the output
render_subdir = 'render-processed-images/'
numpy_subdir = 'render-processed-numpy/'

#process test images
preprocess_generator.process_artificial_images(
    img_dir=artificial_image_dir, 
    output_dir=output_dir, 
    render_dir=render_subdir, 
    numpy_dir=numpy_subdir,
    flip_flag=False
    )

#%% Masks
#input directories
artificial_image_dir = 'D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/test/'
artificial_image_masks_dir = artificial_image_dir+'mask-og/'

#output directories
output_dir = 'D:/TJPersonalCloud/Programming/msds-thesis-data/data-ml/split-data/test/'
render_subdir = 'mask-processed-images/'
numpy_subdir = 'mask-processed-numpy/'
    
preprocess_generator.breakout_mask_images(
    artificial_image_masks_dir=artificial_image_masks_dir, 
    output_dir=output_dir, 
    render_dir=render_subdir,
    numpy_dir=numpy_subdir,
    flip_flag=False)