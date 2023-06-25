# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 19:28:37 2021

@author: Taylor Jensen
Identify and move the train, test, and validation files to their respective directories.
*** Performed after preprocessing****

data breakout:
    - 9766 images total
    - train 70% of images = 6836
    - test 20% of images = 1953
    - validate 10% of images = 976
"""

import os
from sklearn.model_selection import train_test_split
import numpy
import pickle

class Namesplit():
    def __init__(self, mask_dir, render_dir):
        self.mask_dir = mask_dir
        self.render_dir = render_dir
        self.split_lists = []
        self.split_filenames = ['y_train', 'y_test', 'x_train', 'x_test', 'y_val', 'x_val']
        self.list_data = {}

    def split_data_filenames(self):
        """
        Returns the names of the files to be brought into train, test, and validation sets.
        """
        mask_dir_list = os.listdir(self.mask_dir)
        render_dir_list = os.listdir(self.render_dir)
    
        # training, testing
        y_train, y_test, x_train, x_test = train_test_split(
            mask_dir_list,
            render_dir_list, 
            test_size = 0.2,
            shuffle=True, 
            random_state=0
        )

        #split training again into validation
        y_train, y_val, x_train, x_val = train_test_split(
            y_train, 
            x_train,
            test_size=0.124, # 976/(9766 * 0.8) = 0.124 split for validation set
            shuffle=True,
            random_state=0
        )

        self.split_lists = [y_train, y_test, x_train, x_test, y_val, x_val]
    
    def merge_split_list_data(self):
        """
        Returns the lists of filenames for each split in a dictionary
        """
        for data, filename in zip(self.split_lists, self.split_filenames):
            self.list_data[filename] = data

    def save_lists(self, out_dir, separate_files=True, **kwargs):
        """
        Write out pickled lists to specified directory. Returns the data that has been written.
        """
        self.merge_split_list_data()
        if separate_files == True:
            for data, filename in zip(self.split_lists, self.split_filenames):
                with open(out_dir+filename+'.pkl', 'wb') as f: 
                    pickle.dump(data,f)
                print(f"Wrote {filename} to \n {out_dir}")
        
        if separate_files == False:
            with open(out_dir+'train_test_val_filename_splits.pkl', 'wb') as f:
                pickle.dump(self.list_data, f)
                print(f"Wrote train_test_val_filename_splits.pkl to \n {out_dir}")
        
        return self.list_data