#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 12:53:34 2021

@author: gws584
"""
import os
import glob
import numpy as np

def make_folds(N=4):
    data_folds_list = list()
    image_dir = os.path.join('RGCtypes_validated_473','train_subset')
    for folder_name in glob.glob(os.path.join(image_dir,'*')):
        [_, cell_type] = os.path.split(folder_name)
        image_list = glob.glob(os.path.join(folder_name,'*.png'))
        n_images = len(image_list)
        print(cell_type)
        print(n_images)

        n_of_each = int(n_images/N)
        # using list comprehension 
        F = [image_list[i:i + N] for i in range(0, n_images, n_of_each)]
        #print(F)
        
        for i in range(N):
            curList = F[i]
            print(curList)
            print(len(curList))
            for j in range(len(curList)):
                entry = dict()
                entry['fold'] = i
                entry['label']= cell_type
                entry['image_path'] = curList[j]
                data_folds_list.append(entry)
                
    return data_folds_list

def get_fold(data_folds_list, f):
    return [data_folds_list[i] for i in range(len(data_folds_list)) if data_folds_list[i]['fold'] == f]

def holdout_fold(data_folds_list, f):
    return [data_folds_list[i] for i in range(len(data_folds_list)) if not data_folds_list[i]['fold'] == f]

