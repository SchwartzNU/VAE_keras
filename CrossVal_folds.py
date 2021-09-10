#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 12:53:34 2021

@author: gws584
"""
import os
import glob
from shutil import copy2

def make_folds(N=4,image_dir = os.path.join('RGCtypes_validated_473','train_subset')):
    data_folds_list = list()
    for folder_name in glob.glob(os.path.join(image_dir,'*')):
        [_, cell_type] = os.path.split(folder_name)
        image_list = glob.glob(os.path.join(folder_name,'*.png'))
        n_images = len(image_list)
        print(cell_type)
        print(n_images)

        n_of_each = int(n_images/N)
        # using list comprehension 
        F = [image_list[i:i + n_of_each] for i in range(0, n_images, n_of_each)]
        #print(F)
        
        for i in range(len(F)):
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

def make_image_folder_for_fold(data_folds_list, f, val_status='validated'):
    base_name = 'train_fold_{}'.format(f)
    folder_list = list(set([data_folds_list[i]['label'] for i in range(len(data_folds_list))]))
    for i in range(len(folder_list)):
        os.makedirs(os.path.join(base_name, 'train', val_status, folder_list[i]), exist_ok = True)
        os.makedirs(os.path.join(base_name, 'test', val_status, folder_list[i]), exist_ok = True)
    
    train_set = holdout_fold(data_folds_list, f)
    test_set = get_fold(data_folds_list, f)
    
    for i in range(len(train_set)):
        copy2(train_set[i]['image_path'], os.path.join(base_name, 'train', val_status, train_set[i]['label']))
        
    for i in range(len(test_set)):
        copy2(test_set[i]['image_path'], os.path.join(base_name, 'test', val_status, test_set[i]['label']))
    
    