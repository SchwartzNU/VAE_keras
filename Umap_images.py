#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 12:57:45 2021

@author: gws584
"""

import umap
from PIL import Image
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

def load_data_and_labels(images_dir='umap_test', im_size=32*300):
    labels_val = list()
    labels_all = list()
    
    image_data = None
    
    type_dict = {}
    label_ind = 0
    #validated dataset    
    for filename in glob.glob(os.path.join(images_dir,'validated','*','*.jpg')):
        #print(filename)
        if not os.path.isdir(filename): #don't run on direcctories
            [path, _] = os.path.split(filename)
            [_, label] = os.path.split(path)
            if not label in type_dict:
                type_dict[label] = label_ind
                label_ind = label_ind + 1
            label_int = type_dict[label]                
            labels_val.append(label_int)
            labels_all.append(label_int)
            img = Image.open(filename)
            if image_data is None:
                image_data = np.array(img.getdata()).reshape([1,im_size])
            else:
                image_data = np.append(image_data,np.array(img.getdata()).reshape([1,im_size]),axis=0)
            #print(image_data.shape)
            
    #unvalidated dataset
    for filename in glob.glob(os.path.join(images_dir,'unvalidated','*','*.jpg')):
        #print(filename)
        if not os.path.isdir(filename): #don't run on direcctories
            [path, _] = os.path.split(filename)
            [_, label] = os.path.split(path)
            labels_val.append(-1)
            if label in type_dict:
                labels_all.append(type_dict[label])
            else:
                labels_all.append(-1)
            img = Image.open(filename)
            if image_data is None:
                image_data = np.array(img.getdata()).reshape([1,im_size])
            else:
                image_data = np.append(image_data,np.array(img.getdata()).reshape([1,im_size]),axis=0)
            #print(image_data.shape)
    
    return image_data, labels_val, labels_all, type_dict

def plot_umap(data,labels_sup,labels_all,label_dict,supervised='semi',plotpoints='all'):
    labels_val_only = [np.nan if x == -1 else x for x in labels_sup]
    labels_unval_only = [np.nan if labels_sup[i] > -1 else labels_all[i] for i in range(len(labels_all))]
    if supervised == 'semi':
        embedding = umap.UMAP().fit_transform(data, y=labels_sup)
    elif supervised == 'full':
        embedding = umap.UMAP().fit_transform(data, y=labels_all)
    else:
        embedding = umap.UMAP().fit_transform(data)
    fig, ax = plt.subplots(1, figsize=(5, 4))
    if plotpoints == 'validated':
        color_vals = labels_val_only
    elif plotpoints == 'unvalidated':
        color_vals = labels_unval_only
    else:
        color_vals = labels_all
    plt.scatter(*embedding.T, s=2, c=color_vals, cmap='Spectral', alpha=1.0)
    plt.setp(ax, xticks=[], yticks=[])
    n_types = len(label_dict)
    cbar = plt.colorbar(boundaries=np.arange(n_types+1)-0.5)
    cbar.set_ticks(np.arange(n_types))
    cbar.set_ticklabels(np.array(list(label_dict.keys())))
    plt.title('UMAP {}_superivsed'.format(supervised));

