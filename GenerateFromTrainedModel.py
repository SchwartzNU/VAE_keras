#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 14:48:39 2021

@author: gws584
"""
import os
import numpy as np
import tensorflow as tf


def load_dataset_with_labels(dataset_dir):
    train_set = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(dataset_dir, 'train'),
        label_mode="categorical",
        labels = "inferred",
        color_mode="grayscale",
        batch_size=32,
        image_size=(32, 300),
        shuffle=True,
        seed=2,
    )

    test_set = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(dataset_dir, 'test'),
        label_mode="categorical", 
        labels = "inferred",
        color_mode="grayscale",
        batch_size=32,
        image_size=(32, 300),
        shuffle=True,
        seed=2,
    )
    return train_set, test_set

def get_label_list(dataset):
    L = list();
    dataset_unbatched = dataset.unbatch()
    for _, label in dataset_unbatched.as_numpy_iterator():
        temp = np.argwhere(np.array(label > 0))
        L.append(dataset.class_names[temp[0,0]])
    return L


def generate_data(dataset, N_per_type=10):
    types = get_label_list(dataset)
    unique_types = list(set(types))
    for i in range(len(unique_types)):
        ind = [index for index, element in enumerate(types) if element == unique_types[i]]
        os.makedir(unique_types[i],exist_ok=True)
        for j in range(N_per_type):
            
        print(ind)
        
#z_mean, _, _ = vae.encoder.predict(train_set)