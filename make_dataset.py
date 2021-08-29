#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 20:41:40 2021

@author: gws584
"""
import os
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
        seed=1,
        validation_split=0.15,
        subset='training',
    )

    test_set = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(dataset_dir, 'test'),
        label_mode="categorical", 
        labels = "inferred",
        color_mode="grayscale",
        batch_size=32,
        image_size=(32, 300),
        shuffle=True,
        seed=1,
        validation_split=0.15,
        subset='validation',
    )
    return train_set, test_set
    
def load_dataset_no_labels(dataset_dir):
    train_set = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(dataset_dir, 'train'),
        label_mode=None,
        color_mode="grayscale",
        batch_size=32,
        image_size=(32, 300),
        shuffle=True,
        seed=1,
        validation_split=0.15,
        subset='training',
    )

    test_set = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(dataset_dir, 'test'),
        label_mode=None, 
        color_mode="grayscale",
        batch_size=32,
        image_size=(32, 300),
        shuffle=True,
        seed=1,
        validation_split=0.15,
        subset='validation',
    )
    return train_set, test_set
