#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 20:41:40 2021

@author: gws584
"""
import numpy as np
import os
import tensorflow as tf

#dataset_dir = 'RGCtypes_1887'
#labels = np.loadtxt('labels_int.txt').astype(int).tolist()


def load_dataset(dataset_dir):

    train_set = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
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
        dataset_dir,
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
