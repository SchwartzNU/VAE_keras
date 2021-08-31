#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 14:48:39 2021

@author: gws584
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

def generate_data(model, dataset, N_per_type=5):
    types = get_label_list(dataset)
    unique_types = list(set(types))
    dataset_unbatched = dataset.unbatch()
    plt.ioff()
    for i in range(len(unique_types)):
        print(unique_types[i])
        #ind = [index for index, element in enumerate(types) if element == unique_types[i]]
        os.makedirs(os.path.join('generated',unique_types[i]),exist_ok=True)
        gen_counter = 0
        while gen_counter < N_per_type:
            for d, label in dataset_unbatched.as_numpy_iterator():
                temp = np.argwhere(np.array(label > 0))
                cur_type = dataset.class_names[temp[0,0]]
                if cur_type == unique_types[i]:
                    _, _, z_sample = model.encoder.predict(d[None,:,:,:])
                    new_data = model.decoder.predict(z_sample)
                    # breakpoint()
                    plt.figure(figsize=(8, 2),
                               tight_layout=True,
                               dpi=300,
                               frameon=False)
                    plt.imshow(new_data[0, :, :, 0], cmap='gray')
                    plt.axis('off')
                    fname = os.path.join('generated',unique_types[i],'{:4d}.jpg'.format(gen_counter))
                    plt.savefig(fname)
                    plt.close()
                    gen_counter = gen_counter+1
                    
#z_mean, _, _ = vae.encoder.predict(train_set)