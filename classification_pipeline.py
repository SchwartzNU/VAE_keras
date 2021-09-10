# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 09:56:34 2021

@author: SchwartzLab
"""

#1. train VAE from each fold with a line like this:
#python main.py -mode train -dataset_dir train_fold_2 -latent_dim 3 -epochs 500 -KL_multiplier 20

#2. generate synthetic data with the VAE
#python main.py -mode generate -dataset_dir train_fold_2 -latent_dim 3 -var_scale 0.1 -N_per_type 100

#3. run umap on the training data, including the synthetic images

#4. fit a SVM to the training data, cross validated on the test data

#steps 3 and 4 run in an optimizer for hyperparameters