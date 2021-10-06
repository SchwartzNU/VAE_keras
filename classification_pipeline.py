# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 09:56:34 2021

@author: SchwartzLab
"""

#1. train VAE from each fold with a line like this:
#python main.py -mode train -dataset_dir train_fold_2 -latent_dim 3 -epochs 500 -KL_multiplier 20

#2. generate synthetic data with the VAE
#python main.py -mode generate -dataset_dir train_fold_2 -loadweights -1 -latent_dim 3 -var_scale 0.1 -N_per_type 100

#3. run umap on the training data, including the synthetic images

#%% init
from Umap_images import RGCtypes_umap
import os

Nfolds = 4
gen_name = 'generated_latdim3_varScale_0.01'

#%% load the images into the Umap_images class 
U_list = list()
for i in range(Nfolds):
    print('Loading images for fold {}'.format(i))
    base_dir = 'train_fold_{}'.format(i)
    U = RGCtypes_umap(validated_images_dir=os.path.join(base_dir, 'train', 'validated'), 
                      unvalidated_images_dir=os.path.join(base_dir, 'test', 'validated')) #TODO: change the naming here
    U.load_generated_data_and_labels(os.path.join(base_dir,gen_name))
    U.make_umap_dataset('v+g', 'u')
    U_list.append(U)
        
#%% umap fitting     
for i in range(Nfolds):
    print('Fitting umap for fold {}'.format(i))    
    U = U_list[i]
    U.fit_umap(supervised='full',n_neighbors=900,min_dist=.5)
    U.plot_umap(dataset='test')
    
#%% SVM fitting    
#4. fit a SVM to the training data
for i in range(Nfolds):
    print('Fitting svm for fold {}'.format(i))    
    U = U_list[i]
    U.svm_optimizer()
    
#5. check SVM performance on test data


#steps 3 and 4 run in an optimizer for hyperparameters