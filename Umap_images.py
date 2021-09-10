#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 12:57:45 2021

@author: gws584
"""

import umap
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

class RGCtypes_umap:
    im_size = 32*300
    validated_data = []
    validated_labels = []
    unvalidated_data = []
    unvalidated_labels = []
    generated_data = []
    generated_labels = []
    type_dict = {}
    train_data = []
    test_data = []
    train_labels = []
    test_labels = []
    train_labels_semi = []
    embedding = []
    reducer = []
    
    def __init__(self, validated_images_dir=os.path.join('RGCtypes_validated_473','train_subset'), unvalidated_images_dir='unvalidated_subset'):
        self.validated_data, self.validated_labels = self.load_data_and_labels(validated_images_dir)
        self.unvalidated_data, self.unvalidated_labels = self.load_data_and_labels(unvalidated_images_dir)
                
    def load_data_and_labels(self,images_dir):
        labels_val = list()
        
        image_data = None
        
        label_ind = 0
        for filename in glob.glob(os.path.join(images_dir,'*','*.png')):
            #print(filename)
            if not os.path.isdir(filename): #don't run on direcctories
                [path, _] = os.path.split(filename)
                [_, label] = os.path.split(path)
                if not label in self.type_dict:
                    self.type_dict[label] = label_ind
                    label_ind = label_ind + 1
                label_int = self.type_dict[label]                
                labels_val.append(label_int)
                img = plt.imread(filename)
                if len(img.shape) == 3:
                    img = img[:,:,0]
                if image_data is None:
                    image_data = img.reshape([1,self.im_size])
                else:
                    image_data = np.append(image_data,img.reshape([1,self.im_size]),axis=0)
                    
        return image_data, labels_val
    
    def load_generated_data_and_labels(self,images_dir):
        self.generated_data, self.generated_labels = self.load_data_and_labels(images_dir)
    
    def make_umap_dataset(self, train_set, test_set):
        if train_set == 'v':
            train_data = self.validated_data
            train_labels = self.validated_labels
        elif train_set == 'v+u':
            train_data = np.concatenate((self.validated_data, self.unvalidated_data), axis=0)
            train_labels = self.validated_labels + self.unvalidated_labels
            self.train_labels_semi = self.validated_labels + [-1] * len(self.unvalidated_labels)
        elif train_set == 'g':
            train_data = self.generated_data
            train_labels = self.generated_labels
        elif train_set == 'v+g':
            train_data = np.concatenate((self.validated_data, self.generated_data), axis=0)
            train_labels = self.validated_labels + self.generated_labels
            self.train_labels_semi = self.validated_labels + [-1] * len(self.generated_labels)
        
        if test_set == 'v':
            test_data = self.validated_data
            test_labels = self.validated_labels
        elif test_set == 'u':
            test_data = self.unvalidated_data
            test_labels = self.unvalidated_labels
        elif test_set == 'g':
            test_data = self.generated_data
            test_labels = self.generated_labels
    
        self.train_data = train_data
        self.test_data = test_data
        self.train_labels = train_labels 
        self.test_labels = test_labels
    
    def fit_umap(self,supervised='semi',n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean'):
        self.reducer = umap.UMAP(n_neighbors=n_neighbors,
                            min_dist=min_dist,
                            n_components=n_components,
                            metric=metric)                       
        if supervised == 'semi':
            self.reducer.fit(self.train_data, y=self.train_labels_semi)
        elif supervised == 'full':
            self.reducer.fit(self.train_data, y=self.train_labels)
            
        self.embedding = self.reducer.transform(self.train_data)    
                
    def plot_umap(self):     
        test_embedding = self.reducer.transform(self.test_data)          
        fig, ax = plt.subplots(1, figsize=(14, 10))
        plt.scatter(*test_embedding.T, s=15, c=self.test_labels, cmap='Spectral', alpha=.5)
        plt.setp(ax, xticks=[], yticks=[])
        n_types = len(self.type_dict)
        cbar = plt.colorbar(boundaries=np.arange(n_types+1)-0.5)
        cbar.set_ticks(np.arange(n_types))
        cbar.set_ticklabels(np.array(list(self.type_dict.keys())))
        plt.title('UMAP');
        plt.show()
    
    def fit_svm(self,C=1.0,kernel='linear',degree=2,gamma='scale',coef0=0.0):
        svm = SVC(C=C,
                 kernel=kernel,
                 degree=degree,
                 gamma=gamma,
                 coef0=coef0)
        svm.fit(self.embedding, self.train_labels)
        return svm.score(self.embedding, self.train_labels)
        
    def svm_optimizer(self):
        #define the space of hyperparameters to search
        search_space = list()
        search_space.append(Real(1e-4, 10.0, 'log-uniform', name='C'))
        search_space.append(Categorical(['linear', 'poly'], name='kernel'))
        search_space.append(Integer(1, 4, name='degree'))
        search_space.append(Real(1e-4, 10.0, 'log-uniform', name='gamma'))

        #define the function used to evaluate a given configuration
        @use_named_args(search_space)
        def evaluate_model(**params):
            # configure the model with specific hyperparameters
            model = SVC()
            model.set_params(**params) 
            
            # define test harness
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        	# calculate 5-fold cross validation
            result = cross_val_score(model, self.embedding, self.train_labels, 
                                     cv=cv, 
                                     n_jobs=-1, 
                                     scoring='accuracy',
                                     verbose=True)
        	# calculate the mean of the scores
            estimate = np.mean(result)
        
            # convert from a maximizing score to a minimizing score
            return 1.0 - estimate
        
        # perform optimization
        result = gp_minimize(evaluate_model, search_space, n_calls=20, verbose=True)
        # summarizing finding:
        print('Best Accuracy: %.3f' % (1.0 - result.fun))
        print('Best Parameters: %s' % (result.x))













