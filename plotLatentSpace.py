#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 20:07:59 2021

@author: gws584
"""''
import numpy as np
import matplotlib.pyplot as plt 

#%% plot latent space

def plot_latent_space(vae, n=20, figsize=(11,3)):
    # display a n*n 2D manifold of digits
    nR = 32
    nC = 300
    scale = 50.0
    figure = np.zeros((nR * n, nC * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            reconstruction = x_decoded[0].reshape(nR, nC)
            figure[
                i * nR : (i + 1) * nR,
                j * nC : (j + 1) * nC,
            ] = reconstruction

    plt.figure(figsize=figsize,dpi=600)
    # start_range = digit_size // 2
    # end_range = n * digit_size + start_range
    # pixel_range = np.arange(start_range, end_range, digit_size)
    # sample_range_x = np.round(grid_x, 1)
    # sample_range_y = np.round(grid_y, 1)
    # plt.xticks(pixel_range, sample_range_x)
    # plt.yticks(pixel_range, sample_range_y)
    # plt.xlabel("z[0]")
    # plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.savefig('latent_space_manifold.png')
    plt.show()

plot_latent_space(vae)    