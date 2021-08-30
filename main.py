#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 22:46:07 2021

@author: gws584
"""
#%% init
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import make_dataset
import os
import matplotlib.pyplot as plt 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-latent_dim", 
                    type=int, 
                    default=2,
                    help="latent dimensions")
parser.add_argument("-epochs", 
                    type=int, 
                    default=100,
                    help="number of training epochs")
args = parser.parse_args()

latent_dim = args.latent_dim
epochs = args.epochs

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


encoder_inputs = keras.Input(shape=(32, 300, 1))
x = layers.Conv2D(64, (2,3), activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(128, (2,3), activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, 
                         name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(8 * 75 * 128, activation="relu")(latent_inputs)
x = layers.Reshape((8, 75, 128))(x)
x = layers.Conv2DTranspose(64, (2,3), activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(128, (2,3), activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation=None, padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            # reconstruction = tf.boolean_mask(reconstruction, tf.math.is_finite(reconstruction)) #get rid of nans
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    

#%% load dataset
dataset_dir = 'RGCtypes_1887'

[train_set, test_set] = make_dataset.load_dataset_no_labels(dataset_dir)

#mnist_digits = np.concatenate([x_train, x_test], axis=0)
#mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

# %%make test sample

num_examples_to_generate = 5

# Pick a sample of the test set for generating output images
for test_batch in test_set.take(1):
    # Pick a sample of the test set for generating output images
    test_sample = test_batch[0:num_examples_to_generate, :, :, :]


# %%make model
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())

# %%checkpoint callback

os.makedirs('checkpoint_latdim{}'.format(latent_dim), exist_ok=True)        
file_name = "weights_epoch_{epoch:03d}.h5"
checkpoint_filepath = os.path.join('checkpoint_latdim{}'.format(latent_dim), file_name)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    save_weights_only=True,
    monitor='loss',
    save_best_only=True,
    filepath=checkpoint_filepath)

# %%image save callback

class SaveSampleImagesCallback(keras.callbacks.Callback):
    def __init__(self, test_sample):
        self.data = test_sample
        
    def on_epoch_end(self, epoch, logs=None):
        [z_mean, z_logvar, z_sample]  = self.model.encoder.predict(self.data)
        predictions = self.model.decoder.predict(z_sample)
        N = self.data.shape[0]    
        fig = plt.figure(figsize=(8, 2),
                         tight_layout=True,
                         dpi=300)
        
        for i in range(N):
            fig.add_subplot(2, N, i + 1, title='data')
            plt.imshow(self.data[i, :, :, 0], cmap='gray')
            plt.axis('off')
            fig.add_subplot(2, N, N + i + 1, title='reconstruction')
            plt.imshow(predictions[i, :, :, 0], cmap='gray')
            plt.axis('off')
        
        plt.savefig('training_img_latdim{}/image_at_epoch_{:04d}.png'.format(latent_dim,epoch))
        # plt.show()

os.makedirs('training_img_latdim{}'.format(latent_dim), exist_ok=True)        
images_callback = SaveSampleImagesCallback(test_sample)

#%% train

vae.fit(train_set, epochs=epochs, batch_size=32, callbacks=[model_checkpoint_callback, images_callback])



