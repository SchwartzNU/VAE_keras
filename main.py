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

images_callback = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", 
                        type=str,
                        default='train',
                        help="run mode: train | generate")
    parser.add_argument("-N_per_type", 
                        type=int,
                        default=10,
                        help="number of examples per type in generate mode")
    parser.add_argument("-log_var_scale", 
                        type=float,
                        default=8.0,
                        help="variancce scaling in latent space for generate mode")
    parser.add_argument("-write_train_img", 
                        type=bool,
                        default=False,
                        help="whether to write training images and loss on each epoch") #TODO: separate these
    parser.add_argument("-latent_dim", 
                        type=int, 
                        default=2,
                        help="latent dimensions")
    parser.add_argument("-epochs", 
                        type=int, 
                        default=500,
                        help="number of training epochs")
    parser.add_argument("-loadweights", 
                        type=int,
                        help="index weight file to start from")
    parser.add_argument("-batch_size", 
                        type=int,
                        default=32,
                        help="training batch size")
    parser.add_argument("-cross_validate",
                        type=bool,
                        default=True,
                        help="cross validate on each epoch, holding out 20% of the training set")
    parser.add_argument("-adam_alpha", 
                        type=float, 
                        default=0.001,
                        help="beta_1 parameter for the Adam optimizer algorithm")
    parser.add_argument("-adam_beta1", 
                        type=float, 
                        default=0.9,
                        help="beta_1 parameter for the Adam optimizer algorithm")
    parser.add_argument("-adam_beta2", 
                        type=float, 
                        default=0.999,
                        help="beta_2 parameter for the Adam optimizer algorithm")

    args = parser.parse_args()

    latent_dim = args.latent_dim
    epochs = args.epochs

    if args.loadweights is not None:
        weights_fname = os.path.join('checkpoint_latdim{}'.format(latent_dim), 
                                    f'weights_epoch_{args.loadweights:03d}.h5')
        load_weights = args.loadweights
    else:
        weights_fname = None
        load_weights = 0
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
    x = layers.Dense(16*latent_dim, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, 
                            kernel_initializer='zeros',
                            name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(8 * 75 * 128, activation="relu")(latent_inputs)
    x = layers.Reshape((8, 75, 128))(x)
    x = layers.Conv2DTranspose(64, (2,3), activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(128, (2,3), activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation='relu', padding="same")(x)
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
    if args.mode == 'train':
        dataset_dir = 'RGCtypes_1887'
        [train_set, test_set] = make_dataset.load_dataset_no_labels(dataset_dir)

    # %%make test sample

    if args.mode == 'train' and args.write_train_img == True:
        num_examples_to_generate = 5
        # Pick a sample of the test set for generating output images
        for test_batch in test_set.take(1):
            # Pick a sample of the test set for generating output images
            test_sample = test_batch[0:num_examples_to_generate, :, :, :]


    # %%make model
    vae = VAE(encoder, decoder)
    optimizer = keras.optimizers.Adam(learning_rate = args.adam_alpha,
                                    beta_1 = args.adam_beta1,
                                    beta_2 = args.adam_beta2)

    vae.compile(optimizer=optimizer)

    # %%checkpoint callback

    if args.mode == 'train':
        os.makedirs('checkpoint_latdim{}'.format(latent_dim), exist_ok=True)        
        file_name = "weights_epoch_{epoch:03d}.h5"
        checkpoint_filepath = os.path.join('checkpoint_latdim{}'.format(latent_dim), file_name)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            save_weights_only=True,
            monitor='loss',
            save_best_only=True,
            filepath=checkpoint_filepath)

    # %%image save callback
    if args.mode == 'train' and args.write_train_img == True:
        class SaveSampleImagesCallback(keras.callbacks.Callback):
            def __init__(self, test_sample):
                self.data = test_sample
                if weights_fname is not None:
                    self.loss_file = open('loss_file_latdim{}.txt'.format(latent_dim), "a")
                    self.loss_file.write(f'loaded weights {args.loadweights:03d}\n')
                else:
                    self.loss_file = open('loss_file_latdim{}.txt'.format(latent_dim), "w")
                    self.loss_file.write('loss\treconstruction_loss\tkl_loss\n')
                
            def on_epoch_end(self, epoch, logs=None):
                loss_str = '{:7.2f}\t{:7.2f}\t{:7.2f}\n'.format(
                    logs['loss'],logs['reconstruction_loss'],logs['kl_loss'])
                print(loss_str)
                self.loss_file.write(loss_str)
                                    
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
                
                # TODO: should fix epoch...
                plt.savefig('training_img_latdim{}/image_at_epoch_{:04d}.png'.format(latent_dim,epoch))
                # plt.show()
                plt.close()
            
            def on_train_end(self, logs=None):
                self.loss_file.close()
        
        os.makedirs('training_img_latdim{}'.format(latent_dim), exist_ok=True)        
        images_callback = SaveSampleImagesCallback(test_sample)


    #%% train
    if args.mode == 'train':
        if weights_fname is not None:
            vae.built = True;
            vae.load_weights(weights_fname) 
        if args.write_train_img == True:
            callback_list = [model_checkpoint_callback, images_callback]
        else:
            callback_list = [model_checkpoint_callback]
        
        if args.cross_validate:
            vae.fit(train_set, 
                    epochs=load_weights+epochs, 
                    batch_size=args.batch_size, 
                    callbacks=callback_list, 
                    initial_epoch = load_weights)

        else:
            vae.fit(train_set, 
                    validation_split = 0.2,
                    epochs=load_weights+epochs,                     
                    batch_size=args.batch_size, 
                    callbacks=callback_list, 
                    initial_epoch = load_weights)

    #%% generate data
    if args.mode == 'generate':
        vae.built = True
        if weights_fname is not None:
            vae.load_weights(weights_fname) 
        import GenerateFromTrainedModel as gen 
        validated_dir = 'RGCtypes_validated_473'
        (train_set, test_set) = make_dataset.load_dataset_with_labels(validated_dir)
        gen.generate_data(vae,train_set,N_per_type=args.N_per_type,log_var_scale=args.log_var_scale)

def cleanup():
    print('interrupting gracefully')
    if images_callback is not None:
        images_callback.loss_file.close()
    print('interrupted gracefully')
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        cleanup()

