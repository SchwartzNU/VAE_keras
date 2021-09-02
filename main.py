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

            self.validation_data = None
            self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
            self.reconstruction_loss_tracker = keras.metrics.Mean(
                name="reconstruction_loss"
            )
            self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

            self.validation_total_loss_tracker = keras.metrics.Mean(name="validation_total_loss")
            self.validation_reconstruction_loss_tracker = keras.metrics.Mean(
                name="validation_reconstruction_loss"
            )
            self.validation_kl_loss_tracker = keras.metrics.Mean(name="validation_kl_loss")

        def fit(self, *args, **kwargs):
            if 'validation_split' in kwargs.keys():
                fract = kwargs.pop('validation_split')
                # n_training = int(args[0].unbatch().shape[0] * fract)
                data = args[0].unbatch()
                n_total = sum(1 for _ in data)
                n_validate = int(n_total * fract)
                n_training = n_total - n_validate

                validation_data = tf.convert_to_tensor(list(data.skip(n_training)))
                # print(len(validation_data))
                print(validation_data.shape)
                data = data.take(n_training).batch(kwargs['batch_size'])
                self.validation_data = validation_data
            super(VAE,self).fit(data, **kwargs)

        @property
        def metrics(self):
            return [
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
                self.validation_total_loss_tracker,
                self.validation_reconstruction_loss_tracker,
                self.validation_kl_loss_tracker
            ]
        
        @tf.function
        def train_step(self, data):
            with tf.GradientTape() as tape:
                kl_loss, total_loss, reconstruction_loss = self.get_loss(data)
                validation_kl_loss, validation_total_loss, validation_reconstruction_loss = self.get_loss(self.validation_data)
            
                grads = tape.gradient(total_loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                self.total_loss_tracker.update_state(total_loss)
                self.reconstruction_loss_tracker.update_state(reconstruction_loss)
                self.kl_loss_tracker.update_state(kl_loss)

                self.validation_total_loss_tracker.update_state(validation_total_loss)
                self.validation_reconstruction_loss_tracker.update_state(validation_reconstruction_loss)
                self.validation_kl_loss_tracker.update_state(validation_kl_loss)
            
                #self.validation_data <- use this to add a new loss
                return {
                    "loss": self.total_loss_tracker.result(),
                    "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                    "kl_loss": self.kl_loss_tracker.result(),
                    "validation_loss": self.validation_total_loss_tracker.result(),
                    "validation_reconstruction_loss": self.validation_reconstruction_loss_tracker.result(),
                    "validation_kl_loss": self.validation_kl_loss_tracker.result(),
                    
                }
        
        def get_loss(self, data):
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
            return kl_loss,total_loss,reconstruction_loss

        

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
                    self.loss_file.write('loss\treconstruction_loss\tkl_loss\tvalidation_loss\tvalidation_reconstruction_loss\tvalidation_kl_loss\n')
                
            def on_epoch_end(self, epoch, logs=None):
                # loss_str = '{:7.2f}\t{:7.2f}\t{:7.2f}\n'.format(
                #     logs['loss'],logs['reconstruction_loss'],logs['kl_loss'])
                loss_str = '{:7.2f}\t{:7.2f}\t{:7.2f}\t{:7.2f}\t{:7.2f}\t{:7.2f}\n'.format(
                    logs['loss'],logs['reconstruction_loss'],logs['kl_loss'],logs['validation_loss'],logs['validation_reconstruction_loss'],logs['validation_kl_loss'])
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
        print(f'Training model with required memory = {keras_model_memory_usage_in_bytes(vae, batch_size=32)/1024**3:.02f} GB')
        if weights_fname is not None:
            vae.built = True;
            vae.load_weights(weights_fname) 
        if args.write_train_img == True:
            callback_list = [model_checkpoint_callback, images_callback]
        else:
            callback_list = [model_checkpoint_callback]        
        if not args.cross_validate:
            vae.fit(train_set, 
                    epochs=load_weights+epochs, 
                    batch_size=args.batch_size, 
                    callbacks=callback_list, 
                    initial_epoch = load_weights)

        else:
            #need to add validation data here
            vae.fit(train_set, 
                    validation_split = 0.1,
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

def keras_model_memory_usage_in_bytes(model, *, batch_size: int):
    """
    Return the estimated memory usage of a given Keras model in bytes.
    This includes the model weights and layers, but excludes the dataset.

    The model shapes are multipled by the batch size, but the weights are not.

    Args:
        model: A Keras model.
        batch_size: The batch size you intend to run the model with. If you
            have already specified the batch size in the model itself, then
            pass `1` as the argument here.
    Returns:
        An estimate of the Keras model's memory usage in bytes.

    """
    default_dtype = tf.keras.backend.floatx()
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            internal_model_mem_count += keras_model_memory_usage_in_bytes(
                layer, batch_size=batch_size
            )
        single_layer_mem = tf.as_dtype(layer.dtype or default_dtype).size
        out_shape = layer.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.trainable_weights]
    )
    non_trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.non_trainable_weights]
    )

    total_memory = (
        batch_size * shapes_mem_count
        + internal_model_mem_count
        + trainable_count
        + non_trainable_count
    )
    return total_memory


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

