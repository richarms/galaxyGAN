from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import os

import PIL
from tensorflow.keras import layers

# load the pre-augmented image array
im_arr = np.load("data/vac_im.dat.npy").reshape(23194, 150, 150,1).astype('float32')
im_arr = im_arr[:,11:139, 11:139,:]
im_arr = (im_arr - 127.5) / 127.5 # Normalize the images to [-1, 1]

BUFFER_SIZE = 24000 #60000
BATCH_SIZE = 256 #16


#use a tensorflow dataset object for ease of shuffling, splitting into train/test/verify, etc
train_dataset = tf.data.Dataset.from_tensor_slices(im_arr).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# a function to compose a model from various sequential layers. Which is an unusual thing to do in OO-land.
# a kind of complex-class constructor, maybe?
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(16 * 16 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((16, 16, 256)))
    # assert model.output_shape == (None, 16, 16, 256) # Note: None is the batch size
    print(model.output_shape)

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    # assert model.output_shape == (None, 32, 32, 128)
    print(model.output_shape)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    # assert model.output_shape == (None, 64, 64, 64)
    print(model.output_shape)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    # assert model.output_shape == (None, 128, 128, 1)
    print(model.output_shape)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[128, 128, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


# actually contruct the models of the generator and the discriminator of a GAN
generator = make_generator_model()
discriminator = make_discriminator_model()


# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

