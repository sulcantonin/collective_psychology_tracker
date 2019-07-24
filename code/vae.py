import tensorflow as tf
import numpy as np
import keras.layers as kfl
from keras.models import Model
import numpy as np


# normalization of the input images, it would be better if this remain mystery
def normalize(X):
    X = X.astype(np.float32) / 255
    for i in range(X.shape[-1]):
        X[..., i] = X[..., i] - np.mean(X[..., i].ravel())
        X[..., i] = X[..., i] / np.sqrt(np.sum(X[..., i].ravel() ** 2))
    return X



def encode(self, x):
    mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
    return mean, logvar

# the default parameters means that thetre are 3 layers, each with 10 feature maps, convolution size 3x3,
# the same settings for the decoder
def autoencoder_vae(x_dim,  # width
                y_dim,  # height
                n_filters=[3, 10, 10, 10],  # number of dimensions of each layer, the input image is RGB
                # thats why the first element is 3 (one dim for each color channel)
                filter_sizes=[3, 3, 3, 3],  # size of the convolution filter, 3x3
                latent_dim = 100):
    lambda_L1 = tf.placeholder(tf.float32, 1, name='lambda')
    x = kfl.Input((x_dim,y_dim,n_filters[0]), name = 'x')


    current_input = x

    encoder = []
    decoder = []

    for l in range(1,len(n_filters)):
        current_input = kfl.Conv2D(n_filters[l],
                               filter_sizes[l],
                               strides=(2,2),
                               padding = 'same',
                               activation='relu')(current_input)
        encoder += [current_input,]

    encoder_output_shape = current_input.shape.as_list()
    current_input = kfl.Dense(latent_dim + latent_dim)(current_input)
    encoder += [current_input,]
    current_input = kfl.Flatten()(current_input)
    encoder += [current_input, ]

    z = current_input

    current_input = kfl.Dense( np.prod(encoder_output_shape[1:]))(current_input)
    decoder += [decoder, ]
    current_input = kfl.Reshape( encoder_output_shape[1:]) (current_input)
    decoder += [decoder, ]

    for l in reversed(range(1,len(n_filters))):

        current_input = kfl.Conv2DTranspose( n_filters[l],
                                         filter_sizes[l],
                                         strides=(2,2),
                                         padding='same',
                                         activation='relu')(current_input)
        decoder += [decoder, ]
    '''
    # %%
    # The output of the last layer has the same size as input
    y = current_input
    # cost function measures pixel-wise difference of the x and reconstructed output of the network +
    # regularizer which encourages usage minimum number
    cost = tf.reduce_sum(tf.square(y - x_tensor)) + lambda_L1 * tf.reduce_sum(tf.abs(z))

    # %%
    return {'x': x,
            'z': z,
            'y': y,
            'l': lambda_L1,
            'cost': cost}
    '''

    return None

def init_training(ae, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    return optimizer, session


# Traning procedure
def training_epoch(ae, optimizer, session, X, l):
    session.run(optimizer, feed_dict={ae['x']: X, ae['l']: [l]})
    return session.run((ae['y'], ae['z'], ae['cost']), feed_dict={ae['x']: X, ae['l']: [l]})

def fft_low_pass_filter(x, band=60):
    xf = np.fft.rfft(x)
    xf[band:] = 0
    return np.fft.irfft(xf)

autoencoder_vae(10,10)