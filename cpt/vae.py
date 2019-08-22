from keras.layers import Conv2D, Conv2DTranspose, Flatten, Dense, Input, Lambda, Reshape, BatchNormalization
from keras.models import Model
from keras.losses import mse
import keras.backend as K
import numpy as np
import tensorflow as tf


# https://keras.io/examples/mnist_denoising_autoencoder/

class vae:

    # normalization of the input images, it would be better if this remain mystery
    def normalize(X):
        X = X.astype(np.float32) / 255
        for i in range(X.shape[-1]):
            X[..., i] = X[..., i] - np.mean(X[..., i].ravel())
            X[..., i] = X[..., i] / np.sqrt(np.sum(X[..., i].ravel() ** 2))
        return X

    # the default parameters means that thetre are 3 layers, each with 10 feature maps, convolution size 3x3,
    # the same settings for the decoder
    def __init__(self, x_dim,  # width
                 y_dim,  # height
                 n_filters=[3, 10, 10, 10],  # number of dimensions of each layer, the input image is RGB
                 # thats why the first element is 3 (one dim for each color channel)
                 filter_sizes=[3, 3, 3, 3],  # size of the convolution filter, 3x3
                 latent_dim=100):
        self.latent_dim = latent_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_filters = n_filters
        self.filter_sizes = filter_sizes

        self.x = Input((x_dim, y_dim, n_filters[0]), name='x')

        current_input = self.x
        # encoder
        for nf, fs in zip(n_filters[1:], filter_sizes[1:]):
            current_input = Conv2D(nf, fs,
                                   strides=2,
                                   padding='same',
                                   activation='relu')(current_input)
            # current_input = BatchNormalization(momentum=0.1)(current_input)
        # building the latent code from the encoder outputs as a dense layer
        encoder_conv_shape = current_input.shape.as_list()

        current_input = Flatten()(current_input)
        z_mean = Dense(latent_dim, name='z_mean')(current_input)
        z_log_var = Dense(latent_dim, name='z_log_var')(current_input)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        self.z = Lambda(self.sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        self.encoder = Model(self.x, [z_mean, z_log_var, self.z], name='encoder')
        # K.utils.plot_model(self.encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

        # decoder
        z_sampling = Input((self.latent_dim,), name='z_sampling')
        current_input = Dense(np.prod(encoder_conv_shape[1:]))(z_sampling)
        current_input = Reshape(encoder_conv_shape[1:])(current_input)

        for nf, fs in zip(reversed(n_filters[:-1]), reversed(filter_sizes[:-1])):
            current_input = Conv2DTranspose(nf, fs,
                                            strides=2,
                                            padding='same',
                                            activation='relu')(current_input)

        # instantiate decoder model
        self.decoder = Model(z_sampling, current_input, name='decoder')

        # instantiate VAE model
        self.outputs = self.decoder(self.encoder(self.x)[2])
        self.encoder.summary()
        self.decoder.summary()
        self.net = Model(self.x,
                         self.outputs,
                         name='vae_mlp')

        def loss(x, output):

            reconstruction_loss = mse(x, output)#  * (x_dim * y_dim)
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            return K.mean(reconstruction_loss + kl_loss)

        self.net.compile(optimizer='adam', loss=loss)

    # reparameterization trick
    # instead of sampling from Q(z|X), sample eps = N(0,I)
    # z = z_mean + sqrt(var)*eps

    def train(self, X, epochs):
        self.net.fit(X, X, epochs=epochs)
        self.net.save_weights('vae_mlp_mnist.h5')

    def sampling(self, args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.

        # Arguments
            args (tensor): mean and log of variance of Q(z|X)

        # Returns
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon


nx = 64
X = np.random.rand(nx, nx, 3, 1000)
X = np.transpose(X, (3, 0, 1, 2))

vae = vae(nx, nx)
vae.train(X, 10)
