import tensorflow as tf
import numpy as np

# the default parameters means that theree are 3 layers, each with 10 feature maps, convolution size 3x3,
# the same settings for the decoder
def autoencoder(x_dim,  # width
                y_dim,  # height
                n_filters=[3, 10, 10, 10],  # number of dimensions of each layer, the input image is RGB
                # thats why the first element is 3 (one dim for each color channel)
                filter_sizes=[3, 3, 3, 3]):  # size of the convolution filter, 3x3
    # you can ignore this, it is just to get better results,
    init = tf.contrib.layers.xavier_initializer()
    # defining the input x for the tensorflow
    x = tf.placeholder(tf.float32, [None, x_dim * y_dim * n_filters[0]], name='x')
    xTensor = tf.reshape(x, [-1, x_dim, y_dim, n_filters[0]])
    # defining the reguralizer for the latent space
    # this value of value of lambda_L1 the less peaks you get
    lambdaL1 = tf.placeholder(tf.float32, 1, name='lambda')

    current_input = xTensor

    # Build the encoder
    encoder = []
    shapes = []
    # stacking the encoders
    for l, no in enumerate(n_filters[1:]):
        # number of inputs from previous layer
        ni = current_input.get_shape().as_list()[-1]
        # defining size of the kernel
        W_shape = [filter_sizes[l], filter_sizes[l], ni, no]
        # storing the size if the input (for decoder they should match)
        shapes.append(current_input.get_shape().as_list())
        # creating variable for convolution kernel and bias
        # it need to be defined as variable because they can be optimized during the traning
        W = tf.Variable(init(W_shape))
        b = tf.Variable(init([W_shape[-1]]))
        # storing the encoder
        encoder.append(W)
        # leaky_relu ( convolution ( input ) )
        output = tf.nn.leaky_relu(tf.nn.conv2d(current_input, W, strides=[1, 2, 2, 1], padding='SAME') + b)
        # current output is input for next layer
        current_input = output

    # %%
    # store the latent representation
    z = current_input
    # going reversely from the latent space and upsampling to the input size
    encoder.reverse()
    shapes.reverse()

    # %%
    # Build the decoder using the same weights
    for l, shape in enumerate(shapes):
        W_shape = encoder[l].get_shape().as_list()
        W = tf.Variable(init(W_shape))
        b = tf.Variable(init([W_shape[-2]]))
        # leaky_relu ( transpose_convolution ( input ))
        output = tf.nn.leaky_relu(tf.nn.conv2d_transpose(current_input, W,
                                                         tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                                                         strides=[1, 2, 2, 1], padding='SAME') + b)
        current_input = output

    # %%
    # The output of the last layer has the same size as input
    y = current_input
    # cost function measures pixel-wise difference of the x and reconstructed output of the network +
    # regularizer which encourages usage minimum number
    cost = tf.reduce_sum(tf.square(y - xTensor)) + lambdaL1 * tf.reduce_sum(tf.abs(z))

    # %%
    return {'x': x,
            'z': z,
            'y': y,
            'l': lambdaL1,
            'cost': cost}


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
