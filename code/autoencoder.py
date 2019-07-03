import numpy as np
import tkinter as tk
from tkinter import filedialog
import cv2
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import scipy.signal as signal
import peakutils

# after the window is determined, all detected frames need to be resized to one specific size
width = 64
height = 64


# the default parameters means that thetre are 3 layers, each with 10 feature maps, convolution size 3x3,
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
    x_tensor = tf.reshape(x, [-1, x_dim, y_dim, n_filters[0]])
    # defining the reguralizer for the latent space
    # this value of value of lambda_L1 the less peaks you get
    lambda_L1 = tf.placeholder(tf.float32, 1, name='lambda')

    current_input = x_tensor

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
    cost = tf.reduce_sum(tf.square(y - x_tensor)) + lambda_L1 * tf.reduce_sum(tf.abs(z))

    # %%
    return {'x': x,
            'z': z,
            'y': y,
            'l': lambda_L1,
            'cost': cost}


# Traning procedure
def train(ae, X, l):
    # learning rate, the larger value -> faster traning, but if the value is too large the
    # training algorithm won't converge
    learning_rate = 0.001
    # consider this as a blackbox, we define a training algorithm on the defined cost sum( (x - y)^2) + abs(z)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    # %%
    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # %%
    # 500 training iterations, if we have enough data it is STRONGLY RECOMMENED to not to use whole tranining dataset,
    # but randomly select a subset for each epoch
    n_epochs = 100
    for epoch_i in range(n_epochs):
        sess.run(optimizer, feed_dict={ae['x']: X, ae['l']: [l]})
        print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: X, ae['l']: [l]}))
    return sess


# this function helps
def resize_list_of_images(list_of_images, w, h):
    # we want to return a tensor [height,width,3,frame]
    X = np.zeros((h, w, 3, len(list_of_images)))
    for f in range(len(list_of_images)):
        X[..., f] = cv2.resize(list_of_images[f], (w, h))
    return X


def roi_selection(fname_in, fname_out):
    # opening reader for the video
    v_in = cv2.VideoCapture(fname_in)

    # reading first frame
    ret, frame = v_in.read()

    # user selection of the first frame
    bbox = cv2.selectROI("roi", frame)

    # number of frames of the input video
    fps = v_in.get(cv2.CAP_PROP_FPS)
    # the output video
    v_out = cv2.VideoWriter(fname_out, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (bbox[2], bbox[3]))

    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

    while ret:
        frame_roi = frame[p1[1]:p2[1], p1[0]:p2[0], :]
        ret, frame = v_in.read()
        cv2.imshow('roi', frame_roi)
        cv2.waitKey(1)
        v_out.write(frame_roi)
    cv2.destroyAllWindows()
    v_out.release()
    return True


# this function returns list of detected frames for the window selected in the first selected frame
def tracking_selection(fname_in, fname_out):
    # opening reader for the video
    v_in = cv2.VideoCapture(fname_in)
    fps = v_in.get(cv2.CAP_PROP_FPS)

    # initializing tracker
    tracker = cv2.TrackerMIL_create()

    frames = []
    tracks = []

    # buffering frames in memory
    while True:
        ret, frame = v_in.read()
        if not ret:
            break
        frames += [frame, ]
        tracks += [None, ]
    n_frames = len(frames)
    frame = 0

    img = frames[frame].copy()

    # select the region of interest for tracker
    tracks[0] = cv2.selectROI('frame', img)
    # initialize the tracker
    tracker.init(img, tracks[0])

    w = tracks[0][2]
    h = tracks[0][3]

    while True:
        img = frames[frame].copy()

        ok, bbox = tracker.update(img)
        print(bbox, ok)

        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

        # visualisation
        cv2.rectangle(img, p1, p2, (255, 0, 0))
        cv2.imshow('frame', img)

        key = cv2.waitKey()
        # move backwards
        if key == ord('a'):
            frame = max(frame - 1, 0)
        # move forward
        if key == ord('d'):
            frame = min(frame + 1, n_frames - 1)
        # select bounding box
        if key == ord('s'):
            img = frames[frame].copy()
            bbox = cv2.selectROI('frame', img)
            tracker = cv2.TrackerMIL_create()
            tracker.init(img, bbox)
        # quit
        if key == ord('q'):
            break

        w += bbox[2]
        h += bbox[3]
        tracks[frame] = bbox
        print('frame {0} of {1} frames'.format(frame, len(frames)))

    w = int(w / frame)
    h = int(h / frame)

    v_out = cv2.VideoWriter(fname_out, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (w, h))

    for frame in range(len(frames)):
        bbox = tracks[frame]
        if bbox is None:
            break

        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

        img = frames[frame][p1[1]:p2[1], p1[0]:p2[0], :]
        img = cv2.resize(img, (w, h))

        v_out.write(img)
    v_out.release()
    cv2.destroyAllWindows()
    return True


def video2volume_selection(fname_in, fname_out):
    # opening reader for the video
    v_in = cv2.VideoCapture(fname_in)

    # reading first frame
    ret, frame = v_in.read()

    w = int(v_in.get(3))
    h = int(v_in.get(4))
    nf = int(v_in.get(cv2.CAP_PROP_FRAME_COUNT)-1)

    V = np.zeros((h, w, 3, nf), np.float32)
    for frame in range(nf):
        ret, V[..., frame] = v_in.read()
        print("frame {0} of {1}".format(frame,nf))

    np.save(fname_out, V)
    v_in.release()
    return True


# normalization of the input images, it would be better if this remain mystery
def normalize(X):
    X = X.astype(np.float32) / 255
    for i in range(X.shape[-1]):
        X[..., i] = X[..., i] - np.mean(X[..., i].ravel())
        X[..., i] = X[..., i] / np.sqrt(np.sum(X[..., i].ravel() ** 2))
    return X


def low_pass_filter(x, N=2, Wn=0.01):
    # N = 2  # Filter order
    # Wn = 0.01  # Cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')

    # Second, apply the filter
    return signal.filtfilt(B, A, x)


def just_another_filter(x, band=60):
    xf = np.fft.rfft(x)
    xf[band:] = 0
    return np.fft.irfft(xf)


def tsne(x):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    # folder = '/home/sulcanto/Downloads/alex/anka/isolated feeding baseline 1/feeding_2_30sec/'
    # fname = 'feeding_2_30sec_z.npy'

    # z = np.load(folder + fname)
    # z = np.reshape((z.shape[0],-1))

    x_manifold2 = TSNE(n_components=2).fit_transform(x)

    plt.plot(z[:, 0], z[:, 1])
    plt.show()

    # clustering = KMeans(n_clusters=6,random_state=0).fit(z_manifold2)
    # c = clustering.predict(z_manifold2)

    n_c = 6
    clustering = GaussianMixture(n_components=n_c).fit(z_manifold2)
    d = np.zeros(n_c, np.float32)
    for i in range(0, n_c):
        d[i] = np.prod(np.linalg.eigvals(clustering.covariances_[i]))
        # plt.plot(z_manifold2[c == i,0],z_manifold2[c == i,1],'.')

    c = clustering.predict(z_manifold2)
    for i in range(0, n_c):
        ind = np.where(c == i)[0]
        plt.plot(ind, np.ones(ind.shape) * i)


def roi_selection_callback():
    fname_in = filedialog.askopenfilename()
    fname_out = filedialog.asksaveasfilename()
    roi_selection(fname_in, fname_out)


def tracking_selection_callback():
    fname_in = filedialog.askopenfilename()
    fname_out = filedialog.asksaveasfilename()
    tracking_selection(fname_in, fname_out)


def video2volume_selection_callback():
    fname_in = filedialog.askopenfilename()
    fname_out = filedialog.asksaveasfilename()
    video2volume_selection(fname_in, fname_out)

def train_encoder_selection_callback():
    tf.set_random_seed(2)
    # autoencoder
    volume_in = filedialog.askopenfilename()

    X = np.load(volume_in)
    X = normalize(X)

    width = X.shape[1]
    height = X.shape[0]
    n_frames = X.shape[-1]

    ae = autoencoder(width, height)
    X = X.transpose((3, 0, 1, 2))

    sess = train(ae, X.reshape((n_frames, -1)), 25.0)


root = tk.Tk()

button_roi_selection = tk.Button(root, text="ROI Selection", command=roi_selection_callback)
button_tracking_selection = tk.Button(root, text="Tracking Selection", command=tracking_selection_callback)
button_video2volume_selection = tk.Button(root, text="Transform video into Volume", command=video2volume_selection_callback)
button_train_encoder_selection = tk.Button(root, text="Train Autoencoder", command=train_encoder_selection_callback)


button_roi_selection.pack()
button_tracking_selection.pack()
button_video2volume_selection.pack()
button_train_encoder_selection.pack()
root.mainloop()


'''
# saver = tf.train.Saver()
# sess = tf.Session()
# saver.save(sess, "./data/model.ckpt")
z = sess.run(ae['z'], feed_dict={ae['x']: X.reshape((n_frames, -1))})
y = sess.run(ae['y'], feed_dict={ae['x']: X.reshape((n_frames, -1))})
np.save(fname[:-4] + '_z', z)
np.save(fname[:-4] + '_y', y)

# visualisation part
# latent space
z = np.load(fname[:-4] + '_z.npy')
# input
vin = cv2.VideoCapture(fname)
# number of frames, the last column
n_frames = z.shape[0]
# calculating mean of absolute value of the latent code
z_mean = np.mean(np.abs(z.reshape(n_frames, -1)), 1)
'''

## this part is for heartbeat sensing
'''
z_mean_lp = low_pass_filter(z_mean, 2, 0.05)
z_mean_hp = np.abs(z_mean - z_mean_lp)
d_z_mean_hp = z_mean_hp[:-1] - z_mean_hp[1:]

plt.subplot(3, 1, 1)
plt.plot(z_mean)
plt.title('sum(|z|)')

plt.subplot(3, 1, 2)
plt.plot(z_mean_hp)
plt.title('high freq z_mean')

plt.subplot(3, 1, 3)
plt.plot(d_z_mean_hp)
plt.title('derivative of high freq.z_mean')
plt.savefig(fname[:-4] + '_graphs.png')

plt.show()

beat = d_z_mean_hp > 0.0 

fourcc = cv2.VideoWriter_fourcc(*'XVID')
vout = None

for b in range(len(beat)):
    ret, frame = vin.read()

    if vout is None:
        vout = cv2.VideoWriter(fname[:-4] + '_res.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))

    if beat[b]:

        frame[0:20, 0:20, 2] = 255
        frame[0:20, 0:20, 0:2] = 0
    vout.write(frame)

vout.release()
np.save(fname[:-4] + '_beat', beat)
np.save(fname[:-4] + '_z_mean_hp', z_mean_hp)
np.save(fname[:-4] + 'd_z_mean_hp', d_z_mean_hp)
'''

'''
# breath detection
z = np.load(fname[:-4] + '_z.npy')
vin = cv2.VideoCapture(fname)
z_mean = np.mean(np.abs(z.reshape(z.shape[0], -1)), 1)

np.save(fname[:-4] + '_z_mean', z_mean)

z_mean_filtered = just_another_filter(z_mean, band)
np.save(fname[:-4] + '_z_mean_filtered', z_mean_filtered)

breath_indices = peakutils.indexes(z_mean_filtered, 0)
breath = np.zeros_like(z_mean)
breath[breath_indices] = 1

plt.subplot(2, 1, 1)
plt.plot(z_mean)
plt.title('sum(|z|)')

plt.subplot(2, 1, 2)
plt.plot(z_mean_filtered)
plt.plot(breath_indices, z_mean_filtered[breath_indices], 'ro')
plt.title('filtered sum(|z|)')

plt.savefig(fname[:-4] + '_graphs.png')
plt.show()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
vout = None

for b in range(len(breath)):
    ret, frame = vin.read()

    if vout is None:
        vout = cv2.VideoWriter(fname[:-4] + '_res.avi', fourcc, vin.get(cv2.CAP_PROP_FPS),
                               (frame.shape[1], frame.shape[0]))

    if breath[b]:
        frame[bbox[1] - 20:bbox[1], bbox[0] - 20:bbox[0], 2] = 255
        frame[bbox[1] - 20:bbox[1], bbox[0] - 20:bbox[0], 0:2] = 0
    vout.write(frame)

vout.release()

print(fname.split('/')[-3])
print(fname.split('/')[-1])
print("beats:" + str(len(breath_indices)))
print("nframes:" + str(vin.get(cv2.CAP_PROP_FRAME_COUNT)))
print("fps:" + str(vin.get(cv2.CAP_PROP_FPS)))

np.save(fname[:-4] + '_breath_indices.npy', breath_indices)

'''