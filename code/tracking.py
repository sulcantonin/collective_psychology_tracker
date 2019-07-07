import numpy as np
from tkinter import filedialog
import cv2
import tensorflow as tf
import numpy as np
import scipy.signal as signal
import peakutils
import csv


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
def tracking_selection(fname_in, fname_out_video, fname_out_csv):
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
        print('frame {0} of {1} frames'.format(frame + 1, len(frames)))

    w = int(w / frame)
    h = int(h / frame)

    v_out = cv2.VideoWriter(fname_out_video, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (w, h))
    csv_out = open(fname_out_csv, 'w')
    for frame in range(len(frames)):
        bbox = tracks[frame]
        if bbox is None:
            break

        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

        img = frames[frame][p1[1]:p2[1], p1[0]:p2[0], :]
        img = cv2.resize(img, (w, h))

        csv.writer(csv_out).writerow(bbox)

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
    nf = int(v_in.get(cv2.CAP_PROP_FRAME_COUNT) - 1)

    V = np.zeros((h, w, 3, nf), np.float32)
    for frame in range(nf):
        ret, V[..., frame] = v_in.read()
        print("frame {0} of {1}".format(frame + 1, nf))

    np.save(fname_out, V)
    v_in.release()
    return True





def low_pass_filter(x, N=2, Wn=0.01):
    # N = 2  # Filter order
    # Wn = 0.01  # Cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')

    # Second, apply the filter
    return signal.filtfilt(B, A, x)


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
