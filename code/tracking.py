import cv2
import numpy as np
from skimage import morphology
from skimage.measure import label, regionprops
import utils
# this function helps
from roigui import MOTTrackingROI


# automatically select the bounding box by calculating cumulative
# difference between all frames
def roi_detection(filename_in, settings):
    v_in = cv2.VideoCapture(filename_in)
    # reading first frame
    ret, frame = v_in.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    frame_diff = np.zeros(frame.shape[0:2], np.float32)
    print('automatic roi selection, this may take a while ...')
    while True:
        ret, frame_new = v_in.read()
        if ret == 0:
            break
        frame_new = cv2.cvtColor(frame_new, cv2.COLOR_RGB2GRAY)
        frame_diff = frame_diff + (frame_new - frame) ** 2
        frame = frame_new
        print('.', end='')

    s = morphology.disk(settings['automatic_roi_morph_disk_radius'])
    roi_sigma_mask = utils.sigma_test(frame_diff, settings['automatic_roi_selection_sigma_mult'])
    roi_sigma_mask = morphology.erosion(roi_sigma_mask, s)
    roi_sigma_mask = morphology.dilation(roi_sigma_mask, s)
    # np.save('/home/as/r', roi_sigma_mask)
    label_img = label(roi_sigma_mask)
    regions = regionprops(label_img)
    if len(regions) > 0:
        bbox = regions[np.argmax([r.area for r in regions])].bbox
        # transforming from regionprops  (min_row, min_col, max_row, max_col) to (x,y,w,h)
        return [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]]
    else:
        return None


# this function serves for a manual selection of the bounding box from the
# averaged temprial image differentiation
def roi_selection(fname_in):
    v_in = cv2.VideoCapture(fname_in)
    # reading first frame
    ret, frame_mean = v_in.read()
    frame_mean = cv2.cvtColor(frame_mean, cv2.COLOR_RGB2GRAY)
    n_frames = int(v_in.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = v_in.read()
        if ret == 0:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = frame.astype(np.float32)
        frame /= np.max(frame[:])
        frame_mean = frame_mean + frame
    frame_mean = frame_mean / n_frames
    frame_mean_normalized = frame_mean / np.max(frame_mean[:])
    cv2.namedWindow('roi', 2)  # automatic sizing of the window
    bbox = cv2.selectROI("roi", np.max(frame_mean_normalized[:]) - frame_mean_normalized, False, False)
    return bbox


def roi(filename_in, filename_out, settings):
    # automatic selection of bounding box
    if settings['automatic_roi_selection'] > 0:
        bbox = roi_detection(filename_in, settings)
    else:
        bbox = roi_selection(filename_in)

    # opening reader for the video
    v_in = cv2.VideoCapture(filename_in)
    n_frames = int(v_in.get(cv2.CAP_PROP_FRAME_COUNT))
    # reading first frame
    ret, frame = v_in.read()

    # number of frames of the input video
    fps = v_in.get(cv2.CAP_PROP_FPS)
    # the output video
    v_out = cv2.VideoWriter(filename_out, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (bbox[2], bbox[3]))

    while ret:
        frameROI = utils.cut_frame(frame, bbox)
        ret, frame = v_in.read()
        cv2.imshow('roi', frameROI)
        cv2.waitKey(1)
        v_out.write(frameROI)
    cv2.destroyAllWindows()
    v_out.release()
    return True


def str2tracker(tracker_name):
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }

    # grab the appropriate object tracker using our dictionary of
    # OpenCV object tracker objects
    return OPENCV_OBJECT_TRACKERS[tracker_name]()


def init_trackers(trackers, frame, rois, tracker):
    for i, roi in enumerate(rois):
        # update only a newly selected rois
        if roi is not None:
            trackers[i] = str2tracker(tracker)  # cv2.TrackerMIL_create()
            trackers[i].init(frame, tuple(roi))
    return trackers


# this function returns list of detected frames for the window selected in the first selected frame
def tracking_selection(filename_in, filename_out_video, filename_out_csv, settings):
    # opening reader for the video
    v_in = cv2.VideoCapture(filename_in)

    n_frames = int(v_in.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = v_in.get(cv2.CAP_PROP_FPS)
    n_trackers = 10

    ret, frame = v_in.read()
    # initial bounding boxes around the
    MOTGUI = MOTTrackingROI(frame, rois=None, name="Select Bounding Boxes around Tracked Objects")
    initial_roi = MOTGUI.getROIs()

    # initializing tracker, 10 trackers x nFrames
    tracks = [[None] * n_trackers for _ in range(n_frames)]
    trackers = [None] * n_trackers

    tracks[0] = initial_roi
    trackers = init_trackers(trackers, frame, initial_roi, settings['tracker'])

    frame_id = 1  # first one was used for initializaiton
    timeout = 0
    while True:
        v_in.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = v_in.read()
        # end of the video or fail
        if ret == 0:
            break

        for i, tracker in enumerate(trackers):
            # tracker not initialized
            if tracker is None:
                continue
            ok, bbox = tracker.update(frame)
            if ok:
                tracks[frame_id][i] = bbox

        cv2.imshow("tracking", utils.draw_ROIs(frame, tracks[frame_id]))

        key = cv2.waitKey(timeout)
        # go backwards
        if key == ord('a'):
            frame_id = np.maximum(0, frame_id - 1)
            v_in.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        # go forward
        if key == ord('d'):
            frame_id = np.minimum(frame_id + 1, n_frames)
            v_in.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        if key == ord('s'):
            MOTGUI = MOTTrackingROI(frame, tracks[frame_id])
            init_trackers(trackers, frame, MOTGUI.getROIs(), settings['tracker'])
        # play
        if key == ord('p'):
            if timeout == 0:
                timeout = 1
            else:
                timeout = 0

        # if the autoplay is active, add the frames automatically
        if timeout != 0:
            frame_id = frame_id + 1

        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    # exporting data
    v_in.release()
    v_in = cv2.VideoCapture(filename_in)
    n_frames = int(v_in.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(v_in.get(cv2.CAP_PROP_FPS))
    v_in_width = int(v_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_in_height = int(v_in.get(cv2.CAP_PROP_FRAME_HEIGHT))

    volumes = [[None] * n_frames for _ in range(n_trackers)]

    if settings['tracker_video_output']:
        v_out = cv2.VideoWriter(filename_out_video, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (v_in_width, v_in_height))

    for frame_id in range(n_frames):
        ret, frame = v_in.read()
        for t in range(n_trackers):
            bbox = tracks[frame_id][t]
            if bbox is None:
                continue
            volumes[t][frame_id] = utils.cut_frame(frame.copy(), bbox)
        if settings['tracker_video_output']:
            v_out.write(utils.draw_ROIs(frame, tracks[frame_id]))

    if settings['tracker_video_output']:
        v_out.release()

    #
    # saving the volumes
    #
    # list of frames  where each item is list of tracks to list of tracers where each item is a list of frames
    tracks_transposed = [list(i) for i in zip(*tracks)]
    for t in range(n_trackers):
        V = utils.resize_image_list(volumes[t])

        # saving the volume
        if V.ravel().sum() > 0:  # testing if the volume is not empty, if sum(V[:]) == 0, then there is nothing to save
            frameSum = V.reshape((-1, V.shape[-1])).sum(0)

            # filtering out the ignored frames
            V = V[..., np.logical_and(frameSum > 0, np.isfinite(frameSum))]

            filenameOutputVolume = filename_out_video.split('.')[:-1][0] + '_' + str(t)
            np.save(filenameOutputVolume, V)
            print('saved ' + filenameOutputVolume)
            #

            filename_output_bbox_csv = filename_out_csv.split('.')[:-1][0] + '_' + str(t) + '.csv'
            tt = tracks_transposed[t]
            # saving only the non empty 4-element bounding boxes
            f = open(filename_output_bbox_csv, 'w')
            f.write(utils.bbox2str(tt))
            f.close()
            print('saved ' + filename_output_bbox_csv)


def video_to_volume(filenameIn, filenameOut):
    # opening reader for the video
    v_in = cv2.VideoCapture(filenameIn)

    # reading first frame
    ret, frame = v_in.read()

    w = int(v_in.get(3))
    h = int(v_in.get(4))
    nf = int(v_in.get(cv2.CAP_PROP_FRAME_COUNT) - 1)

    V = np.zeros((h, w, 3, nf), np.float32)
    for frame in range(nf):
        ret, V[..., frame] = v_in.read()
        print("frame {0} of {1}".format(frame + 1, nf))

    np.save(filenameOut, V)
    v_in.release()
    return True


'''
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
