import cv2
import numpy as np
from skimage import morphology
from skimage.measure import label, regionprops
import utils
# this function helps
from roigui import MOTTrackingROI


# automatically select the bounding box by calculating cumulative
# difference between all frames
def roiDetection(filename_in,
                 morph_disk_radius,
                 automatic_roi_selection_sigma_mult):
    vIn = cv2.VideoCapture(filename_in)
    # reading first frame
    ret, frame = vIn.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    frame_diff = np.zeros(frame.shape[0:2], np.float32)
    print('automatic roi selection, this may take a while ...')
    while True:
        ret, frame_new = vIn.read()
        if ret == 0:
            break
        frame_new = cv2.cvtColor(frame_new, cv2.COLOR_RGB2GRAY)
        frame_diff = frame_diff + (frame_new - frame) ** 2
        frame = frame_new
        print('.', end='')

    s = morphology.disk(morph_disk_radius)
    roiSigmaMask = utils.sigma_test(frame_diff, automatic_roi_selection_sigma_mult)
    roiSigmaMask = morphology.erosion(roiSigmaMask, s)
    roiSigmaMask = morphology.dilation(roiSigmaMask, s)
    # np.save('/home/as/r', roiSigmaMask)
    label_img = label(roiSigmaMask)
    regions = regionprops(label_img)
    if len(regions) > 0:
        bbox = regions[np.argmax([r.area for r in regions])].bbox
        # transforming from regionprops  (min_row, min_col, max_row, max_col) to (x,y,w,h)
        return [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]]
    else:
        return None


def roiSelection(fnameIn):
    vIn = cv2.VideoCapture(fnameIn)
    # reading first frame
    ret, frameMean = vIn.read()
    frameMean = cv2.cvtColor(frameMean, cv2.COLOR_RGB2GRAY)
    n_frames = int(vIn.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = vIn.read()
        if ret == 0:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = frame.astype(np.float32)
        frame /= np.max(frame[:])
        frameMean = frameMean + frame
    frameMean = frameMean / n_frames
    frameMeanNormalized = frameMean / np.max(frameMean[:])
    cv2.namedWindow('roi', 2)
    # cv2.resizeWindow('roi', 600, 600)
    bbox = cv2.selectROI("roi", np.max(frameMeanNormalized[:]) - frameMeanNormalized, False, False)
    return bbox


def roi(filenameIn, filenameOut,
        automaticROISelection,
        automaticROISelectionSigmaMult=3,
        morph_disk_radius=5):
    # automatic selection of bounding box
    if automaticROISelection > 0:
        bbox = roiDetection(filenameIn, morph_disk_radius, automaticROISelectionSigmaMult)
    else:
        bbox = roiSelection(filenameIn)

    # opening reader for the video
    vIn = cv2.VideoCapture(filenameIn)
    n_frames = int(vIn.get(cv2.CAP_PROP_FRAME_COUNT))
    # reading first frame
    ret, frame = vIn.read()

    # number of frames of the input video
    fps = vIn.get(cv2.CAP_PROP_FPS)
    # the output video
    vOut = cv2.VideoWriter(filenameOut, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (bbox[2], bbox[3]))

    while ret:
        frameROI = utils.cutFrame(frame, bbox)
        ret, frame = vIn.read()
        cv2.imshow('roi', frameROI)
        cv2.waitKey(1)
        vOut.write(frameROI)
    cv2.destroyAllWindows()
    vOut.release()
    return True


def str2tracker(trackerName):
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
    return OPENCV_OBJECT_TRACKERS[trackerName]()


def initTrackers(trackers, frame, rois, tracker):
    for i, roi in enumerate(rois):
        # update only a newly selected rois
        if roi is not None:
            trackers[i] = str2tracker(tracker)  # cv2.TrackerMIL_create()
            trackers[i].init(frame, tuple(roi))
    return trackers


# this function returns list of detected frames for the window selected in the first selected frame
def trackingSelection(filenameIn, filenameOutVideo, filenameOutCsv, trackerName):
    # opening reader for the video
    vIn = cv2.VideoCapture(filenameIn)

    nFrames = int(vIn.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vIn.get(cv2.CAP_PROP_FPS)
    nTrackers = 10

    ret, frame = vIn.read()
    # initial bounding boxes around the
    MOTGUI = MOTTrackingROI(frame, rois=None, name="Select Bounding Boxes around Tracked Objects")
    initialROIs = MOTGUI.getROIs()

    # initializing tracker, 10 trackers x nFrames
    tracks = [[None] * nTrackers for _ in range(nFrames)]
    trackers = [None] * nTrackers

    tracks[0] = initialROIs
    trackers = initTrackers(trackers, frame, initialROIs, trackerName)

    frameId = 1  # first one was used for initializaiton
    timeout = 0
    while True:
        vIn.set(cv2.CAP_PROP_POS_FRAMES, frameId)
        ret, frame = vIn.read()
        # end of the video or fail
        if ret == 0:
            break

        for i, tracker in enumerate(trackers):
            # tracker not initialized
            if tracker is None:
                continue
            ok, bbox = tracker.update(frame)
            if ok:
                tracks[frameId][i] = bbox

        cv2.imshow("tracking", utils.drawROIs(frame, tracks[frameId]))

        key = cv2.waitKey(timeout)
        # go backwards
        if key == ord('a'):
            frameId = np.maximum(0, frameId - 1)
            vIn.set(cv2.CAP_PROP_POS_FRAMES, frameId)
        # go forward
        if key == ord('d'):
            frameId = np.minimum(frameId + 1, nFrames)
            vIn.set(cv2.CAP_PROP_POS_FRAMES, frameId)
        if key == ord('s'):
            MOTGUI = MOTTrackingROI(frame, tracks[frameId])
            initTrackers(trackers, frame, MOTGUI.getROIs(), trackerName)
        # play
        if key == ord('p'):
            if timeout == 0:
                timeout = 1
            else:
                timeout = 0

        # if the autoplay is active, add the frames automatically
        if timeout != 0:
            frameId = frameId + 1

        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    # exporting data
    vIn.release()
    vIn = cv2.VideoCapture(filenameIn)
    nFrames = int(vIn.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vIn.get(cv2.CAP_PROP_FPS))
    vInWidth = int(vIn.get(cv2.CAP_PROP_FRAME_WIDTH))
    vInHeight = int(vIn.get(cv2.CAP_PROP_FRAME_HEIGHT))

    vOut = cv2.VideoWriter(filenameOutVideo, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (vInWidth, vInHeight))
    volumes = [[None] * nFrames for _ in range(nTrackers)]

    for frameId in range(nFrames):
        ret, frame = vIn.read()
        for t in range(nTrackers):
            bbox = tracks[frameId][t]
            if bbox is None:
                continue
            volumes[t][frameId] = utils.cutFrame(frame.copy(), bbox)

        vOut.write(utils.drawROIs(frame, tracks[frameId]))
    vOut.release()

    # saving the volumes
    tracksTransposed = [list(i) for i in zip(*tracks)]
    for t in range(nTrackers):
        # calculating average width and height
        h = np.zeros(nFrames)
        w = np.zeros(nFrames)

        # going though all the frames, storing width and height
        for frameId in range(nFrames):
            if volumes[t][frameId] is not None:
                h[frameId] = volumes[t][frameId].shape[0]
                w[frameId] = volumes[t][frameId].shape[1]
            else:
                h[frameId] = np.nan
                w[frameId] = np.nan
        # the mean width and height
        h = np.nanmean(h).astype(np.uint16)
        w = np.nanmean(w).astype(np.uint16)
        # resizing the volume to a specific size
        V = utils.resizeListOfImages(volumes[t], w, h)
        # saving if there is anything to save
        if V.ravel().sum() > 0:
            frameSum = V.reshape((-1, V.shape[-1])).sum(0)
            V = V[..., np.logical_and(frameSum > 0, np.isfinite(frameSum))]

            filenameOutputVolume = filenameOutVideo.split('.')[:-1][0] + '_' + str(t)
            np.save(filenameOutputVolume, V)
            print('saved ' + filenameOutputVolume)
            #
            filenameOutputBBOXCsv = filenameOutCsv.split('.')[:-1][0] + '_' + str(t) + '.csv'
            tt = tracksTransposed[t]
            # saving only the non empty 4-element bounding boxes
            FINSH THIS STUFF
            # tt = tt[ [tt[i] is not None for i in range(len(tt))] ]
            # np.savetxt(filenameOutputBBOXCsv, np.array(tt), delimiter=",")

            # np.ndarray(tracksTransposed[t]).tofile(filenameOutputCsv, sep=',', format='%i')
            print('saved ' + filenameOutputBBOXCsv)


def video2volumeSelection(fnameIn, fnameOut):
    # opening reader for the video
    v_in = cv2.VideoCapture(fnameIn)

    # reading first frame
    ret, frame = v_in.read()

    w = int(v_in.get(3))
    h = int(v_in.get(4))
    nf = int(v_in.get(cv2.CAP_PROP_FRAME_COUNT) - 1)

    V = np.zeros((h, w, 3, nf), np.float32)
    for frame in range(nf):
        ret, V[..., frame] = v_in.read()
        print("frame {0} of {1}".format(frame + 1, nf))

    np.save(fnameOut, V)
    v_in.release()
    return True


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
