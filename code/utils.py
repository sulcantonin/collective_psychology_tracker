import cv2
import numpy as np


def draw_ROIs(img, rois):
    if img is not None and len(rois) > 0:
        for t, bbox in enumerate(rois):
            if bbox is not None:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                # visualisation
                cv2.rectangle(img, p1, p2, (255, 0, 0))
                cv2.putText(img, str(t), p1, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        return img
    return img


def resize_image_list(list_of_images, sz=None):
    # we want to return a tensor [height,width,3,frame]

    # calculating average width and height of no width and height is given
    if sz is None:
        h = []
        w = []

        # going though all the frames, storing width and height
        for f, frame in enumerate(list_of_images):
            if frame is not None:
                h = h + [frame.shape[0], ]
                w = w + [frame.shape[1], ]
        # the mean width and height
        h = np.nanmean(np.asarray(h)).astype(np.uint16)
        w = np.nanmean(np.asarray(w)).astype(np.uint16)
        sz = [h, w]
    X = np.zeros((sz[0], sz[1], 3, len(list_of_images)))

    for f, img in enumerate(list_of_images):
        if img is None:
            continue
        if img.shape[0] < 0 or img.shape[1] < 0:
            continue
        X[..., f] = cv2.resize(img, (sz[1], sz[0]))
    return X


def bbox2str(bboxes):
    out = ""
    for frame, bbox in enumerate(bboxes):
        if bbox is None:
            continue

        out = out + str(frame) + ',' + ','.join([str(b) for b in bbox]) + '\n'
    return out


def sigma_test(X, sigmaMult):
    return np.abs((X[:] - np.mean(X[:]))) > sigmaMult * np.std(X[:])


def cut_frame(frame, bbox):
    p1 = [int(bbox[0]), int(bbox[1])]
    p2 = [int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])]

    h = frame.shape[0]
    w = frame.shape[1]

    p1[0] = np.maximum(p1[0], 0)
    p1[1] = np.maximum(p1[1], 0)
    p2[0] = np.minimum(p2[0], w)
    p2[1] = np.minimum(p2[1], h)
    return frame[p1[1]:p2[1], p1[0]:p2[0], :]




def write_volume_to_video(V, filenameOut):
    width = V.shape[1]
    height = V.shape[0]
    vOut = cv2.VideoWriter(filenameOut, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, (width, height))

    for f in range(V.shape[-1]):
        vOut.write(V[..., f])
    vOut.release()


# normalization of the input images, it would be better if this remain mystery
def normalize(X):
    X = X.astype(np.float32) / X.max()
    for i in range(X.shape[-1]):
        X[..., i] = X[..., i] - np.mean(X[..., i].ravel())
        X[..., i] = X[..., i] / np.sqrt(np.sum(X[..., i].ravel() ** 2))
    return X
