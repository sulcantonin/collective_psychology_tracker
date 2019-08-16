import cv2
import numpy as np

def drawROIs(img, rois):
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


def resizeListOfImages(list_of_images, w, h):
    # we want to return a tensor [height,width,3,frame]
    X = np.zeros((h, w, 3, len(list_of_images)))
    for f, img in enumerate(list_of_images):
        if img is None:
            continue
        X[..., f] = cv2.resize(img, (w, h))
    return X


def sigma_test(X, sigma_mult):
    return np.abs((X[:] - np.mean(X[:]))) > sigma_mult * np.std(X[:])


def cutFrame(frame, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    return frame[p1[1]:p2[1], p1[0]:p2[0], :]


def writeVolume2Video(V, filenameOut):
    width = V.shape[1]
    height = V.shape[0]
    vOut = cv2.VideoWriter(filenameOut, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, (width, height))

    for f in range(V.shape[-1]):
        vOut.write(V[..., f])
    vOut.release()

# normalization of the input images, it would be better if this remain mystery
def normalize(X):
    X = X.astype(np.float32) / 255
    for i in range(X.shape[-1]):
        X[..., i] = X[..., i] - np.mean(X[..., i].ravel())
        X[..., i] = X[..., i] / np.sqrt(np.sum(X[..., i].ravel() ** 2))
    return X