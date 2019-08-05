import cv2
import numpy as np


class MOTTrackingROI:
    # list with bounding boxes with ROIs
    __ROIs = []
    # the clean, bound box free image
    __imageClean = None
    # location where user pressed current x and y
    __currentLButtonXYDown = [None, None]
    __currentROI = 0

    # sets a current roi at location current_roi
    def setROI(self, bbox, current_roi):
        if len(self.getROIs()) >= current_roi:
            self.__ROIs[current_roi] = bbox

    def getROIs(self):
        return self.__ROIs

    def setImage(self, image):
        self.__imageClean = image.copy()

    def getImage(self):
        if self.__imageClean is None:
            return None
        else:
            return self.__imageClean.copy()
    @staticmethod
    def drawROIs(img,rois):
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

    def selectROI(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.__currentLButtonXYDown[0] = x
            self.__currentLButtonXYDown[1] = y

        elif event == cv2.EVENT_LBUTTONUP:
            bbox = [None] * 4
            bbox[0] = np.minimum(self.__currentLButtonXYDown[0], x)
            bbox[1] = np.minimum(self.__currentLButtonXYDown[1], y)
            bbox[2] = np.maximum(self.__currentLButtonXYDown[0], x) - bbox[0]
            bbox[3] = np.maximum(self.__currentLButtonXYDown[1], y) - bbox[1]
            self.setROI(bbox, self.__currentROI)

    def __init__(self, img, rois = None, name="Select ROI"):
        self.__currentROI = 0
        if rois is None:
            self.__ROIs = [None] * 10
        else:
            self.__ROIs = rois
        self.setImage(img)

        cv2.namedWindow(name)
        cv2.setMouseCallback(name, self.selectROI)

        while True:
            cv2.imshow(name, MOTTrackingROI.drawROIs(img.copy(), self.getROIs()))
            key = cv2.waitKey(1)

            if ord('9') >= key >= ord('0'):
                self.__currentROI = int(chr(key))
            if ord('q') == key:
                break
        cv2.destroyAllWindows()
