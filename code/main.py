from tracking import *
import tkinter as tk
import autoencoder_gui as aeg
import configparser

root = tk.Tk()

#
# loading configuration
#
config = configparser.ConfigParser()
config.read('./config.ini')

settings = dict()

settings['automatic_roi_selection'] = config.getboolean('default', 'automatic_roi_selection', fallback=1)
settings['automatic_roi_selection_sigma_mult'] = config.getfloat('default', 'automatic_roi_selection_sigma_mult',
                                                                 fallback=3)
settings['atomatic_roi_morph_disk_radius'] = config.getint('default','automatic_roi_morph_disk_radius',fallback=5)
settings['tracker'] = config.get('default','tracker', fallback='mil')

settings['tracker_video_output'] = config.getboolean('default', 'tracker_video_output', fallback=1)

def userLeftOutputEmpty(filenameIn, fileSuffix, fileType=None):
    path = filenameIn.split('/')[:-1]
    fname = filenameIn.split('/')[-1].split('.')[0]
    if fileType is None:
        fileType = filenameIn.split('/')[-1].split('.')[-1]

    return ''.join([p + '/' for p in path]) + fname + '_' + fileSuffix + '.' + fileType


def roiSelectionCallback():
    filenameIn = tk.filedialog.askopenfilename()
    # nothing chosen
    if len(filenameIn) == 0:
        return

    filenameOut = tk.filedialog.asksaveasfilename(title='output video file')

    if filenameOut is "":
        filenameOut = userLeftOutputEmpty(filenameIn, 'roi')

    roi(filenameIn, filenameOut, settings)


def trackingSelectionCallback():
    filenameIn = tk.filedialog.askopenfilename()
    # nothing chosen
    if len(filenameIn) == 0:
        return

    filenameOutVideo = tk.filedialog.asksaveasfilename(title='output video file')
    filenameOutCsv = tk.filedialog.asksaveasfilename(title='output csv file')

    if filenameOutVideo is "":
        filenameOutVideo = userLeftOutputEmpty(filenameIn, 'tracking')
    if filenameOutCsv is "":
        filenameOutCsv = userLeftOutputEmpty(filenameIn, 'tracking', 'csv')
    tracking_selection(filenameIn, filenameOutVideo, filenameOutCsv, settings)


def video2volumeSelectionCallback():
    filenameIn = tk.filedialog.askopenfilename()
    filenameOut = tk.filedialog.asksaveasfilename()

    if filenameOut is "":
        filenameOut = userLeftOutputEmpty(filenameIn, '', 'npy')

    video_to_volume(filenameIn, filenameOut)


def autoencoderCallback():
    autoencoder_window = tk.Toplevel(root)
    aeg.autoencoder_gui(autoencoder_window)


buttonROISelection = tk.Button(root, text="ROI Selection", command=roiSelectionCallback)
buttonTrackingSelection = tk.Button(root, text="Tracking Selection", command=trackingSelectionCallback)
buttonVideo2volumeSelection = tk.Button(root, text="Transform video into Volume",
                                        command=video2volumeSelectionCallback)
buttonAutoencoderSelection = tk.Button(root, text="Autoencoder", command=autoencoderCallback)

buttonROISelection.pack()
buttonTrackingSelection.pack()
buttonVideo2volumeSelection.pack()
buttonAutoencoderSelection.pack()
root.mainloop()
