from tracking import *
import tkinter as tk
import autoencodergui as aeg
import configparser

root = tk.Tk()

#
# loading configuration
#
config = configparser.ConfigParser()
config.read('./config.ini')

automatic_roi_selection = config.getboolean('default',
                                            'automatic_roi_selection',
                                            fallback=1)

automatic_roi_selection_sigma_mult = config.getfloat('default',
                                                     'automatic_roi_selection_sigma_mult',
                                                     fallback=3)
morph_disk_radius = config.getint('default',
                                  'morph_disk_radius',
                                  fallback=5)


def userLeftOutputEmpty(fname_in, fname_suffix, file_type=None):
    path = fname_in.split('/')[:-1]
    fname = fname_in.split('/')[-1].split('.')[0]
    if file_type is None:
        file_type = fname_in.split('/')[-1].split('.')[-1]

    return ''.join([p + '/' for p in path]) + fname + '_' + fname_suffix + '.' + file_type


def roiSelectionCallback():
    filenameIn = filedialog.askopenfilename()
    # nothing chosen
    if len(filenameIn) == 0:
        return

    filenameOut = filedialog.asksaveasfilename(title='output video file')

    if filenameOut is "":
        filenameOut = userLeftOutputEmpty(filenameIn, 'roi')

    roi(filenameIn, filenameOut, automatic_roi_selection, automatic_roi_selection_sigma_mult, morph_disk_radius)


def tracking_selection_callback():
    fname_in = filedialog.askopenfilename()
    # nothing chosen
    if len(fname_in) == 0:
        return

    filenameOutVideo = filedialog.asksaveasfilename(title='output video file')
    filenameOutCsv = filedialog.asksaveasfilename(title='output csv file')

    if filenameOutVideo is "":
        filenameOutVideo = userLeftOutputEmpty(fname_in, 'tracking')
    if filenameOutCsv is "":
        filenameOutCsv = userLeftOutputEmpty(fname_in, 'tracking', 'csv')
    trackingSelection(fname_in, filenameOutVideo, filenameOutCsv)


def video2volumeSelectionCallback():
    filenameIn = filedialog.askopenfilename()
    filenameOut = filedialog.asksaveasfilename()

    if filenameOut is "":
        filenameOut = userLeftOutputEmpty(filenameIn, '', 'npy')

    video2volumeSelection(filenameIn, filenameOut)


def autoencoderCallback():
    autoencoder_window = tk.Toplevel(root)
    aeg.autoencoderGUI(autoencoder_window)


button_roi_selection = tk.Button(root, text="ROI Selection", command=roiSelectionCallback)
button_tracking_selection = tk.Button(root, text="Tracking Selection", command=tracking_selection_callback)
button_video2volume_selection = tk.Button(root, text="Transform video into Volume",
                                          command=video2volumeSelectionCallback)
button_autoencoder_selection = tk.Button(root, text="Autoencoder", command=autoencoderCallback)

button_roi_selection.pack()
button_tracking_selection.pack()
button_video2volume_selection.pack()
button_autoencoder_selection.pack()
root.mainloop()
