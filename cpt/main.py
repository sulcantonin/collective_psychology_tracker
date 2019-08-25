from tracking import *
import tkinter as tk
import autoencoder_gui as aeg
import utils
import configparser

root = tk.Tk()

#
# loading configuration
#
configure_script_name = 'config.ini'
config = configparser.ConfigParser()
config.read(configure_script_name)

settings = dict()

settings['automatic_roi_selection'] = config.getboolean('default', 'automatic_roi_selection', fallback=1)
settings['automatic_roi_selection_sigma_mult'] = config.getfloat('default', 'automatic_roi_selection_sigma_mult',
                                                                 fallback=3)
settings['atomatic_roi_morph_disk_radius'] = config.getint('default', 'automatic_roi_morph_disk_radius', fallback=5)
settings['tracker'] = config.get('default', 'tracker', fallback='mil')

settings['tracker_video_output'] = config.getboolean('default', 'tracker_video_output', fallback=1)
settings['tracker_npyvolume_output'] = config.getboolean('default','tracker_npyvolume_output', fallback=1)

settings['autoencoder_gui_width'] = config.getint('default', 'autoencoder_gui_width', fallback=800)
settings['autoencoder_gui_height'] = config.getint('default','autoencoder_gui_height', fallback=600)


def roi_selection_callback():
    filename_in = tk.filedialog.askopenfilename()
    # nothing chosen
    if len(filename_in) == 0:
        return

    filename_out = tk.filedialog.asksaveasfilename(title='output video file')

    if filename_out is "":
        filename_out = utils.user_left_output_empty(filename_in, 'roi')

    roi(filename_in, filename_out, settings)


def tracking_selection_callback():
    filename_in = tk.filedialog.askopenfilename()
    # nothing chosen
    if len(filename_in) == 0:
        return

    filename_out_video = tk.filedialog.asksaveasfilename(title='output video file')
    filename_out_csv = tk.filedialog.asksaveasfilename(title='output csv file')

    if filename_out_video is "":
        filename_out_video = utils.user_left_output_empty(filename_in, 'tracking')
    if filename_out_csv is "":
        filename_out_csv = utils.user_left_output_empty(filename_in, 'tracking', 'csv')
    tracking_selection(filename_in, filename_out_video, filename_out_csv, settings)

def autoencoder_callback():
    autoencoder_window = tk.Toplevel(root,width = settings['autoencoder_gui_width'],height = settings['autoencoder_gui_height'])
    autoencoder_window.resizable(0, 0)
    aeg.autoencoder_gui(autoencoder_window)

def run():
    button_roi_selection = tk.Button(root, text="ROI Selection", command=roi_selection_callback)
    button_tracking_selection = tk.Button(root, text="Tracking Selection", command=tracking_selection_callback)
    button_autoencoder_selection = tk.Button(root, text="Autoencoder", command=autoencoder_callback)

    button_roi_selection.pack()
    button_tracking_selection.pack()
    button_autoencoder_selection.pack()
    root.mainloop()

run()
