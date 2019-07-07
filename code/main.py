from tracking import *
import tkinter as tk
import autoencoder_gui as aeg
root = tk.Tk()


def user_left_output_empty(fname_in, fname_suffix, file_type=None):
    path = fname_in.split('/')[:-1]
    fname = fname_in.split('/')[-1].split('.')[0]
    if file_type is None:
        file_type = fname_in.split('/')[-1].split('.')[-1]

    return ''.join([p + '/' for p in path]) + fname + '_' + fname_suffix + '.' + file_type


def roi_selection_callback():
    fname_in = filedialog.askopenfilename()
    fname_out = filedialog.asksaveasfilename(title='output video file')
    if fname_out is "":
        fname_out = user_left_output_empty(fname_in, 'roi')
    roi_selection(fname_in, fname_out)


def tracking_selection_callback():
    fname_in = filedialog.askopenfilename()
    fname_out_video = filedialog.asksaveasfilename(title='output video file')
    fname_out_csv = filedialog.asksaveasfilename(title='output csv file')
    if fname_out_video is "":
        fname_out_video = user_left_output_empty(fname_in, 'tracking')
    if fname_out_csv is "":
        fname_out_csv = user_left_output_empty(fname_in, 'tracking', 'csv')
    tracking_selection(fname_in, fname_out_video, fname_out_csv)


def video2volume_selection_callback():
    fname_in = filedialog.askopenfilename()
    fname_out = filedialog.asksaveasfilename()

    if fname_out is "":
        fname_out = user_left_output_empty(fname_in, '', 'npy')

    video2volume_selection(fname_in, fname_out)

def autoencoder_callback():
    autoencoder_window = tk.Toplevel(root)
    aeg.autoencoder_gui(autoencoder_window)

button_roi_selection = tk.Button(root, text="ROI Selection", command=roi_selection_callback)
button_tracking_selection = tk.Button(root, text="Tracking Selection", command=tracking_selection_callback)
button_video2volume_selection = tk.Button(root, text="Transform video into Volume", command=video2volume_selection_callback)
button_autoencoder_selection = tk.Button(root, text="Autoencoder", command=autoencoder_callback)


button_roi_selection.pack()
button_tracking_selection.pack()
button_video2volume_selection.pack()
button_autoencoder_selection.pack()
root.mainloop()