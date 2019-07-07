import peakutils
import tensorflow as tf
import numpy as np
import tkinter as tk
import autoencoder as ae

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import filedialog


class autoencoder_gui:

    def __init__(self, window):
        self.window = window

        self.load_input_button = tk.Button(window, text="Load NPY Input", command=self.load_input_callback)
        self.load_input_button.grid(row=0, column=0)
        self.load_input_text = tk.Text(window, height=1, width=20)
        self.load_input_text.grid(row=0, column=1)

        self.filter_sizes_text = tk.Text(window, height=1, width=20)
        self.filter_sizes_text.grid(row=1, column=0)
        self.filter_sizes_text.configure(state='normal')
        self.filter_sizes_text.insert(tk.END, 'Filter Sizes:')
        self.filter_sizes_text.configure(state='disabled')

        self.filter_sizes_string = tk.StringVar()
        self.filter_sizes_string.set('3, 3, 3, 3')
        self.filter_sizes_entry = tk.Entry(window, textvariable=self.filter_sizes_string)
        self.filter_sizes_entry.grid(row=1, column=1)

        self.n_filters_text = tk.Text(window, height=1, width=20)
        self.n_filters_text.grid(row=2, column=0)
        self.n_filters_text.configure(state='normal')
        self.n_filters_text.insert(tk.END, 'Filter Sizes:')
        self.n_filters_text.configure(state='disabled')
        self.n_filters_string = tk.StringVar()
        self.n_filters_string.set('3, 10, 10, 10')
        self.n_filters_entry = tk.Entry(window, textvariable=self.n_filters_string)
        self.n_filters_entry.grid(row=2, column=1)

        self.lambda_l1_text = tk.Text(window, height=1,width=20)
        self.lambda_l1_text.grid(row=3,column = 0)
        self.lambda_l1_text.configure(state='normal')
        self.lambda_l1_text.insert(tk.END, 'L1 Reguralization on Latent Code:')
        self.lambda_l1_text.configure(state='disabled')
        self.lambda_l1_string = tk.StringVar()
        self.lambda_l1_string.set('100.0')
        self.lambda_l1_entry = tk.Entry(window, textvariable=self.lambda_l1_string)
        self.lambda_l1_entry.grid(row=3, column=1)

        self.init_autoencoder_button = tk.Button(window, text="Initialize Network",
                                                 command=self.init_autoencoder_callback)
        self.init_autoencoder_button.grid(row=4, column=0, columnspan = 2)

        self.learning_rate_text = tk.Text(window, height=1,width=20)
        self.learning_rate_text.grid(row=5,column = 0)
        self.learning_rate_text.configure(state='normal')
        self.learning_rate_text.insert(tk.END, 'Learning Rate:')
        self.learning_rate_text.configure(state='disabled')
        self.learning_rate_string = tk.StringVar()
        self.learning_rate_string.set('0.001')
        self.learning_rate_entry = tk.Entry(window, textvariable=self.learning_rate_string)
        self.learning_rate_entry.grid(row=5, column=1)

        self.n_epochs_text = tk.Text(window, height=1,width=20)
        self.n_epochs_text.grid(row=6,column = 0)
        self.n_epochs_text.configure(state='normal')
        self.n_epochs_text.insert(tk.END, 'Traning Epochs:')
        self.n_epochs_text.configure(state='disabled')
        self.n_epochs_string = tk.StringVar()
        self.n_epochs_string.set('20')
        self.n_epochs_entry = tk.Entry(window, textvariable=self.n_epochs_string)
        self.n_epochs_entry.grid(row=6, column=1)

        self.band_text = tk.Text(window, height=1,width=20)
        self.band_text.grid(row=7,column = 0)
        self.band_text.configure(state='normal')
        self.band_text.insert(tk.END, 'Band:')
        self.band_text.configure(state='disabled')
        self.band_string = tk.StringVar()
        self.band_string.set('25')
        self.band_entry = tk.Entry(window, textvariable=self.band_string)
        self.band_entry.grid(row=7, column=1)

        self.train_autoencoder_button = tk.Button(window, text="Train Network", command=self.train_autoencoder_callback)
        self.train_autoencoder_button.grid(row=8, column=0, columnspan = 2)

        self.latent_code_figure = Figure() # figsize=(6, 6)
        self.latent_code_canvas = FigureCanvasTkAgg(self.latent_code_figure, master=window)
        self.latent_code_canvas.get_tk_widget().grid(row = 0,column = 2,rowspan = 8, columnspan = 1)

        self.V = None
        self.ae = None
        self.optimizer = None

    def init_autoencoder_callback(self):
        filter_sizes = [int(s) for s in self.filter_sizes_string.get().split(',')]
        n_filters = [int(s) for s in self.n_filters_string.get().split(',')]
        if self.V is not None:
            n_filters[0] = self.V.shape[2]
            self.ae = ae.autoencoder(self.V.shape[0], self.V.shape[1], n_filters, filter_sizes)
        else:
            tk.messagebox.showinfo("Warning", "Load Traning data (NPY Volume) First!")
            return

    def load_input_callback(self):
        fname_in = filedialog.askopenfilename(title="Load NUMPY Volume", filetypes=(("npy files", "*.npy"),))
        self.V = np.load(fname_in)
        self.load_input_text.delete(1.0, tk.END)
        self.load_input_text.insert(tk.END, 'Volume ' + str(self.V.shape))

    def train_autoencoder_callback(self):
        if self.V is None or self.ae is None:
            tk.messagebox.showinfo("Warning", "Load Traning data (NPY Volume) (and/or) initialize autoencoder")
            return
        learning_date = float(self.learning_rate_string.get())
        n_epochs = int(self.n_epochs_entry.get())
        lambda_l1 = float(self.lambda_l1_string.get())
        band = int(self.band_string.get())

        assert learning_date > 0
        assert n_epochs> 0
        assert lambda_l1 >= 0
        assert band > 0

        n_frames = self.V.shape[-1]
        # changing to a propper tf format
        training_data = ae.normalize(self.V)
        training_data = training_data.transpose((3, 0, 1, 2)).reshape(n_frames, -1)

        self.optimizer, self.session = ae.init_training(self.ae, learning_date)

        for epoch in range(n_epochs):
            y, z, cost = ae.training_epoch(self.ae,self.optimizer, self.session,training_data,lambda_l1)
            print('epoch {0}/{1} cost {2}'.format(epoch,n_epochs, cost))

            y, z, cost = ae.training_epoch(self.ae, self.optimizer, self.session, training_data, lambda_l1)

            z_mean = np.mean(np.abs(z.reshape(z.shape[0], -1)), 1)
            z_mean_f = ae.fft_low_pass_filter(z_mean, band)

            peaks = peakutils.indexes(z_mean_f, 0)

            plot_code = self.latent_code_figure.add_subplot(211)
            plot_code.cla()
            plot_code.plot(z_mean_f,'b')
            plot_code.plot(peaks, z_mean_f[peaks],'ro')

            plot_cost = self.latent_code_figure.add_subplot(212)
            # plot_cost.cla()
            plot_cost.plot(epoch, cost, 'rx')

            self.latent_code_canvas.draw()