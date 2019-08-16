import peakutils
import numpy as np
import tkinter as tk
import autoencoder as ae
import utils

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import filedialog


class autoencoderGUI:

    def __init__(self, window):
        self.window = window

        self.loadInputButton = tk.Button(window, text="Load NPY Input", command=self.loadInputCallback)
        self.loadInputButton.grid(row=0, column=0)
        self.loadInputText = tk.Text(window, height=1, width=20)
        self.loadInputText.grid(row=0, column=1)

        self.filterSizesText = tk.Text(window, height=1, width=20)
        self.filterSizesText.grid(row=1, column=0)
        self.filterSizesText.configure(state='normal')
        self.filterSizesText.insert(tk.END, 'Filter Sizes:')
        self.filterSizesText.configure(state='disabled')

        self.filterSizesString = tk.StringVar()
        self.filterSizesString.set('3, 3, 3, 3')
        self.filter_sizes_entry = tk.Entry(window, textvariable=self.filterSizesString)
        self.filter_sizes_entry.grid(row=1, column=1)

        self.nFiltersText = tk.Text(window, height=1, width=20)
        self.nFiltersText.grid(row=2, column=0)
        self.nFiltersText.configure(state='normal')
        self.nFiltersText.insert(tk.END, 'Filter Sizes:')
        self.nFiltersText.configure(state='disabled')
        self.nFiltersString = tk.StringVar()
        self.nFiltersString.set('3, 10, 10, 10')
        self.n_filters_entry = tk.Entry(window, textvariable=self.nFiltersString)
        self.n_filters_entry.grid(row=2, column=1)

        self.lambdaL1Text = tk.Text(window, height=1, width=20)
        self.lambdaL1Text.grid(row=3, column=0)
        self.lambdaL1Text.configure(state='normal')
        self.lambdaL1Text.insert(tk.END, 'L1 Reguralization on Latent Code:')
        self.lambdaL1Text.configure(state='disabled')
        self.lambdaL1String = tk.StringVar()
        self.lambdaL1String.set('100.0')
        self.lambda_l1_entry = tk.Entry(window, textvariable=self.lambdaL1String)
        self.lambda_l1_entry.grid(row=3, column=1)

        self.initAutoencoderButton = tk.Button(window, text="Initialize Network",
                                               command=self.initAutoencoderCallback)
        self.initAutoencoderButton.grid(row=4, column=0, columnspan=2)

        self.learningRateText = tk.Text(window, height=1, width=20)
        self.learningRateText.grid(row=5, column=0)
        self.learningRateText.configure(state='normal')
        self.learningRateText.insert(tk.END, 'Learning Rate:')
        self.learningRateText.configure(state='disabled')
        self.learningRateString = tk.StringVar()
        self.learningRateString.set('0.001')
        self.learningRateEntry = tk.Entry(window, textvariable=self.learningRateString)
        self.learningRateEntry.grid(row=5, column=1)

        self.nEpochsText = tk.Text(window, height=1, width=20)
        self.nEpochsText.grid(row=6, column=0)
        self.nEpochsText.configure(state='normal')
        self.nEpochsText.insert(tk.END, 'Traning Epochs:')
        self.nEpochsText.configure(state='disabled')
        self.nEpochsString = tk.StringVar()
        self.nEpochsString.set('20')
        self.nEpochsEntry = tk.Entry(window, textvariable=self.nEpochsString)
        self.nEpochsEntry.grid(row=6, column=1)

        self.bandText = tk.Text(window, height=1, width=20)
        self.bandText.grid(row=7, column=0)
        self.bandText.configure(state='normal')
        self.bandText.insert(tk.END, 'Band:')
        self.bandText.configure(state='disabled')
        self.bandString = tk.StringVar()
        self.bandString.set('25')
        self.bandEntry = tk.Entry(window, textvariable=self.bandString)
        self.bandEntry.grid(row=7, column=1)

        self.trainAutoencoderButton = tk.Button(window, text="Train Network", command=self.trainAutoencoderCallback)
        self.trainAutoencoderButton.grid(row=8, column=0, columnspan=1)

        self.saveResultsButton = tk.Button(window, text="Save Results", command=self.saveResultsCallback)
        self.saveResultsButton.grid(row=8, column=1, columnspan=1)

        self.latentCodeFigure = Figure()  # figsize=(6, 6)
        self.latentCodeCanvas = FigureCanvasTkAgg(self.latentCodeFigure, master=window)
        self.latentCodeCanvas.get_tk_widget().grid(row=0, column=2, rowspan=8, columnspan=1)

        self.V = None
        self.ae = None
        self.optimizer = None
        self.latentCode = None
        self.latentCodeMeanFiltered = None
        self.session = None
        self.peaks = None

    def initAutoencoderCallback(self):
        filterSizes = [int(s) for s in self.filterSizesString.get().split(',')]
        nFilters = [int(s) for s in self.nFiltersString.get().split(',')]
        if self.V is not None:
            nFilters[0] = self.V.shape[2]
            self.ae = ae.autoencoder(self.V.shape[0], self.V.shape[1], nFilters, filterSizes)
        else:
            tk.messagebox.showinfo("Warning", "Load Traning data (NPY Volume) First!")
            return

    def loadInputCallback(self):
        filenameIn = filedialog.askopenfilename(title="Load NUMPY Volume", filetypes=(("npy files", "*.npy"),))
        self.V = np.load(filenameIn)
        self.loadInputText.delete(1.0, tk.END)
        self.loadInputText.insert(tk.END, 'Volume ' + str(self.V.shape))

    def saveResultsCallback(self):
        if self.V is None or self.ae is None:
            tk.messagebox.showinfo("Warning", "Train Autoencoder")
            return
        filenameOutVideo = tk.filedialog.asksaveasfilename(title='output video file ')
        filenameOutCSV = tk.filedialog.asksaveasfilename(title='output video file ')

        if filenameOutVideo is not "" and \
                self.V is not None and \
                self.peaks is not None:

            Vout = self.V.copy()
            Vout[0:5, 0:5, 0, self.peaks] = Vout.max()
            Vout[0:5, 0:5, 1:3, self.peaks] = Vout.max()
            Vout = Vout.astype(np.uint8)

            utils.writeVolume2Video(Vout,filenameOutVideo)
        if filenameOutCSV is not "" and self.peaks is not None:
            # np.savetxt(filenameOutCSV, self.peaks.astype(np.uint32), delimiter=",")
            self.peaks.tofile(filenameOutCSV, sep=',', format='%i')

    def trainAutoencoderCallback(self):
        if self.V is None or self.ae is None:
            tk.messagebox.showinfo("Warning", "Load Training data (NPY Volume) (and/or) initialize autoencoder")
            return
        learningRate = float(self.learningRateString.get())
        nEpochs = int(self.nEpochsEntry.get())
        lambdaL1 = float(self.lambdaL1String.get())
        band = int(self.bandString.get())

        assert learningRate > 0
        assert nEpochs > 0
        assert lambdaL1 >= 0
        assert band > 0

        nFrames = self.V.shape[-1]
        training_data = utils.normalize(self.V)
        training_data = training_data.transpose((3, 0, 1, 2)).reshape(nFrames, -1)

        self.optimizer, self.session = ae.initTraining(self.ae, learningRate)

        for epoch in range(nEpochs):
            y, z, cost = ae.trainingEpoch(self.ae, self.optimizer, self.session, training_data, lambdaL1)
            print('epoch {0}/{1} cost {2}'.format(epoch, nEpochs, cost))

            y, z, cost = ae.trainingEpoch(self.ae, self.optimizer, self.session, training_data, lambdaL1)

            self.latentCode = np.mean(np.abs(z.reshape(z.shape[0], -1)), 1)
            self.latentCodeMeanFiltered = ae.fftLowPassFilter(self.latentCode, band)
            self.peaks = peakutils.indexes(self.latentCodeMeanFiltered, 0)

            plotLatentCode = self.latentCodeFigure.add_subplot(211)
            plotLatentCode.cla()
            plotLatentCode.plot(self.latentCodeMeanFiltered, 'b')
            plotLatentCode.plot(self.peaks, self.latentCodeMeanFiltered[self.peaks], 'ro')

            plot_cost = self.latentCodeFigure.add_subplot(212)
            plot_cost.plot(epoch, cost, 'rx')

            self.latentCodeCanvas.draw()
