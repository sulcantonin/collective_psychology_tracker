The tool you are going to operate with is designed to help you with processing, tracking and analyzing repetitive patterns of videos. It consists of three main parts, where one part is designed to help you with pre-process the video and select only the relevant part (to minimize the computational overhead), the second part is designed to help you track your object(s), where we designed a minimalistic interface where you can select your object, move forward and backward in the video and eventually correct tracks when it is necessary and lastly detect the repetitive patterns by a fully automated statistical framework based on state-of-the-art encoder-decoder minimalistic architecture. 

**All crashes, bugs, and suggestions should be reported to Antonin Sulc**

# Theory

## Region of Interest (ROI) selection
The first part of the pipeline is here to help you with cutting the video into the relevant part, 
because processing high-resolution videos are very slow (it is a tradeoff for the ability to go backward, 
it may change in future releases of the OpenCV, but the resolution of the input videos play a drastic 
role in the playback speed).

You can rely on the automatic ROI selection (*automatic_roi_selection = 1* in config.ini), or do it by hand (*automatic_roi_selection = 0*). 

### Automatic ROI Selection
The automatic ROI selection goes though all frames and cumulates a squared difference of consecutive frames *frameDiff* and selects the largest bounding box around the region which is found as relevant.

The pixels which belong the the relevant region are those whose value is *abs(frameDiff - mean(frameDiff)) > sigma * std(frameDiff)* (a six sigma test, the sigma is the *automatic_roi_selection_sigma_mult* parameter in config.ini).

Since the mask is pixel-wise, it can easily contain holes, thus we added a morphology filter on the mask to fill the missing holes of the detected region. The *automatic_roi_morph_disk_radius* defines the radius of the disk used for erosion and dilation.

The file is automatically saved with the same name as the input video with the suffix *_roi*. 

![Difference of Consecutive Frames](https://github.com/sulcantonin/collective_psychology_tracker/blob/master/materials/images/roiFrameDiff.png "Difference of Consecutive Frames")

### Manual ROI Selection
The manual ROI selection calculates the *frameDiff* likewise the automatic ROI selection, but shows the frameDiff to you so you can have an overview which parts of the scene are changing and thus are relevant.

You confirm your selection of the bounding box by pressing **Enter**

## Tracking
The tracking GUI provides an interface to the state-of-the-art tracking model-based trackers available in OpenCV, which means that all available trackers need at least one sample determined by the bounding box given by the user. 

The trackers are the following:

* **csrt**, based on Alan Lukezic, Tomas Vojir, Luka Cehovin Zajc, Jiri Matas, and Matej Kristan. Discriminative correlation filter tracker with channel and spatial reliability. International Journal of Computer Vision, 2018.
* **kcf**, bassed on J. F. Henriques, R. Caseiro, P. Martins, and J. Batista. Exploiting the circulant structure of tracking-by-detection with kernels. In proceedings of the European Conference on Computer Vision, 2012 and M. Danelljan, F.S. Khan, M. Felsberg, and J. van de Weijer. Adaptive color attributes for real-time visual tracking. In Computer Vision and Pattern Recognition (CVPR), 2014 IEEE Conference on, pages 1090–1097, June 2014.
* **boosting**, based on Helmut Grabner, Michael Grabner, and Horst Bischof. Real-time tracking via on-line boosting. In BMVC, volume 1, page 6, 2006.
* **mil**, based on Boris Babenko, Ming-Hsuan Yang, and Serge Belongie. Visual tracking with online multiple instance learning. In Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on, pages 983–990. IEEE, 2009. 
* **tld**, based on Zdenek Kalal, Krystian Mikolajczyk, and Jiri Matas. Tracking-learning-detection. Pattern Analysis and Machine Intelligence, IEEE Transactions on, 34(7):1409–1422, 2012.
* **medianflow**, based on Zdenek Kalal, Krystian Mikolajczyk, and Jiri Matas. Forward-backward error: Automatic detection of tracking failures. In Pattern Recognition (ICPR), 2010 20th International Conference on, pages 2756–2759. IEEE, 2010.
* **mosse**, based on David S. Bolme, J. Ross Beveridge, Bruce A. Draper, and Man Lui Yui. Visual object tracking using adaptive correlation filters. In Conference on Computer Vision and Pattern Recognition (CVPR), 2010.

The type of tracker used for object tracking object(s) can be chosen in config.ini, field *tracker*.

You can choose up to 10 objects to be tracked. 

When the GUI pops-up on the initial frame, you can choose the initial bounding boxes for the objects you want to track. By using numbers (either above regular alphabet or on Numpad), you can switch between different objects and by dragging a mouse you can set a bounding box to determine the area which specifies the object. When you are done, press **Q** to quit the selection interface. Note that you do not have to specify the bounding box for all/any objects if they do not appear in the first frame.

To move to next frame, press **D**, to move to backward, press **A**, to switch to the bounding box selection GUI (in case you won't change the current tracked bounding box - *the tracker will be restarted *) press **S**). If you want to run automatic playback press **P** (and to stop it), **Q** to quit.

![Selecting of Objects to Track](https://github.com/sulcantonin/collective_psychology_tracker/blob/master/materials/images/tracking.png "Selecting of Objects to Track")

After quitting (pressing the **Q**) the program automatically creates for each tracklet a file which is called *$filename_$trackletid*, where $filename is input video filename (without suffix) and $trackletid a number <0,9> which corresponds to an ID of the tracked object. The files for each tracklet are:
* a CSV file with a bounding box [left-top-x,left-top-y,right-bottom-x,right-bottom-y], see image below
* a npy volume (can be huge) which is cut-out of the bounding box resized to the mean size of the bounding boxes during the tracking (can be disabled by tracker_npyvolume_output = 0 in config.ini)

![ROI](https://github.com/sulcantonin/collective_psychology_tracker/blob/master/materials/images/selectionbbox.png "ROI")

Additionally, for validation, the program creates a video with detected bounding boxes (can be turned off via tracker_video_output = 0)

## Extracting the repetitive patterns from the video
This part uses the autoencoder architecture. There are plenty of tutorials introducing autoencoders, for example, https://www.jeremyjordan.me/autoencoders/

![Autoencoder](https://github.com/sulcantonin/collective_psychology_tracker/blob/master/materials/images/cnn_ae.png "Autoencoder")

An intuitive explanation of an autoencoder is it is a black box (*convolutional neural net*), which has a bottleneck in the form a latent code which has carries much less information than the input (*image*). During the training, you are giving the images to the input and expect that you will get the same image on the input, but the image was squeezed into the minimal representation of the bottleneck (*latent code*). The minor changes in the image are largely visible in a change of the latent code and by the study of this code, we can investigate anomalies of the examined object or find repetitive patterns. As a result, when the network is trained, each frame can be compressed to its latent code. 

The GUI provides access to a very simple configuration of the network, i.e. number of convolutional layers, size of the convolutions, number of feature maps, etc. 

![Convolutions](https://github.com/sulcantonin/collective_psychology_tracker/blob/master/materials/images/conv.jpg "Convolution")

The GUI looks like it is shown in the image below:

![Autoencoder GUI](https://github.com/sulcantonin/collective_psychology_tracker/blob/master/materials/images/ae_gui.png "Autoencoder GUI")

The latent code is often highly dimensional, but we would like to have a feeling how does it look like in a simple 1D plot. We calculate mean of the absolute value of each frame which turns out to be sufficient to preserve some information (that is what you see on the right top plot, a horizontal axis is a frame number, vertical is a corresponding latent code)

You can load the volume which can be either your input or an arbitrary four-dimensional volume. The volume produced by tracking has the following format (row-major order as numpy): height, width, channels, frames. 

The two graphs on the right show the mean value of the absolute value of the latent code (top) and current L2 loss of training (bottom). **If the video in the volume contains some repetitive visual content, the peaks in the latent code should be present for the frames when the event happens**. 

The fields:
* *Volume Size* is current size of the input volume 
* *Filter Sizes* is the size of the convolutional filter (the smallest is 3, I recommend you to keep it like this). For instance, the convolution in the figure above is 3x3 convolution
* *Filter Sizes* is a number of features. It says how many separate convolutions we want to have on each level. The number of elements should be identical as *Filter Sizes*

The default configuration works relatively well, adding more complexity usually causes trouble and it is often better to just remove one layer rather than add.

After these parameters are set the network can be initialized by the button *Initialize Network*

The fields:
* *Learning Rate* say how big the steps to train network should be. The rule of thumb is that when no changes in the latent code and loss during training are noticed, it is necessary to increase the learning rate. When the latent code changes drastically and the loss does not get lower it is necessary to decrease the learning rate. 
* *Training Epochs* say how many iterations we should perform. Since we do not have enough data it is sufficient to perform only very few interactions (like 20). When the loss does not change, we performed unnecessary many iterations. 
* *Band* is says which frequencies are filtered. The latent codes are usually very bumpy and higher frequencies can be filtered. The lower the value gets the narrowed the passband gets which means that higher frequencies are filtered and the latent code get intuitively smoother, see the figure below.

![Bandwidth](https://github.com/sulcantonin/collective_psychology_tracker/blob/master/materials/images/band.png "Bandwidth")

Once you have everything set, press Train Network. In case nothing happens for a few seconds, it probably means that the input volume is too large. If the loss (graph in the right bottom) does not go down, follow the suggestions about *Learning Rate* above and once you are done, you can save the results:

* peaks in the CSV files. Peaks are the red dots on the latent code they should correspond to the repetitive pattern
* video with visualization of the peaks in the top left corner. It reproduces the input volume and creates a video where the top left corner has a small black box when the peak (the red dot) was detected.

# Installation
The library runs on Python 3.6. 

You can install the necessary packages with pip3. There is a setup.py script with a list of all necessary packages and everything should be done automatically by pip3 by the following command:

pip3 install .

Install the following packages, the easiest way is to simply run pip3 install $library, where the necessary libraries are:

- numpy 
- scipy
- matplotlib
- setuptools (required by tensorflow)
- wheel  (required by tensorflow)
- tensorflow
- opencv-python
- opencv-contrib-python
- PeakUtils
- skimage


Unfortunately, the last library *TKInter* seems to be separated from pip and need an explicit installation 

# Running the App #
Once you have made all the steps necessary to run the program, you can run the program by simply running the main.py script as a python script:

python3 main.py
