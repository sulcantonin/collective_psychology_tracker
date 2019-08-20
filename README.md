The tool you are going to operate with is designed to help you with processing, tracking and analyzing repetitive patterns of videos. It consist of three main parts, where one part is designed to help you with pre-process the video and select only the relevant part (to minimize the computational overhead), the second part is deisgned to help you track your object(s), where we designed a minimalistic interface where you can select your object, move forward and backward in the video and eventaully correct tracks when it is necessary and lastly detect the repetitive patterns by a fully automated statistical framework based on state-of-the-art encoder-decoder minimalistic architecture. 

# Theory

## Region of Interest (ROI) selection
The first part of the pipeline is here to help you with cutting the video into the relevant part, 
because proceessing high-resolution videos is very slow (it is a tradeoff for ability to go backwards, 
it may change in future releases of the opencv, but the resolution of the input videos play a drastic 
role in the playback speed).

You can rely on the automatic ROI selection (*automatic_roi_selection = 1* in config.ini), or do it by hand (*automatic_roi_selection = 0*). 

### Autoamtic ROI Selection
The automatic ROI selection goes though all frames and cumulates a squared difference of consecutive frames *frameDiff* and selects the largest bounding box around around the region which is found as relevant.

The pixels which belong the the relevant region are those whose value is *abs(frameDiff - mean(frameDiff)) > sigma * std(frameDiff)* (a six sigma test, the sigma is the *automatic_roi_selection_sigma_mult* parameter in config.ini).

Since the mask is pixelwise, it can easily contain holes, thus we added a morpohology filter on the mask to fill the missing holes of the detected region. The *automatic_roi_morph_disk_radius* defines the radius of the disk used for erosion and dilation.

The file is automatically saved with the same name as the input video with the suffix *_roi*. 

### Manual ROI Selection
The manual ROI selection caculates the *frameDiff* likewise the automatic ROI selection, but shows the frameDiff to you so you can have an overview which parts of the scene are changing and thus are relevant.

# Installation
The library runs on Python 3.6. 

Install following packages, the easiest way is to simply run pip3 install $library, where the necssary libraries are:

- numpy 
- scipy
- matplotlib
- setuptools (required by tensorflow)
- wheel  (required by tensorflow)
- tensorflow
- opencv-python
- opencv-contrib-python
- PeakUtils

Unfortunately, installation of all these libraries is aboslute necessary to have access to all features of the tool.

# Running the App #
Once you have made all the steps necessary to run the program, you can run the program by simply running the main.py script as a python script:

python3 main.py

# GUI #
Once you have the program running, you can have following options
