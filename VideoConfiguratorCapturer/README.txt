This folder contains programs for capturing video, for labeling video manually, and for running inference
on a captured video.


# VideoConfiguratorCapturer

An actual VideoConfiguratorCapturer application is implemented in the main.py. This is a Tkinter GUI program, that allows to specify boundaries of a table, as well as capturing the video of the pool 
game and storing it into the file.
main.py depends on video_capturers.py and video_preprocessors.py to capture the input video,
and to apply a projective transform on it. All the configuration settings are specified through the
Google Protocol buffers, as specified in the camera_config.proto. Two .pbtxt files contain the config for
RealSense camera, and for the regular generic USB camera.

Currently, main.py is a pretty messy piece of code that should be re-written. Currently all the code - 
UI and some processing is mixed in the same file. It has even some parts of image processing (i.e. 
motion detection) implemented there. Also, the controller logic and UI is completely mixed together as well.
What needs to be done:
-  move all the image-related transforms (including motion detection) to another library module. It will
likely be reused by different tools.
- split UI and control logic, according to the MVC pattern.
- We will likely write similar apps (e.g. to visualize depth, for detecting balls, etc). So overall, think how we can re-use as much UI-related code as we can. 

# SimpleAnnotator

This is a very simple OpenCV script for manually labeling balls on the video. We use it to create a "ground
truth" dataset to validate the detector accuracy.

# ObjectDetector

This script is doing actual inference on an video. It loads a video of a game, loads the trained network,
and detects balls on it.
 

# Installation for VideoConfiguratorCapturer (probably, incomplete)

1. Install common dependencies.

Make sure Pip is installed:
	sudo apt install python3-pip


2. Install Realsense SDK 

2.1. Follow instructions at the official RealSense repository - how to install RealSense SDK: 
https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md

Alternative instructions are available at https://robots.uc3m.es/installation-guides/install-realsense2.html

2.2. Install pyrealsense2 Python package:
	pip install pyrealsense2 

	
3. Install tkinter package.
TkInter is a Python library for producing GUI using Tcl/Tk. Install it using:
	sudo apt install python3-tk python3-pil python3-pil.imagetk


