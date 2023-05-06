import threading
import sys
import os
import argparse
import tkinter as tk
import cv2
from PIL import Image, ImageTk
from enum import Enum

import video_capturers
import video_preprocessors

import time

# This is a definition of the protobuf config for the camera.
from google.protobuf.json_format import Parse
from google.protobuf.json_format import MessageToJson
import camera_config_pb2 as camera_config

# sudo apt install python3-tk
# sudo apt install python3-pil python3-pil.imagetk
# sudo apt  install protobuf-compiler

# pip install pyrealsense2
# pip install protobuf

# Run the program:
# python3 ./main.py --config_path=./camera_config.realsense.pbtxt --video_output_path=./out.mp4

config = camera_config.CameraConfig
config_mutex = threading.Lock()    


Action = Enum('Action', ['set_bottom_right', 'set_bottom_left', 'set_top_right', 'set_top_left', 'none'])


class VideoConfiguratorCapturerApp:
    def __init__(self, window, config_path, video_output_path):
        """Initializes the Tkinter application for configuring and capturing the video.
        
        Args:
            window: the tk.Tk() window instance
            config_path: File where the config file will be read from and where it will be saved, if required.
            video_output_path: File where the captured (and, if specified by a config - transformed) video will be saved into.
        """
        self.window = window
        self.window.title("SmartPool configurator and capturer")
        
        # Timestamp for benchmarking the GUI application.
        self.previous_processing_end = None

        self.config_path = config_path
        self.video_output_path = video_output_path

        # Get the specified device type from a config, try to open that device to read frames.
        try:
            self.capturer= video_capturers.CreateVideoCapturer(config.camera_config)
        except Exception:
            sys.exit(1)

        # Set up a video preprocessor
        self.video_preprocessor = video_preprocessors.CapturePreprocessingPipeline(
            self.capturer, config.preprocess_config)

        # Create a canvas where the video will be displayed
        self.canvas = tk.Canvas(window, width=self.capturer.width, height=self.capturer.height)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.onLeftButton)
        self.canvas.bind("<Button-3>", self.onRightButton)
        
        # Create a frame on the right with all the controls
        self.controls_frame = tk.Frame(self.window)
        self.controls_frame.pack(side="right", fill="y", padx=10, pady=10)

        # Create buttons for setting boundaries of the table on the image 
        self.btn_set_boundaries = tk.Button(
            self.controls_frame, text="Set transform boundaries", command=self.set_boundaries)
        self.btn_set_boundaries.grid(row=0, column=0, pady=10)
        self.btn_set_boundaries.config(width=20)

        self.btn_clear_boundaries = tk.Button(
            self.controls_frame, text="Clear transform boundaries", command=self.clear_boundaries)
        self.btn_clear_boundaries.grid(row=1, column=0, pady=10)
        self.btn_clear_boundaries.config(width=20)

        self.enable_transform_checkbox = tk.BooleanVar(value=config.preprocess_config.do_translation)
        self.enable_transform_widget = tk.Checkbutton(
            self.controls_frame, text="Enable transform",
            variable=self.enable_transform_checkbox, command=self.onEnableTranslation)
        self.enable_transform_widget.grid(row=2, column=0, pady=10)
        self.enable_transform_widget.config(width=20)

        self.btn_start_capture = tk.Button(
            self.controls_frame, text="Start video capture", command=self.start_capture)
        self.btn_start_capture.grid(row=3, column=0, pady=10)
        self.btn_start_capture.config(width=20)
        if video_output_path is None or not config.preprocess_config.do_translation:
            self.btn_start_capture.config(state="disabled")

        self.btn_stop_capture = tk.Button(
            self.controls_frame, text="Stop video capture", command=self.stop_capture)
        self.btn_stop_capture.grid(row=4, column=0, pady=10)
        self.btn_stop_capture.config(width=20)
        self.btn_stop_capture.config(state="disabled")

        # Create a button for saving the config into Jsonfile
        self.btn_save_config = tk.Button(
            self.controls_frame, text="Save config", command=self.save_config)
        self.btn_save_config.grid(row=5, column=0, pady=10)
        self.btn_save_config.config(width=20)

        # Clear action
        self.action = Action.none

        # create a text label at the bottom of the window
        self.status_label = tk.Label(self.controls_frame, text="Ready.", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.grid(row=6, column=0, pady=10)

        # Set up a desired GUI refresh rate to be 2 times the camera FPS        
        self.delay = int(1000 / (self.capturer.fps * 2)) 
        self.update()
        
        # Now start the thread that does video capturing and processing.
        self.processingThread = threading.Thread(target=self.processingThreadFn)
        self.processingThread.start()
        # Register the atexit function, so thread will be stopped when the main class exits.
        self.exitEvent = threading.Event() 
        root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # An event used to enable / disable video recording
        self.recordEvent = threading.Event()

        # Set up a background subtractor to detect motion.
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.background_subtractor.setShadowValue(0)

        self.window.mainloop()


    def hasMotion(self, frame):
        """Tries to detect motion in an input frame. 

        Returns:
          True, if there was noticeable motion in a frame.
        """
        if frame is None:
            return False, None
        # Don't add a gaussian blur here. It only increases false alarms due to 
        # shadows
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgMask = self.background_subtractor.apply(gray)
        
        # Apply a threshold to the difference image
        thresh = cv2.threshold(fgMask, 50, 255, cv2.THRESH_BINARY)[1]
    
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.erode(thresh, kernel, iterations=1)
        # thresh = cv2.dilate(thresh, kernel, iterations=1)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # mask_frame may contain a visualization of a motion detector.
        # mask_frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) < 70:
                continue
            motion_detected = True
            # Uncomment for visualization
            # (x, y, w, h) = cv2.boundingRect(contour)
            # cv2.rectangle(mask_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return motion_detected


    def processingThreadFn(self):
        """This is a thread that reads frames from the camera and preprocess them.
        It must be started at the beginning of the program, so all the pre-processing will run in background"""
        desired_size = (
            config.preprocess_config.translate_resolution.x,
            config.preprocess_config.translate_resolution.y
        )
        video_out = None
        if self.video_output_path is not None:
            # fourcc = cv2.VideoWriter_fourcc(*'avc1')
#             fourcc = cv2.VideoWriter_fourcc('H','F','Y','U')
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            video_out = cv2.VideoWriter(
                self.video_output_path,
                fourcc, config.camera_config.fps,
                desired_size, True)

        while True:
            self.video_preprocessor.Process()
            frame = self.video_preprocessor.GetProcessedFrame()

            if self.exitEvent.is_set():
                break
            if self.recordEvent.is_set():
                if config.detect_motion:
                    if self.hasMotion(frame):
                        video_out.write(frame)
                else:
                    video_out.write(frame)

        if video_out is not None:
            print("Processing thread: closing video writer...")
            video_out.release()

        print("Processing thread exiting normally...")

    def on_closing(self):
        """This function is called when the user attempts to close the main app window.""" 
        print("Stopping the preprocessor thread...")
        self.exitEvent.set()
        self.processingThread.join()
        print("The preprocessor exited.")
        # Don't forget to call destroy, or the app window will hang around.
        self.window.destroy()


    def onLeftButton(self, event):
        print("LMouse clicked at", event.x, event.y)
        if self.action == Action.none:
            return

        if self.action == Action.set_bottom_right:
            with config_mutex:
                config.preprocess_config.source_region.lower_right.x = event.x
                config.preprocess_config.source_region.lower_right.y = event.y
                self.action = Action.set_bottom_left
                self.status_label['text'] = 'Click bottom left of the table'
            return

        if self.action == Action.set_bottom_left:
            with config_mutex:
                config.preprocess_config.source_region.lower_left.x = event.x
                config.preprocess_config.source_region.lower_left.y = event.y
                self.action = Action.set_top_left
                self.status_label['text'] = 'Click top left of the table'
            return

        if self.action == Action.set_top_left:
            with config_mutex:
                config.preprocess_config.source_region.upper_left.x = event.x
                config.preprocess_config.source_region.upper_left.y = event.y
                self.action = Action.set_top_right
                self.status_label['text'] = 'Click top right of the table'
            return

        if self.action == Action.set_top_right:
            with config_mutex:
                config.preprocess_config.source_region.upper_right.x = event.x
                config.preprocess_config.source_region.upper_right.y = event.y
                self.action = Action.none
                self.status_label['text'] = ''
                self.video_preprocessor.UpdateConfig(config.preprocess_config)
            return
        
    
    def onRightButton(self, event):
        print("RMouse clicked at", event.x, event.y)

    def onEnableTranslation(self):
        """ Called when the user changes the checkbox for enabling/ disabling image region transformation.
            Here "transformation" is when we cut a region of the image corresponding to a pool table,
            and operating with only that region. """ 
        with config_mutex:
            config.preprocess_config.do_translation = self.enable_transform_checkbox.get()
        # If we are switching on the translation, reset the canvas (where the translation region is defined).
        # Otherwise the region boundaries will be still drawn on the canvas. 
        self.canvas.delete("all")
        self.video_preprocessor.UpdateConfig(config.preprocess_config)
        if self.video_output_path is not None and config.preprocess_config.do_translation:
            self.btn_start_capture.config(state="normal")
        else:
            self.btn_start_capture.config(state="disabled")

    def snapshot(self):
        frame = self.video_preprocessor.GetProcessedFrame()
        if frame is not None:
            cv2.imwrite("snapshot.png", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def set_boundaries(self):
        self.action = Action.set_bottom_right
        self.status_label['text'] = 'Click bottom right of the table'

    def clear_boundaries(self):
        with config_mutex:
            config.preprocess_config.source_region.Clear()
    
    def save_config(self):
        config_json = MessageToJson(config) 
        with open(self.config_path, "wt") as f:
            f.write(config_json)

    def start_capture(self):
        self.btn_start_capture.config(state="disabled")
        self.btn_stop_capture.config(state="normal")
        self.enable_transform_widget.config(state="disabled")
        self.recordEvent.set()
        pass

    def stop_capture(self):
        self.btn_start_capture.config(state="normal")
        self.btn_stop_capture.config(state="disabled")
        self.enable_transform_widget.config(state="normal")
        self.recordEvent.clear()
        pass

    def update(self):
        # self.video_preprocessor.Process()
        # if self.enable_transform_checkbox.get():
        #     current_frame = self.video_preprocessor.GetProcessedFrame()
        # else:
        #     current_frame = self.video_preprocessor.GetCameraFrame()
        current_bgr_frame = self.video_preprocessor.GetProcessedFrame()

        if current_bgr_frame is not None:
            current_frame = cv2.cvtColor(current_bgr_frame, cv2.COLOR_BGR2RGB)
            self.frame = ImageTk.PhotoImage(image=Image.fromarray(current_frame))
            self.canvas.create_image(0, 0, image=self.frame, anchor=tk.NW)

            with config_mutex:
                preprocess_config = config.preprocess_config
                if not preprocess_config.do_translation:
                    if preprocess_config.source_region.lower_right.x != 0 and preprocess_config.source_region.lower_left.x != 0: 
                        self.canvas.create_line(
                            preprocess_config.source_region.lower_right.x,
                            preprocess_config.source_region.lower_right.y,
                            preprocess_config.source_region.lower_left.x,
                            preprocess_config.source_region.lower_left.y,
                            width=2,
                            fill = 'green')
                    if preprocess_config.source_region.lower_left.x != 0 and preprocess_config.source_region.upper_left.x != 0: 
                        self.canvas.create_line(
                            preprocess_config.source_region.lower_left.x,
                            preprocess_config.source_region.lower_left.y,
                            preprocess_config.source_region.upper_left.x,
                            preprocess_config.source_region.upper_left.y,
                            width=2,
                            fill = 'green')
                    if preprocess_config.source_region.upper_left.x != 0 and preprocess_config.source_region.upper_right.x != 0: 
                        self.canvas.create_line(
                            preprocess_config.source_region.upper_left.x,
                            preprocess_config.source_region.upper_left.y,
                            preprocess_config.source_region.upper_right.x,
                            preprocess_config.source_region.upper_right.y,
                            width=2,
                            fill = 'green')
                    if preprocess_config.source_region.upper_right.x != 0 and preprocess_config.source_region.lower_right.x != 0: 
                        self.canvas.create_line(
                            preprocess_config.source_region.upper_right.x,
                            preprocess_config.source_region.upper_right.y,
                            preprocess_config.source_region.lower_right.x,
                            preprocess_config.source_region.lower_right.y,
                            width=2,
                            fill = 'green')

        self.current_processing_end = time.time()
        processing_per_second = 0
        if self.previous_processing_end is not None:
            dt_previous = (self.current_processing_end - self.previous_processing_end)
            processing_per_second =  1 / dt_previous
        self.previous_processing_end = self.current_processing_end
            
        # print("Processing time, s: %f, processing FPS: %f" % (self.video_preprocessor.GetProcessingTime(), processing_per_second))
        
        self.window.after(self.delay, self.update)

# Expect we will have a filename passed as a command line argument
parser = argparse.ArgumentParser()
parser.add_argument(
    '--config_path',
    help='File where the config file will be read from and where it will be saved, if required.')
parser.add_argument(
    '--video_output_path',
    help='File where the captured (and, if specified by a config - transformed) video will be saved into.')
args = parser.parse_args()

if args.config_path is None:
    print("Config file name should be passed to an application")
    sys.exit(1)

if os.path.exists(args.config_path):
    with open(args.config_path, "rt") as f:
        config_proto_text = f.read()
    config = Parse(config_proto_text, camera_config.CapturerConfig())
    print(config)

# Now launch the UI application
root = tk.Tk()
VideoConfiguratorCapturerApp(root, args.config_path, args.video_output_path)

