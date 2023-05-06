import threading
import time

import cv2
import numpy as np

import camera_config_pb2 as camera_config


class CapturePreprocessingPipeline:
    """A class that does initial preprocessing of a video, captured from a camera.
    
    It is tread-safe. Preprocessing can be done in one thread, which captures
    video from the camera. And another thread can safely get the processed frames,
    and update the config on-the-fly. 
    """

    def __init__(self, video_capturer, preprocess_config):
        self.config = preprocess_config
        self.video_capturer = video_capturer
    
        # Buffers for storing raw camera frame and a pre-processed frame.    
        self.camera_frame = None
        self.processed_frame = None
        
        # Mutexes for accessing frame and configuration respectively.
        self.frame_mutex = threading.Lock()
        self.config_mutex = threading.Lock()
        
        # Contains processing time statistics.
        self.processing_time = 0    
        
    def isTranslationMatrixDefined(self):
        if (self.config.source_region.lower_left.x == 0 and
            self.config.source_region.lower_left.y == 0):
            return False
        return True

    def doProjectiveTransform(self, input_frame):
        """Applies the projective transformation on the frame from a camera.
        
           Effectively, "cuts" out a position specified by a source_region, 
           specified by the self.config.source_region, and transforms it into
           a destination region, which has resolution self.config.translate_resolution.
        """
        current_frame = None
        if self.config.do_translation and self.isTranslationMatrixDefined():
            input_pts = np.float32([
                [self.config.source_region.lower_left.x, self.config.source_region.lower_left.y],
                [self.config.source_region.upper_left.x, self.config.source_region.upper_left.y],
                [self.config.source_region.upper_right.x, self.config.source_region.upper_right.y],
                [self.config.source_region.lower_right.x, self.config.source_region.lower_right.y],
            ])
            # Note: here we change the order of translation points from an intuitive (0,0), (0,y),
            # (x,y), (x,0) - same as in our input points - because in the image we measure coordinates from the
            # upper left corner. We need the order of points to match translation coordinates.    
            output_pts = np.float32([
                [0, self.config.translate_resolution.y],
                [0, 0],
                [self.config.translate_resolution.x, 0],
                [self.config.translate_resolution.x, self.config.translate_resolution.y],
            ])
            M = cv2.getPerspectiveTransform(input_pts,output_pts)
            current_frame = cv2.warpPerspective(
                input_frame, M,
                (self.config.translate_resolution.x, self.config.translate_resolution.y))
        else:
            current_frame = input_frame
        return current_frame
        

    def UpdateConfig(self, new_preprocess_config):
        """Updates the preprocessor config in this class."""
        with self.config_mutex:
            self.config = camera_config.PreprocessConfig() 
            self.config.CopyFrom(new_preprocess_config)

    def Process(self):
        """Performs preprocessing of a single frame captured from a camera.
        
        Copies raw frame into self.camera_frame, and preprocessed frame into
        self.processed_frame. In this way this can safely run in multithreaded
        environment.
        """
        frame = self.video_capturer.GetFrame()
        processing_start = time.time()
        local_processed_frame = None
        if frame is not None:
            local_processed_frame = self.doProjectiveTransform(frame)
            with self.frame_mutex:
                self.camera_frame = frame.copy()
                self.processed_frame = local_processed_frame.copy()
        processing_end = time.time()
        self.processing_time = (processing_end - processing_start)
    
    def GetCameraFrame(self):
        return_frame = None
        with self.frame_mutex:
            if self.camera_frame is not None:
                return_frame = self.camera_frame.copy()
        return return_frame


    def GetProcessedFrame(self):
        return_frame = None
        with self.frame_mutex:
            if self.processed_frame is not None:
                return_frame = self.processed_frame.copy()
        return return_frame

    def GetProcessingTime(self):
        return self.processing_time
