import sys
import cv2
import pyrealsense2 as rs
import numpy as np
import traceback

import camera_config_pb2

class VideoCapture:
    """ A base class for capturing videos from various devices. """

    def __init__(self, camera_config):
        """Initialize the capturer from the CameraConfig proto."""
        self.width = camera_config.original_resolution.x
        self.height = camera_config.original_resolution.y
        self.fps = camera_config.fps
        self.exposure = camera_config.exposure

    def GetFrame(self):
        """Read the next frame from the device. """
        pass
    
    # def GetWidth(self):
    #     return self.width
    #
    # def GetHeight(self):
    #     return self.height
    #
    # def GetFPS(self):
    #     return self.fps


class VideoCaptureRealSense(VideoCapture):
    """A class that captures videos from Intel RealSense device. """

    def __init__(self, camera_config):
        super().__init__(camera_config)
        
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        print("Detected RealSense device: %s. Enabling color stream at (%d x %d), %d FPS." %
              (device_product_line, self.width, self.height, self.fps))
        config.enable_stream(rs.stream.color,
                             self.width, self.height, rs.format.rgb8, self.fps)
        
        print("Starting the camera pipeline and waiting for the frames...")
        self.pipeline.start(config)
        self.pipeline.wait_for_frames(5000)

        # Get a video sensor
        sensor = self.pipeline.get_active_profile().get_device().query_sensors()[1]
        print("Setting camera parameters.")
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        sensor.set_option(rs.option.exposure, camera_config.exposure)
        sensor.set_option(rs.option.gain, camera_config.gain)

        sensor.set_option(rs.option.brightness, camera_config.brightness)
        sensor.set_option(rs.option.contrast, camera_config.contrast)
        sensor.set_option(rs.option.gamma, camera_config.gamma)
        sensor.set_option(rs.option.hue, camera_config.hue)
        sensor.set_option(rs.option.saturation, camera_config.saturation)
        sensor.set_option(rs.option.sharpness, camera_config.sharpness)
        sensor.set_option(rs.option.white_balance, camera_config.white_balance)

    def __del__(self):
        print("Closing RealSense camera")
        self.pipeline.stop()

    def GetFrame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame:
            color_image = np.asanyarray(color_frame.get_data())
            return color_image
        print("Can't capture frame!")
        return None


class VideoCaptureUSB(VideoCapture):
    def __init__(self, camera_config):
        super().__init__(camera_config)
        self.cap = cv2.VideoCapture(camera_config.camera_source)
        if not self.cap.isOpened():
            raise ValueError("Unable to open USB video source", camera_config.video_source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
    
        print("Camera parameters: width=%d, height=%d, FPS=%d" % (self.width, self.height, self.fps))

    def __del__(self):
        print("Closing camera")
        if self.cap.isOpened():
            self.cap.release()
    
    def GetFrame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
        print("Can't capture frame!")
        return None    


def CreateVideoCapturer(camera_config):
    """Creates a video capturer of a type, specified by a given a configuration.
    Will throw an exception if there were any issues with the video source.
    """
    capturer = None
    try:
        if camera_config.camera_type == camera_config_pb2.CameraConfig.INTEL_REALSENSE:
            capturer = VideoCaptureRealSense(camera_config)
        else:
            capturer = VideoCaptureUSB(camera_config)
    except Exception as e:
        sys.stderr.write("Failed to open a camera of a specified type %s\n" % camera_config.camera_type)
        sys.stderr.write("Make sure that camera type you specified in the cameraType config field matches your device type\n")
        print("Exception: {}".format(e))
        print("Traceback: ")
        traceback.print_exc()
        raise
    return capturer
