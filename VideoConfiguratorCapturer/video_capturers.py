#
# Set of classes that capture the video from a camera, and pass it to the rest of the program.
# The video can be captured from the regular USB camera, or from RealSense camera. For RealSense, we also capture the depth stream.
# It is expected these classes will be running in a separate thread, so we make them as threadsafe as we can.
#
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
        """Read the next frame from the device in the BGR format (aka OpenCV standard). """
        pass

class VideoCaptureRealSense(VideoCapture):
    """A class that captures videos from Intel RealSense device. """

    def __init__(self, camera_config):
        super().__init__(camera_config)
        
        self.camera_config = camera_config
        
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        print("Detected RealSense device: %s. Enabling color stream at (%d x %d), %d FPS." %
              (device_product_line, self.width, self.height, self.fps))

        print("Sensors found:")
        for sensor in device.sensors:
            print(sensor.get_info(rs.camera_info.name))
            if sensor.get_info(rs.camera_info.name) == 'RGB Camera':
                print("Setting camera parameters.")
                sensor.set_option(rs.option.enable_auto_exposure, 0)
                sensor.set_option(rs.option.exposure, self.camera_config.exposure)
                sensor.set_option(rs.option.gain, self.camera_config.gain)

                sensor.set_option(rs.option.brightness, self.camera_config.brightness)
                sensor.set_option(rs.option.contrast, self.camera_config.contrast)
                sensor.set_option(rs.option.gamma, self.camera_config.gamma)
                sensor.set_option(rs.option.hue, self.camera_config.hue)
                sensor.set_option(rs.option.saturation, self.camera_config.saturation)
                sensor.set_option(rs.option.sharpness, self.camera_config.sharpness)
                sensor.set_option(rs.option.white_balance, self.camera_config.white_balance)
            if sensor.get_info(rs.camera_info.name) == 'Stereo Module':
                if self.camera_config.HasField("depth_config"):
                    print("Setting depth module parameters.")
                    if self.camera_config.HasField("depth_config"):
                        if self.camera_config.depth_config.min_range > 0:
                            sensor.set_option(rs.option.min_distance, self.camera_config.depth_config.min_range)  # 10 cm
                        if self.camera_config.depth_config.max_range > 0:
                            sensor.set_option(rs.option.min_distance, self.camera_config.depth_config.max_range)  # 10 cm

        print("Enabling color stream.")
        config.enable_stream(rs.stream.color,
                             self.width, self.height, rs.format.bgr8, self.fps)
        
        if self.camera_config.HasField("depth_config"):
            print("Enabling depth stream.")
            config.enable_stream(
                rs.stream.depth,
                self.camera_config.depth_config.original_resolution.x,
                self.camera_config.depth_config.original_resolution.y,
                rs.format.z16,
                self.camera_config.depth_config.fps)

        print("Starting the camera pipeline and waiting for the frames...")
        self.pipeline.start(config)
        if self.camera_config.HasField("depth_config"):
            self.align = rs.align(rs.stream.color)
        self.pipeline.wait_for_frames(5000)

    def __del__(self):
        print("Closing RealSense camera")
        self.pipeline.stop()

    def GetFrame(self):
        frames = self.pipeline.wait_for_frames()
        if self.camera_config.HasField("depth_config"):
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
        else:
            color_frame = frames.get_color_frame()
            depth_frame = None
        if color_frame is None:
            print("Can't capture frame!")
            return None, None
        color_image = np.asanyarray(color_frame.get_data())
        
        if depth_frame is not None:
            depth_image = np.asanyarray(depth_frame.get_data())
        else:
            depth_image = None
        return color_image, depth_image


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
                return frame
    
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
