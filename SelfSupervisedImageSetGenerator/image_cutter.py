import sys
import os

import argparse

import cv2
import numpy as np

import cut_images_pb2
# import camera_config_pb2 as camera_config
#
# from google.protobuf.json_format import Parse

# The tool to "cut out" the images of a ball from the input pictures, and
# to save them into a seaprate .proto file.
# Later on, the "cut out" images will be pasted into random locations into the previously 
# captured background. 
 
# Run the program:
# --video_path=/home/iscander/eclipse-workspace/ball14_2023_05_11.avi --skip_n_frames=0 --output_path=/home/iscander/eclipse-workspace/ball14_2023_05_11.pb --show_ui=True

class BallDetector():
    """ A class that detects a moving ball in the image using computer vision techniques.
    
    We try to detect elliptic contours in the image. We don't expect any other objects except a single ball,
    so if a contour detected - it will correspond to that ball.
    """

    def __init__(self):
        """ Initialize the background subtractor """
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.fgbg.setDetectShadows(False)
        
        # The number of frames to train the background subtractor.
        self.frames_compute_background = 30
        
        self.frames_count = 0
        
        # Parameters of the ball detector.
        # How much the shape of the contour will be close to the circle.
        self.min_circularity = 0.52
        # Minimum and maximum areas of a candidate ball.
        self.min_blob_area = 200
        self.max_blob_area = 2000


    def detectBall(self, frame):
        """ Detect a single ball in the frame. """
        
        ball_centers = []
        if self.frames_count < self.frames_compute_background:
            self.fgbg.apply(frame)
            self.frames_count = self.frames_count  + 1
            return_frame = frame
        else:
            diff = self.fgbg.apply(frame,  learningRate=0)
            return_frame = diff
            return_frame = cv2.erode(return_frame, None, iterations=1)
            return_frame = cv2.dilate(return_frame, None, iterations=4)

            contours, _ = cv2.findContours(return_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return_frame = cv2.cvtColor(return_frame, cv2.COLOR_GRAY2RGB)
            for _, c in enumerate(contours):
                # Get blob area:
                blobArea = cv2.contourArea(c)
                # Get blob periemter
                blobPerimeter = cv2.arcLength(c, True)
                # Compute circulariity of the blob.
                blobCircularity = (4 * 3.1416 * blobArea)/(blobPerimeter**2)

                # print("Candidate blob. Area: %02f, periemeter: %02f, circularity: %02f" % (blobArea, blobPerimeter, blobCircularity))
                # Compute the center and radius of the ball:
                if blobCircularity > self.min_circularity and blobArea > self.min_blob_area and blobArea < self.max_blob_area:
                    # Approximate the contour to a circle:
                    (x, y), radius = cv2.minEnclosingCircle(c)
                    ball_centers.append((int(x), int(y), int(radius)))

            if ball_centers is not None:
                # Draw the detected circles on the mask image
                for (x, y, r) in ball_centers:
                    # x, y = center
                    cv2.circle(return_frame, (x, y), r, (0, 255, 0), 2)
                    cv2.circle(return_frame, (x, y), 2, (0, 0, 255), 3)
            
    
        return ball_centers, return_frame



def cutBall(frame, x, y, r):
    """ Cuts out the ball from the input frame.
    
    In the process, verifies that the bounding box of the ball doesn't exceed input image dimensions.
    
    Args:
      frame: the input frame
      x, y: coordinates of the center of the ball in the frame.
      r: radius of the ball. 2*r make the size of the ball bounding box.
      
    Returns:
      a "cut out" image of the ball.
    """
      
    top = y-r
    if top < 0:
        top = 0

    bottom = y+r
    if bottom > frame.shape[0]:
        bottom = frame.shape[0]

    left = x-r
    if left < 0:
        left = 0
    
    right = x+r
    if right > frame.shape[1]:
        right = frame.shape[1]
        
    cut_img = frame[top:bottom, left:right]
    
    # , right-left, top-bottom, max((right-left)/2, (top-bottom)/2)
    return cut_img


def displaySingleImage(frame, fps, window_name):
    """Displays a single image in the OpenCV window.
    
    Reacts to user's keypresses. "q" forces to stop the current processing loop.
    "p" pauses it. 
    
    Args:    
      frame: an input image
      fps: the frame rate of the image stream.
      window_name: name of the window to display image in.
      
    Returns:
      True to continue processing of an image. False to exit the processing loop.
    """  
    cv2.imshow(window_name, frame)
    delay = int (1000 / fps)
    # Check for key press to exit the loop
    key = cv2.waitKey(delay) 
    if key == ord('q'):
        return False
    if key == ord('p'):
        cv2.waitKey(-1) # wait until any key is pressed to continue
    return True

#
# def doTestDetector():
#     if args.config_path is None:
#         print("Config file name should be passed to an application")
#         sys.exit(1)
#
#     if os.path.exists(args.config_path):
#         with open(args.config_path, "rt") as f:
#             config_proto_text = f.read()
#         config = Parse(config_proto_text, camera_config.CapturerConfig())
#
#         try:
#             capturer= video_capturers.CreateVideoCapturer(config.camera_config)
#         except Exception:
#             sys.exit(1)
#
#         # Set up a video preprocessor
#         self.video_preprocessor = video_preprocessors.CapturePreprocessingPipeline(
#             self.capturer, config.preprocess_config)
#
#         while True:
#             video_preprocessor.Process()
#             frame = self.video_preprocessor.GetProcessedFrame()


parser = argparse.ArgumentParser()

parser.add_argument(
    '--input_video_path',
    type=str,
    help='File where the captured video resides.')

parser.add_argument(
    '--output_path',
    type=str,
    help='Path where to store the output images.')

parser.add_argument(
    '--skip_n_frames',
    type=int,
    default=0,
    help='For each frame written to the output, N frames will be skipped. Specifies the number of frames to skip.')

parser.add_argument(
    '--show_ui',
    help='Display the original image, binary mask, and cut image to the user.')

args = parser.parse_args()
#
# if args.test_detector:
#     doTestDetector()
#     exit()

# Capture the first frame of the video stream
cap = cv2.VideoCapture(args.input_video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create a window to display the video - original and the mask for detecting balls
if args.show_ui:
    cv2.namedWindow('Combined Videos', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Combined Videos', frame_width, frame_height*2)

video_file_name = os.path.basename(args.input_video_path)
print(f"Parsing video '{video_file_name}'...")
video_file_name, _ = os.path.splitext(video_file_name)

detector = BallDetector()

# The resulting dataset with images.
cut_images_set = cut_images_pb2.CutImageSet()

frame_counter = 1
while True:
    # Read a frame from the first video
    ret, frame = cap.read()
    if not ret:
        break
    balls, frame2 = detector.detectBall(frame)

    for (x, y, r) in balls:
        # Iterate throught the balls we detected on an image
        if len(balls) > 1:
            print("More than 1 ball detected! Skipping frame %d." % frame_counter)
            break

        # cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
        # cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

        if r <= 0:
            print("Radius of the detected ball is negative. Frame %d, (x, y, r) = (%d, %d, %d)" % (frame_counter, x, y, r))

        # Cut out the ball out of the image.
        cut_img = cutBall(frame, x, y, r)
        
        #now, save the cut image in the proto where all the "cuts" will be saved into.
        cut_image = cut_images_set.cut_images.add()
        cut_image.coordinates.x = x
        cut_image.coordinates.y = y
        cut_image.coordinates.r = r
        (cut_image.rows, cut_image.cols, cut_image.channels) =  cut_img.shape
        cut_image.image = cut_img.tobytes()

    if args.show_ui:
        combined_frame = cv2.vconcat([frame, frame2])
        if not displaySingleImage(combined_frame, fps, 'Combined Videos'):
            break

    frame_counter = frame_counter+1

cap.release()
cv2.destroyAllWindows()

print("Parse complete. Total %d frames were read. %d frames were written to the output" %
      (frame_counter, len(cut_images_set.cut_images)))

# Now save all these images of a ball into a file.
if args.output_path is not None:
    print("Saving the cut imageset into a file %s" % args.output_path)
    binary_msg = cut_images_set.SerializeToString()
    directory = os.path.dirname(args.output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(args.output_path, 'wb') as f:
        f.write(binary_msg)
    f.close()

    # Now verify that images were saved successfully by loading them from the file. 
    cut_images_set_proof = cut_images_pb2.CutImageSet()
    with open(args.output_path, 'rb') as f:
        binary_proto_message = f.read()
        cut_images_set_proof.ParseFromString(binary_proto_message)
        print("Read the image set successfully. Total frames available: %d" % len(cut_images_set_proof.cut_images))
        if args.show_ui:
            cv2.namedWindow('Loaded frames', cv2.WINDOW_NORMAL)
            for ser_frame in cut_images_set_proof.cut_images:
                frame = np.frombuffer(ser_frame.image, dtype=np.uint8).reshape(ser_frame.rows, ser_frame.cols, ser_frame.channels)
                if not displaySingleImage(frame, fps, 'Loaded frames'):
                    break
        f.close()                
cv2.destroyAllWindows()
