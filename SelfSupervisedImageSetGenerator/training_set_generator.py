import sys
import os
import math

import argparse
import logging

import cv2
import numpy as np
from pathlib import Path

import cut_images_pb2

logging.basicConfig(level=logging.INFO)

NUM_BALLS = 16

class DatasetWriter():
    def __init__(self, output_folder_path):
        self.output_folder_path = output_folder_path 
        if not os.path.exists(self.output_folder_path):
            os.makedirs(self.output_folder_path)
        self.frame_seq = 0

    def addFrame(self, frame, ball_centers):
        self.frame_seq = self.frame_seq+1
        file_name = f"{self.output_folder_path}/{self.frame_seq}.png" 
        cv2.imwrite(file_name, frame)


class CutImagesForBall:
    def __init__(self, ball_no, file_name):
        self.ball_no = ball_no 
        self.cut_images_set = cut_images_pb2.CutImageSet()
        with open(file_name, 'rb') as f:
            binary_proto_message = f.read()
            self.cut_images_set.ParseFromString(binary_proto_message)

    @classmethod
    def Create(cls, ball_no, input_images_folder):
        file_name = f"{input_images_folder}/ball{ball_no}.pb"
        if Path(file_name).is_file():
            return cls(ball_no, file_name)
        else:
            return None  

    def Sample(self):
        idx = np.random.randint(len(self.cut_images_set.cut_images))
        return self.cut_images_set.cut_images[idx]


class ImageGenerator:
    def __init__(self, backgrounds):
        self.cut_images = {}
        self.backgrounds = backgrounds 

    def AddCutImages(self, ball_no, image_set):
        if ball_no in self.cut_images:
            logging.warning(f"Set of cut-out images for ball {ball_no} already exists in the set. Overwriting")  
        self.cut_images[ball_no] = image_set

    def ballsMayOverlap(self, ball_center, other_ball_centers):
        for ball_center2 in other_ball_centers:
            dx = (ball_center.x - ball_center2.x)*(ball_center.x - ball_center2.x)
            dy = (ball_center.y - ball_center2.y)*(ball_center.y - ball_center2.y)
            dist = math.sqrt(dx + dy)
            if dist < ball_center.r + ball_center2.r:
                return True
        return False
    
    def insertBallAtOriginLocation(self, synth_image, ball_image, ball_center, ball_centers):
        ball_center.x = ball_image.coordinates.x
        ball_center.y = ball_image.coordinates.y
        
        if self.ballsMayOverlap(ball_center, ball_centers.values()):
            logging.warning(f"Likely overlap for balls; skipping image")
            return False

        # We found a good location.
        ball_frame = np.frombuffer(ball_image.image, dtype=np.uint8).reshape(ball_image.rows, ball_image.cols, ball_image.channels)
        # Coordinates of the left upper corner on the image where to put in a ball.
        target_x = ball_center.x - int(ball_image.cols / 2)
        target_y = ball_center.y - int(ball_image.rows / 2)

        if (target_x < 0 or
            target_y < 0 or
            target_y + ball_image.rows > synth_image.shape[0] or
            target_x+ball_image.cols > synth_image.shape[1]):
            logging.warning(f"A patch with ball centered at {ball_center.x}, {ball_center.y} is outside the synthethic image frame")
            return False
        
        synth_image[target_y:target_y+ball_image.rows, target_x:target_x+ball_image.cols] = ball_frame
        return True
        
    
    def insertBallAtRandomLocation(self, synth_image, ball_image, ball_center, ball_centers):
        # On the synthetic image, these are the boundaries where the image of the ball can be pasted into.
        # We compute them in a way that the image of the ball won't get outside the synthetic image boundaries.
        min_x, max_x = int(ball_image.cols / 2), synth_image.shape[1] - int(ball_image.cols / 2)
        min_y, max_y = int(ball_image.rows / 2), synth_image.shape[0] - int(ball_image.rows / 2)

        attempts = 0
        while attempts < 1000:
            # Try to insert the image of the ball into a random location on the background.
            ball_center.x = np.random.randint(min_x, max_x)
            ball_center.y = np.random.randint(min_y, max_y)
            if self.ballsMayOverlap(ball_center, ball_centers.values()):
                attempts += 1
                continue
                # We found a good location.
                ball_frame = np.frombuffer(ball_image.image, dtype=np.uint8).reshape(ball_image.rows, ball_image.cols, ball_image.channels)
                # Coordinates of the left upper corner on the image where to put in a ball.
                target_x = ball_center.x - int(ball_image.cols / 2)
                target_y = ball_center.y - int(ball_image.rows / 2)

                if (target_x < 0 or
                    target_y < 0 or
                    target_y + ball_image.rows > synth_image.shape[0] or
                    target_x+ball_image.cols > synth_image.shape[1]):
                    logging.warning(f"A patch with ball centered at {ball_center.x}, {ball_center.y} is outside the synthethic image frame")
                    return False
                
                synth_image[target_y:target_y+ball_image.rows, target_x:target_x+ball_image.cols] = ball_frame
                break

            if  attempts >= 1000:
                logging.warning(f"Failed to find a position for a ball: the number of attempts exceeded")
                return False
        return True


    def GenerateImage(self):
        # TODO: currently we generate image with all the balls that are present in the cut_images set.
        # In future, generate it with the random number of balls.
        background_idx = np.random.randint(self.backgrounds.shape[3])
        synth_image = self.backgrounds[:,:,:,background_idx].copy()

        # Randomly generate the image of balls on the background
        ball_centers = {}
        for ball_no in range(0, NUM_BALLS):
            if ball_no not in self.cut_images.keys():
                continue
            ball_image = self.cut_images[ball_no].Sample()
            ball_center = cut_images_pb2.BallCoordinates()
            ball_center.x = -1
            ball_center.y = -1
            ball_center.r = ball_image.coordinates.r

            success = self.insertBallAtOriginLocation(synth_image, ball_image, ball_center, ball_centers)
            if not success:
                return None, None

            ball_centers[ball_no] = ball_center

        return synth_image, ball_centers



parser = argparse.ArgumentParser()

parser.add_argument(
    '--input_images_path',
    type=str,
    help='Folder, where the proto file with cut out images reside.')

parser.add_argument(
    '--output_dataset_path',
    type=str,
    help='Path where to store the synthethic images.')

parser.add_argument(
    '--num_images_to_generate',
    type=int,
    help='The number of synthetic images to be generated.')


args = parser.parse_args()

background_video_file_name = f"{args.input_images_path}/background.avi"
if not Path(background_video_file_name).is_file():
    logging.fatal(f"Background video file {background_video_file_name} does not exist.")
    sys.exit(1)

# Capture the first frame of the video stream
cap = cv2.VideoCapture(background_video_file_name)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
channels = 3

logging.info("Opened background video. (width x height x channels x frames) = (%d, %d, %d, %f)" % (frame_width, frame_height, channels, total_frames))

backgrounds = np.zeros((frame_height, frame_width, channels, total_frames), dtype=np.uint8) 

frame_ctr = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    backgrounds[:, :, :, frame_ctr] = frame
    frame_ctr = frame_ctr + 1
cap.release()

generator = ImageGenerator(backgrounds) 
for ball_id in range(0, NUM_BALLS):
    cutout_set = CutImagesForBall.Create(ball_id, args.input_images_path)
    if cutout_set is None:
        logging.warning(f"Set of cut images is empty for the ball '{ball_id}'. No images of that ball will be sampled.")
    else:
        generator.AddCutImages(ball_id, cutout_set)

dataset_writer = DatasetWriter(args.output_dataset_path) 
for i in range(args.num_images_to_generate):
    image, centers = generator.GenerateImage()
    if image is not None and centers is not None:
        dataset_writer.addFrame(image, centers)
        # For debug purposes
        # print(centers)
        #
        # cv2.namedWindow('Synthetic image', cv2.WINDOW_NORMAL)
        # cv2.imshow('Synthetic image', image)
        # key = cv2.waitKey()
        # cv2.destroyAllWindows()
