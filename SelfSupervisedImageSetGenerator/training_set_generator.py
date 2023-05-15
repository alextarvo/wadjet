import sys
import os

import argparse

import cv2
import numpy as np
from pathlib import Path

import cut_images_pb2

NUM_BALLS = 15

class DatasetWriter():
    def __init__(self, dataset_path, output_folder_name):
        if not os.path.exists(dataset_path):
            # If the folder doesn't exist, create it
            os.makedirs(dataset_path)
            print(f"Created dataset main folder in '{dataset_path}'")
        
        self.output_folder_path = os.path.join(dataset_path, output_folder_name)
        if not os.path.exists(self.output_folder_path):
        # if os.path.exists(self.output_folder_path):
        #     raise "The folder '{output_folder_name}' already exists in dataset '{dataset_path}'"
            # If the subfolder doesn't exist, create it
            os.makedirs(self.output_folder_path)
        else:
            print(f"WARNING: The folder '{output_folder_name}' already exists in dataset '{dataset_path}'. Files in it will be overwritten!")
        self.frame_seq = 1

    def addFrame(self, frame):
        file_name = f"{self.output_folder_path}/{self.frame_seq}.png" 
        cv2.imwrite(file_name, frame)
        self.frame_seq = self.frame_seq+1


class CutImagesForBall:
    def __init__(self, ball_no, file_name):
        self.ball_no = ball_no 
        self.cut_images_set = cut_images_pb2.CutImageSet()
        with open(file_name, 'rb') as f:
            binary_proto_message = f.read()
            self.cut_images_set.ParseFromString(binary_proto_message)

    @classmethod
    def create(cls, ball_no, input_images_folder):
        file_name = f"{input_images_folder}/ball_{ball_no}.pb"
        if Path(file_name).is_file():
            return cls(ball_no, file_name)
        else:
            return None  

    def sample(self):
        pass


parser = argparse.ArgumentParser()

parser.add_argument(
    '--input_images_path',
    type=str,
    help='Folder, where the proto file with cut out images reside.')

parser.add_argument(
    '--output_dataset_path',
    type=str,
    help='Path where to store the synthethic images.')


args = parser.parse_args()

background_video_file_name = f"{args.input_images_path}/background.avi"

# Capture the first frame of the video stream
cap = cv2.VideoCapture(background_video_file_name)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
channels = 3

print("Opened background video. (width x height x channels x frames) = (%d, %d, %d, %f)" % (frame_width, frame_height, channels, total_frames))

input_video = np.zeros((frame_height, frame_width, channels, total_frames), dtype=np.int8) 

frame_ctr = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    input_video[:, :, :, frame_ctr] = frame
    frame_ctr = frame_ctr + 1
cap.release()

images_for_ball = {}
for ball_id in range(1, NUM_BALLS+1):
    cutout_set = CutImagesForBall.create(ball_id, args.input_images_path)
    images_for_ball[ball_id] = cutout_set
    if cutout_set is None:
        print(f"Set of cut images is empty for the ball '{ball_id}'. No images of that ball will be sampled.")

