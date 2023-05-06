import sys
import os

import argparse

import cv2

# Run the program:
# --video_path=/mnt/nfs/videos/game_2023_05_05.avi --dataset_path=/home/iscander/Datasets/SelfSupervised_2023_05_05 --skip_n_frames=100

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


parser = argparse.ArgumentParser()
parser.add_argument(
    '--video_path',
    type=str,
    help='File where the captured video resides.')

parser.add_argument(
    '--dataset_path',
    type=str,
    help='Path where the dataset is located. Frames from every video will be stored in a separate folder.')

parser.add_argument(
    '--skip_n_frames',
    type=int,
    default=100,
    help='For each frame written to the output, N frames will be skipped. Specifies the number of frames to skip.')

args = parser.parse_args()

# Capture the first frame of the video stream
cap = cv2.VideoCapture(args.video_path)

video_file_name = os.path.basename(args.video_path)
print(f"Parsing video '{video_file_name}'...")
video_file_name, _ = os.path.splitext(video_file_name)
dataset_writer = DatasetWriter(args.dataset_path, video_file_name)

frame_counter = 1
while True:
    # Read a frame from the first video
    ret, frame = cap.read()
    if not ret:
        break
    if frame_counter % args.skip_n_frames == 0:
        dataset_writer.addFrame(frame)
    frame_counter = frame_counter+1

cap.release()
print(f"Parse complete.")