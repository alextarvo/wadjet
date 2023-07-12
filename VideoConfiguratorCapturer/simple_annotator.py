#
# A simple script for manually annotating the images of the pool game.
# Randomly loads frames from the video of the pool game and shows them to a user.
# The user presses a button that corresponds to the ball ID ('w' - a white ball; "1"-"0" - balls 1 to 10;
# "f1" to "f4" - balls from 11 to 14), and draws a rectangle around it.
# "n" skips the currently proposed frame; "s" saves a labeled frame into a dataset; and "q" exits the 
# program.
# The dataset is in the Yolo format. The images are in stored as files "images/val/<image_index>.png" subfolder, and labels - into .txt files "images/val/<image_index>.txt"; indices start from 0.
#
# Usage example:
# --input_video=/home/iscander/Datasets/ground_truth/game_2023_05_28_artlight2.avi --output_dataset_path=/home/iscander/Datasets/ground_truth/game_2023_05_28_artlight --num_frames_to_label=10 --start_frame_idx_to_save=5 --pad_and_transform=True
# Here start_frame_idx_to_save parameters corresponds to the starting index _in the dataset_. This is
# necessary if we want to add more labeled images to already existing dataset.
#

import os
import glob
import logging
import random

import cv2
import numpy as np

import argparse

logging.basicConfig(level=logging.INFO)

# global vars, related to box drawing.
drawing = False # true if mouse is pressed
x1, y1 = -1, -1

class DatasetWriter():
    # A base abstract class that writes labeled images into a dataset.
    # The dataset format is not specified, and will be defined by a derived class.
    def __init__(self, output_folder_path, start_index=None):
        self.output_folder_path = output_folder_path 
        if not os.path.exists(self.output_folder_path):
            os.makedirs(self.output_folder_path)
        if start_index is not None:
            self.frame_seq=start_index

    def incrementFrameCounter(self):
        self.frame_seq = self.frame_seq+1

class YoloDatasetWriter(DatasetWriter):
    # A class that writes labeled images into the dataset in Yolo format.

    def __init__(self, output_folder_path, start_index=None):
        super().__init__(output_folder_path, start_index)
        self.labels_subfolder = f"{self.output_folder_path}/labels/val" 
        if not os.path.exists(self.labels_subfolder):
            os.makedirs(self.labels_subfolder)
        self.images_subfolder = f"{self.output_folder_path}/images/val" 
        if not os.path.exists(self.images_subfolder):
            os.makedirs(self.images_subfolder)

    def AddFrame(self, frame, ball_boxes_dict):
        self.incrementFrameCounter()
        labels_file_name = f"{self.labels_subfolder}/{self.frame_seq}.txt"
        with open(labels_file_name, 'w') as f:
            for ball_id, ball_coord in ball_boxes_dict.items():
                x = ball_coord["x"]
                y = ball_coord["y"]
                width = ball_coord["width"]
                height = ball_coord["height"]
                yolo_entry = f"{ball_id} {x} {y} {width} {height}\n"
                f.write(yolo_entry)

        image_file_name = f"{self.images_subfolder}/{self.frame_seq}.png" 
        cv2.imwrite(image_file_name, frame)


def draw_rectangle(event, x, y, flags, param):
    # mouse callback function.
    # Draws a rectangle around the ball on an image, and stores the
    # coordinates of the ball in the Yolo format.
    global x1, y1, drawing, ball_id, ball_boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            # draw a rectangle
            frame_resized_copy = frame_resized.copy()
            cv2.rectangle(frame_resized_copy, (x1, y1), (x, y), (0, 255, 0), 1)
            cv2.imshow('image', frame_resized_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # finalize the rectangle
        cv2.rectangle(frame_resized, (x1, y1), (x, y), (0, 255, 0), 2)
        # Dimensions of the frame itself, in pixels.
        width = frame_resized.shape[0]
        height = frame_resized.shape[1]
        # Yolo format: ball center, relative to the image frame
        yolo_x1 = float(abs(x+x1)/2)/width
        yolo_y1 = float(abs(y+y1)/2)/height
        # Ball width and height, also relative.
        yolo_width = float(abs(x-x1))/width
        yolo_height = float(abs(y-y1))/height
        logging.info(f"Ball {ball_id} at ({x1}, {y1}), ({x}, {y}). Relative: ({yolo_x1}, {yolo_y1}), ({yolo_width}, {yolo_height})")
        ball_boxes[ball_id] = {"x": yolo_x1, "y": yolo_y1, "width": yolo_width, "height": yolo_height}

def PadToSize(image, target_width, target_height):
    # This is a temp code for the final project. Make the img squeeze into 640 pixels,
    # and then pad it with black so it will be 640x640, suitable for Yolo detector.
    height, width = image.shape[:2]
    ratio = target_width / float(width)
    target_height_current = int(height*ratio)
    image = cv2.resize(image, (target_width, target_height_current))
    padding_color = [0,0,0]
    padding_height = target_height - target_height_current
    image = cv2.copyMakeBorder(image, 0, padding_height, 0, 0, cv2.BORDER_CONSTANT, value=padding_color)
    # end of temp code
    return image, ratio


parser = argparse.ArgumentParser()
parser.add_argument(
    '--input_video',
    type=str,
    help='Path to the input video of the game of pool Frames will be selected randomly from it.')

parser.add_argument(
    '--output_dataset_path',
    type=str,
    help='Path where to store the dataset of annotated images, in Yolo format.')

parser.add_argument(
    '--pad_and_transform',
    help='Path where to store the synthethic images.')

parser.add_argument(
    '--num_frames_to_label',
    type=int,
    help='The number of frames to fetch from the input video for labeling.')

parser.add_argument(
    '--start_frame_idx_to_save',
    type=int,
    help='The starting index of the frame to save into the Yolo dataset.')

args = parser.parse_args()


writer = YoloDatasetWriter(args.output_dataset_path, args.start_frame_idx_to_save)

# Target dimensions of an image.
target_width = 640
target_height = 640

do_stop = False

video = cv2.VideoCapture(args.input_video)
# Get total frames count.
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
# Randomly sample the indices of video frames we should label
frame_indices = random.sample(range(total_frames), args.num_frames_to_label)
logging.info(f"Random frames to label: {frame_indices}")

for i in range(total_frames):
    ret, frame = video.read()
    if not ret:
        break
    if i not in frame_indices:
        continue

    logging.info(f"Labeling frame {i}")

    ball_id = -1
    ball_boxes = {}
    
    # Load an image from the video.
    frame_padded, _ = PadToSize(frame, target_width, target_height)
    frame_resized = cv2.resize(frame_padded, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_rectangle)
    
    while(1):
        # Event loop for actual labeling. Read a key pressed, and do an appropriate action.
        cv2.imshow('image', frame_resized)
        keyCode = cv2.waitKeyEx(-1) & 0xff
        print(keyCode)
        if keyCode == ord('w'):
            ball_id = 0
        elif keyCode == ord('1'):
            ball_id = 1
        elif keyCode == ord('2'):
            ball_id = 2
        elif keyCode == ord('3'):
            ball_id = 3
        elif keyCode == ord('4'):
            ball_id = 4
        elif keyCode == ord('5'):
            ball_id = 5
        elif keyCode == ord('6'):
            ball_id = 6
        elif keyCode == ord('7'):
            ball_id = 7
        elif keyCode == ord('8'):
            ball_id = 8

        elif keyCode == ord('9'): #F9
            ball_id = 9
        elif keyCode == ord('0'): #F10
            ball_id = 10
        elif keyCode == 190: #F1
            ball_id = 11
        elif keyCode == 191: #F2
            ball_id = 12
        elif keyCode == 192: #F3
            ball_id = 13
        elif keyCode == 193: #F4
            ball_id = 14
        elif keyCode == 194: #F5
            ball_id = 15

        elif keyCode == ord('q'):
            print("Stopping labeling and exiting the program")
            do_stop = True
            break
        elif keyCode == ord('n'):
            print("Skipping current frame")
            break
        elif keyCode == ord('s'):
            # print all rectangles
            for ball_id, box in ball_boxes.items():
                print(f"Ball ID: {ball_id}, box: {box}")
            writer.AddFrame(frame_padded, ball_boxes)
            break
        else:
            print("Unknown key")
    
    cv2.destroyAllWindows()
    

    if do_stop:
        break

print("Labeling complete")
