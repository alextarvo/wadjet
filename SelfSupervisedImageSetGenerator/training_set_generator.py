#
# A program that loads a background video of the table and "cut out" images of the balls on that table.
# Then it starts inserting the randomly selected images of the balls at the background, to generate
# a realistic synthetic images of the table with balls on it. We insert the balls into original locations,
# because they also contain pieces of the table background. Inserting them into random locations will
# yield a "patchy" image of the table.
#

import sys
import os
import math
import glob

import argparse
import logging
import json

import cv2
import numpy as np
from pathlib import Path

import cut_images_pb2

logging.basicConfig(level=logging.INFO)

NUM_BALLS = 16

class BallBox:
    def __init__(self, left, top, right, bottom):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    @classmethod
    def RecomputeRatio(cls, ball_box, ratio):
        left = int(ball_box.left*ratio)
        top = int(ball_box.top*ratio)
        right = int(ball_box.right*ratio)
        bottom = int(ball_box.bottom*ratio)
        return cls(left, top, right, bottom)
    
# TODO(alexta, 2023/07/11): ots of the code below is duplicated in the VideoConfiguratorCapturer/simple_annotator.py.
# We need to put it into a common library.
class DatasetWriter():
    def __init__(self, output_folder_path):
        self.output_folder_path = output_folder_path 
        if not os.path.exists(self.output_folder_path):
            os.makedirs(self.output_folder_path)
        self.frame_seq = 0

    def incrementFrameCounter(self):
        self.frame_seq = self.frame_seq+1
        
    def AddFrame(self, frame):
        self.incrementFrameCounter()


class SimpleDatasetWriter(DatasetWriter):
    def __init__(self, output_folder_path):
        super().__init__(output_folder_path)

    def AddFrame(self, frame, ball_centers):
        super().AddFrame(frame)
        file_name = f"{self.output_folder_path}/{self.frame_seq}.png" 
        cv2.imwrite(file_name, frame)
        file_name = f"{self.output_folder_path}/{self.frame_seq}.json"
        dict_json_centers = {}
        for ball_id, ball_coordinates in ball_centers.items():
            dict_json_centers[ball_id] = {"x": ball_coordinates.x, "y": ball_coordinates.y, "r": ball_coordinates.r}
        with open(file_name, 'w') as f:
            json.dump(dict_json_centers, f)


class YoloDatasetWriter(DatasetWriter):
    def __init__(self, output_folder_path, dataset_type, width, height):
        super().__init__(output_folder_path)
        self.width = width
        self.height = height
        self.ratio = 1.0
        self.images_subfolder = f"{self.output_folder_path}/images/{dataset_type}" 
        if not os.path.exists(self.images_subfolder):
            os.makedirs(self.images_subfolder)
        self.labels_subfolder = f"{self.output_folder_path}/labels/{dataset_type}" 
        if not os.path.exists(self.labels_subfolder):
            os.makedirs(self.labels_subfolder)
        self.show_ui = False
        self.output_boxes = False

    def SetRecomputeRatio(self, ratio):
        self.ratio = ratio

    def SetShowUI(self, show_ui):
        self.show_ui = show_ui

    def SetOutputBoxes(self, output_boxes):
        self.output_boxes = output_boxes

    def showFrame(self, frame):
        cv2.namedWindow('Synthetic image', cv2.WINDOW_NORMAL)
        cv2.imshow('Synthetic image', frame)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    def outputBox(self, frame, x, y, obj_width, obj_height):
        x1 = int((x-obj_width/2)*self.width)
        y1 = int((y-obj_height/2)*self.height)
        x2 = int((x+obj_width/2)*self.width)
        y2 = int((y+obj_height/2)*self.height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
        pass

    def AddFrame(self, frame, ball_boxes):
        super().AddFrame(frame)
        labels_file_name = f"{self.labels_subfolder}/{self.frame_seq}.txt"
        with open(labels_file_name, 'w') as f:
            for ball_id, ball_box in ball_boxes.items():
                box_adjusted = BallBox.RecomputeRatio(ball_box, self.ratio)
                yolo_x = float(box_adjusted.left + box_adjusted.right) / (2 * self.width)
                yolo_y = float(box_adjusted.top + box_adjusted.bottom) / (2 * self.height)
                yolo_width = float(box_adjusted.right - box_adjusted.left) / self.width
                yolo_height = float(box_adjusted.bottom - box_adjusted.top) / self.height
                yolo_entry = f"{ball_id} {yolo_x} {yolo_y} {yolo_width} {yolo_height}\n"
                f.write(yolo_entry)
                if (self.output_boxes):
                    self.outputBox(frame, yolo_x, yolo_y, yolo_width, yolo_height)

        image_file_name = f"{self.images_subfolder}/{self.frame_seq}.png" 
        cv2.imwrite(image_file_name, frame)
        if self.show_ui:
            self.showFrame(frame)


class CutImagesForBall:
    
    MIN_SET_SIZE = 100
    
    def __init__(self, ball_no, file_name):
        self.ball_no = ball_no 
        self.cut_images_set = cut_images_pb2.CutImageSet()
        with open(file_name, 'rb') as f:
            binary_proto_message = f.read()
            self.cut_images_set.ParseFromString(binary_proto_message)

    @classmethod
    def Create(cls, ball_no, input_images_folder):
        file_name = f"{input_images_folder}/ball{ball_no}.pb"
        image_set = None
        if Path(file_name).is_file():
            image_set = cls(ball_no, file_name)
        else:
            return None
        num_images = len(image_set.cut_images_set.cut_images)
        if num_images <  cls.MIN_SET_SIZE:
            logging.warning(f"Cannot create image set for ball {ball_no}: loaded only {num_images}; minimum required is {cls.MIN_SET_SIZE}")
            return None
        else:
            logging.info(f"Created image set for ball {ball_no}: loaded {num_images} images")
        return image_set

    def Sample(self):
        idx = rng.integers(len(self.cut_images_set.cut_images))
        return self.cut_images_set.cut_images[idx]


class ImageGenerator:
    def __init__(self, backgrounds):
        self.cut_images = {}
        self.backgrounds = backgrounds
        self.max_overlap_px = 3 

    def AddCutImages(self, ball_no, image_set):
        if ball_no in self.cut_images:
            logging.warning(f"Set of cut-out images for ball {ball_no} already exists in the set. Overwriting")  
        self.cut_images[ball_no] = image_set

    def ballsMayOverlap(self, ball_center, other_ball_centers):
        for ball_center2 in other_ball_centers:
            dx = (ball_center.x - ball_center2.x)*(ball_center.x - ball_center2.x)
            dy = (ball_center.y - ball_center2.y)*(ball_center.y - ball_center2.y)
            dist = math.sqrt(dx + dy)
            if dist < ball_center.r + ball_center2.r + self.max_overlap_px:
                return True
        return False
    
    def insertBallAtOriginLocation(self, synth_image, ball_image, ball_center, ball_centers):
        ball_center.x = ball_image.coordinates.x
        ball_center.y = ball_image.coordinates.y
        
        if self.ballsMayOverlap(ball_center, ball_centers.values()):
            logging.info(f"Likely overlap for balls; skipping image")
            return False, None

        # We found a good location.
        ball_frame = np.frombuffer(ball_image.image, dtype=np.uint8).reshape(ball_image.rows, ball_image.cols, ball_image.channels)
        ball_image_cols = ball_image.cols
        ball_image_rows = ball_image.rows
        assert ball_image_cols == ball_frame.shape[1]
        assert ball_image_rows == ball_frame.shape[0]

        # Coordinates of the left upper corner on the image where to put in a ball.
        target_x = ball_center.x - int(ball_image_cols / 2)
        target_y = ball_center.y - int(ball_image_rows / 2)

        ball_frame_copy = ball_frame.copy()
        if target_x < 0:
            # ball is touching the wall of the table on the left and gets out of the boundary
            # of the synthetic image
            if target_x + ball_image_cols < 0:
                # some error occurred. The ball is completely outside the image
                logging.warn(f"A patch with ball centered at {ball_center.x}, {ball_center.y} is completely outside the left boundary of a synthethic image frame")
                return False, None
            ball_frame_copy = ball_frame_copy[:, -1*target_x:, :]
            ball_image_cols += target_x
            target_x = 0
        elif target_x+ball_image_cols > synth_image.shape[1]:
            # ball is touching the wall of the table on the right
            if target_x > synth_image.shape[1]:
                # some error occurred. The ball is completely outside the image
                logging.warn(f"A patch with ball centered at {ball_center.x}, {ball_center.y} is completely outside the right boundary of a synthethic image frame")
                return False, None
            overlap_x = target_x+ball_image_cols - synth_image.shape[1]
            ball_image_cols -= overlap_x
            ball_frame_copy = ball_frame_copy[:, :ball_image_cols, :]

        if target_y < 0:
            if target_y + ball_image_rows < 0:
                # some error occurred. The ball is completely outside the image
                logging.warn(f"A patch with ball centered at {ball_center.x}, {ball_center.y} is completely outside the upper boundary of a synthethic image frame")
                return False, None
            ball_frame_copy = ball_frame_copy[-1*target_y:, :, :]
            ball_image_rows += target_y
            target_y = 0
        elif target_y + ball_image_rows > synth_image.shape[0]:
            # ball is touching the wall of the table on the bottom
            if target_y > synth_image.shape[0]:
                # some error occurred. The ball is completely outside the image
                logging.warn(f"A patch with ball centered at {ball_center.x}, {ball_center.y} is completely outside the lower boundary of a synthethic image frame")
                return False, None
            overlap_y = target_y+ball_image_rows - synth_image.shape[0]
            ball_image_rows -= overlap_y
            ball_frame_copy = ball_frame_copy[:ball_image_rows, :, :]
            

        if (target_x < 0 or
            target_y < 0 or
            target_y + ball_frame_copy.shape[0] > synth_image.shape[0] or
            target_x+ball_frame_copy.shape[1] > synth_image.shape[1]):
            logging.info(f"A patch with ball centered at {ball_center.x}, {ball_center.y} is outside the synthethic image frame")
            return False, None

        box = BallBox(target_x, target_y, target_x+ball_image_cols, target_y+ball_image_rows)
        synth_image[box.top:box.bottom, box.left:box.right] = ball_frame_copy
        return True, box


    def insertBallAtRandomLocation(self, synth_image, ball_image, ball_center, ball_centers):
        # On the synthetic image, these are the boundaries where the image of the ball can be pasted into.
        # We compute them in a way that the image of the ball won't get outside the synthetic image boundaries.
        min_x, max_x = int(ball_image.cols / 2), synth_image.shape[1] - int(ball_image.cols / 2)
        min_y, max_y = int(ball_image.rows / 2), synth_image.shape[0] - int(ball_image.rows / 2)

        attempts = 0
        while attempts < 1000:
            # Try to insert the image of the ball into a random location on the background.
            ball_center.x = rng.integers(min_x, max_x)
            ball_center.y = rng.integers(min_y, max_y)
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
                    logging.info(f"A patch with ball centered at {ball_center.x}, {ball_center.y} is outside the synthethic image frame")
                    return False
                
                synth_image[target_y:target_y+ball_image.rows, target_x:target_x+ball_image.cols] = ball_frame
                break

            if  attempts >= 1000:
                logging.warning(f"Failed to find a position for a ball: the number of attempts exceeded")
                return False
        return True


    def GenerateImage(self, num_balls_generate=NUM_BALLS):
        # Get a random background image.
        background_idx = rng.integers(self.backgrounds.shape[3])
        synth_image = self.backgrounds[:,:,:,background_idx].copy()

        # If specified, get the random number of balls. 
        ball_no_mask = np.zeros(NUM_BALLS)
        mask_indices = rng.choice(NUM_BALLS, num_balls_generate, replace=False)
        ball_no_mask[mask_indices] = True

        # Randomly generate the image of balls on the background
        ball_centers = {}
        ball_boxes = {}
        for ball_no in range(0, NUM_BALLS):
            if ball_no not in self.cut_images.keys() or not ball_no_mask[ball_no]:
                continue
            ball_image = self.cut_images[ball_no].Sample()
            ball_center = cut_images_pb2.BallCoordinates()
            ball_center.x = -1
            ball_center.y = -1
            ball_center.r = ball_image.coordinates.r

            success, ball_box = self.insertBallAtOriginLocation(synth_image, ball_image, ball_center, ball_centers)
            if not success:
                return None, None, None

            ball_centers[ball_no] = ball_center
            ball_boxes[ball_no] = ball_box

        return synth_image, ball_centers, ball_boxes

def PadToSize(image, target_width, target_height):
    # This is a temp code for the final project. Make the img squeeze into 640 pixels,
    # and then pad it.
    height, width = image.shape[:2]
    ratio = target_width / float(width)
    target_height_current = int(height*ratio)
    image = cv2.resize(image, (target_width, target_height_current))
    padding_color = [0,0,0]
    padding_height = target_height - target_height_current
    image = cv2.copyMakeBorder(image, 0, padding_height, 0, 0, cv2.BORDER_CONSTANT, value=padding_color)
    # end of temp code
    return image, ratio

def GetNumberOfBalls(randomize_num_balls):
    if randomize_num_balls:
        return rng.integers(3, NUM_BALLS+1)
    else:
        return NUM_BALLS

def LoadBackgroundVideo(background_video_file_name):
    # background_video_file_name = f"{args.input_images_path}/background.avi"
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
    return backgrounds


parser = argparse.ArgumentParser()

parser.add_argument(
    '--input_images_path',
    type=str,
    help='Folder, where the proto file with cut out images reside.')

parser.add_argument(
    '--background_videos_template',
    type=str,
    help='A template for background videos. If more than one video matches a template, all of them will be used.')

parser.add_argument(
    '--output_dataset_path',
    type=str,
    help='Path where to store the synthethic images.')

parser.add_argument(
    '--num_train_images',
    type=int,
    help='The number of synthetic training images to be generated.')

parser.add_argument(
    '--num_val_images',
    type=int,
    help='The number of synthetic validation images to be generated.')

parser.add_argument(
    '--num_test_images',
    type=int,
    help='The number of synthetic test images to be generated.')

parser.add_argument(
    '--randomize_num_balls',
    help='If specified, the number of balls at the image will be randomly selected between 3 and 16.')

parser.add_argument(
    '--show_boxes',
    help='Show boxes around the balls on the resulting image (debug).')

parser.add_argument(
    '--show_ui',
    help='Display the original image, binary mask, and cut image to the user.')

parser.add_argument(
    '--random_seed',
    type=int,
    help='If specified, the random seed for numpy for reproducible results.')


args = parser.parse_args()

if args.random_seed:
    rng = np.random.default_rng(args.random_seed)
else:
    rng = np.random.default_rng()

backgrounds = None
background_video_files = glob.glob(args.background_videos_template)
for background_video_file in background_video_files:
    backgrounds_subset = LoadBackgroundVideo(background_video_file)
    if backgrounds is None:
        backgrounds = backgrounds_subset
    else:
        if backgrounds_subset.shape[0:3] != backgrounds.shape[0:3]:
            logging.fatal(f"Cannot load backgrounds from file {background_video_file}:")
            logging.fatal(f"shape {backgrounds_subset.shape} is different from the rest of the dataset shape {backgrounds.shape}")
            sys.exit(1)
        else:
            backgrounds = np.concatenate((backgrounds, backgrounds_subset), axis=3)


generator = ImageGenerator(backgrounds) 
for ball_id in range(0, NUM_BALLS):
    cutout_set = CutImagesForBall.Create(ball_id, args.input_images_path)
    if cutout_set is None:
        logging.warning(f"Set of cut images is empty for the ball '{ball_id}'. No images of that ball will be sampled.")
    else:
        generator.AddCutImages(ball_id, cutout_set)


target_width = 640
target_height = 640

images_generated = 0
# dataset_writer = SimpleDatasetWriter(args.output_dataset_path) 
dataset_writer = YoloDatasetWriter(args.output_dataset_path, "train", target_width, target_height)
dataset_writer.SetOutputBoxes(args.show_boxes)
dataset_writer.SetShowUI(args.show_ui)
while images_generated < args.num_train_images:
    image, centers, boxes = generator.GenerateImage(GetNumberOfBalls(args.randomize_num_balls))
    if image is not None and boxes is not None:
        image, ratio = PadToSize(image, target_width, target_height)
        dataset_writer.SetRecomputeRatio(ratio)
        dataset_writer.AddFrame(image, boxes)
        images_generated += 1

images_generated = 0
dataset_writer = YoloDatasetWriter(args.output_dataset_path, "val", target_width, target_height)
dataset_writer.SetOutputBoxes(args.show_boxes)
dataset_writer.SetShowUI(args.show_ui)
while images_generated < args.num_val_images:
    image, centers, boxes = generator.GenerateImage(GetNumberOfBalls(args.randomize_num_balls))
    if image is not None and boxes is not None:
        image, ratio = PadToSize(image, target_width, target_height)
        dataset_writer.SetRecomputeRatio(ratio)
        dataset_writer.AddFrame(image, boxes)
        images_generated += 1

images_generated = 0
dataset_writer = YoloDatasetWriter(args.output_dataset_path, "test", target_width, target_height)
dataset_writer.SetOutputBoxes(args.show_boxes)
dataset_writer.SetShowUI(args.show_ui)
while images_generated < args.num_test_images:
    image, centers, boxes = generator.GenerateImage(GetNumberOfBalls(args.randomize_num_balls))
    if image is not None and boxes is not None:
        image, ratio = PadToSize(image, target_width, target_height)
        dataset_writer.SetRecomputeRatio(ratio)
        dataset_writer.AddFrame(image, boxes)
        images_generated += 1

# This is a test code.
# if args.show_boxes:
#     for ball_center in centers.values():
#         box = BallBox(ball_center)
#         # temporary - to make sure it fits 640x640 input for the YOLOv5
#         box.RecomputeRatio(ratio)
#         cv2.rectangle(image, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), thickness=1)
