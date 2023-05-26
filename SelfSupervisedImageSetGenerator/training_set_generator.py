import sys
import os
import math

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
    def __init__(self, center):
        self.x1 = int(center.x-center.r)  
        self.y1 = int(center.y-center.r)
        self.x2 = int(center.x+center.r)  
        self.y2 = int(center.y+center.r)
        
    def RecomputeRatio(self, ratio):
        self.x1 = int(self.x1*ratio)
        self.y1 = int(self.y1*ratio)
        self.x2 = int(self.x2*ratio)
        self.y2 = int(self.y2*ratio)
    

class DatasetWriter():
    def __init__(self, output_folder_path):
        self.output_folder_path = output_folder_path 
        if not os.path.exists(self.output_folder_path):
            os.makedirs(self.output_folder_path)
        self.frame_seq = 0

    def incrementFrameCounter(self):
        self.frame_seq = self.frame_seq+1
        
    def AddFrame(self, frame, ball_centers):
        self.incrementFrameCounter()


class SimpleDatasetWriter(DatasetWriter):
    def __init__(self, output_folder_path):
        super().__init__(output_folder_path)

    def AddFrame(self, frame, ball_centers):
        super().AddFrame(frame, ball_centers)
        file_name = f"{self.output_folder_path}/{self.frame_seq}.png" 
        cv2.imwrite(file_name, frame)
        file_name = f"{self.output_folder_path}/{self.frame_seq}.json"
        dict_json_centers = {}
        for ball_id, ball_coordinates in ball_centers.items():
            dict_json_centers[ball_id] = {"x": ball_coordinates.x, "y": ball_coordinates.y, "r": ball_coordinates.r}
        with open(file_name, 'w') as f:
            json.dump(dict_json_centers, f)


class CocoDatasetWriter(DatasetWriter):
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

    def AddFrame(self, frame, ball_centers):
        super().AddFrame(frame, ball_centers)
        labels_file_name = f"{self.labels_subfolder}/{self.frame_seq}.txt"
        with open(labels_file_name, 'w') as f:
            for ball_id, ball_coordinates in ball_centers.items():
                coco_x = float(ball_coordinates.x * self.ratio) / self.width
                coco_y = float(ball_coordinates.y* self.ratio) / self.height
                coco_width = float(ball_coordinates.r *2 * self.ratio) / self.width
                coco_height = float(ball_coordinates.r *2 * self.ratio) / self.height
                coco_entry = f"{ball_id} {coco_x} {coco_y} {coco_width} {coco_height}\n"
                f.write(coco_entry)
                if (self.output_boxes):
                    self.outputBox(frame, coco_x, coco_y, coco_width, coco_height)

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
            logging.info(f"Likely overlap for balls; skipping image")
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
            logging.info(f"A patch with ball centered at {ball_center.x}, {ball_center.y} is outside the synthethic image frame")
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
                    logging.info(f"A patch with ball centered at {ball_center.x}, {ball_center.y} is outside the synthethic image frame")
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
    '--show_boxes',
    help='Show boxes around the balls on the resulting image (debug).')


parser.add_argument(
    '--show_ui',
    help='Display the original image, binary mask, and cut image to the user.')

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


target_width = 640
target_height = 640

images_generated = 0
# dataset_writer = SimpleDatasetWriter(args.output_dataset_path) 
dataset_writer = CocoDatasetWriter(args.output_dataset_path, "train", target_width, target_height)
dataset_writer.SetOutputBoxes(args.show_boxes)
dataset_writer.SetShowUI(args.show_ui)
while images_generated < args.num_train_images:
    image, centers = generator.GenerateImage()
    if image is not None and centers is not None:
        image, ratio = PadToSize(image, target_width, target_height)
        dataset_writer.SetRecomputeRatio(ratio)
        dataset_writer.AddFrame(image, centers)
        images_generated += 1

images_generated = 0
dataset_writer = CocoDatasetWriter(args.output_dataset_path, "val", target_width, target_height)
dataset_writer.SetOutputBoxes(args.show_boxes)
dataset_writer.SetShowUI(args.show_ui)
while images_generated < args.num_val_images:
    image, centers = generator.GenerateImage()
    if image is not None and centers is not None:
        image, ratio = PadToSize(image, target_width, target_height)
        dataset_writer.SetRecomputeRatio(ratio)
        dataset_writer.AddFrame(image, centers)
        images_generated += 1

images_generated = 0
dataset_writer = CocoDatasetWriter(args.output_dataset_path, "test", target_width, target_height)
dataset_writer.SetOutputBoxes(args.show_boxes)
dataset_writer.SetShowUI(args.show_ui)
while images_generated < args.num_test_images:
    image, centers = generator.GenerateImage()
    if image is not None and centers is not None:
        image, ratio = PadToSize(image, target_width, target_height)
        dataset_writer.SetRecomputeRatio(ratio)
        dataset_writer.AddFrame(image, centers)
        images_generated += 1

# This is a test code.
# if args.show_boxes:
#     for ball_center in centers.values():
#         box = BallBox(ball_center)
#         # temporary - to make sure it fits 640x640 input for the YOLOv5
#         box.RecomputeRatio(ratio)
#         cv2.rectangle(image, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), thickness=1)
