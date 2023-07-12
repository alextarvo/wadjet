#
# This is a simple program that loads a video of a game, loads a trained neural net,
# detects balls on each frame of the video, draws rectangles around them, and saves the 
# resulting video.
# TODO(alexta, 2023/07/11): currently works very, very slow on CPU (~5 FPS). Could not get it
# run on CUDA - but didn't try that hard yet.
#
import sys
import os

import argparse
import logging
import json

import cv2
import numpy as np
from pathlib import Path

# --input_video_path  /home/iscander/Datasets/ground_truth/game_2023_05_28_artlight2.avi --weights_path /home/iscander/Datasets/weights/wadjet_var_16_low_100_0601/best.onnx --output_video_path  /home/iscander/Datasets/results/labeled_game_2023_05_28_artlight2.avi --show_ui True --use_cuda True

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input_video_path',
    type=str,
    help='Path to the input video.')
parser.add_argument(
    '--output_video_path',
    type=str,
    help='Path where to save annotated video.')
parser.add_argument(
    '--weights_path',
    type=str,
    help='Path to the weights of the trained Yolo network.')
parser.add_argument(
    '--show_ui',
    help='Display the original image, binary mask, and cut image to the user in the process.')
parser.add_argument(
    '--use_cuda',
    help='Use video accelerator for inference.')
args = parser.parse_args()

# Load input video with the game recording
if not Path(args.input_video_path).is_file():
    logging.fatal(f"Input video file {args.input_video_path} does not exist.")
    sys.exit(1)

# Image width and height, as expected by the trained Yolo detector.
TARGET_WIDTH = 640
TARGET_HEIGHT = 640

# Each ball will have two colored rectangles around it. 
# For solid balls color of the rectangles will be same; for striped balls - outer color is white.

# These dictionaries map ball ID to a color in the BGR format (OpenCV uses BGR by defatult).
ball_colors_inner = {
    0: (255, 255, 255),
    
    1: (0, 255, 255),
    2: (255, 0, 0),
    3: (0, 0, 250),
    4: (200, 150, 200),
    5: (0, 150, 255),
    6: (0, 120, 0),
    7: (0, 0, 128),

    8: (0, 0, 0),

    9:  (0, 255, 255),
    10: (255, 0, 0),
    11: (0, 0, 250),
    12: (200, 150, 200),
    13: (0, 150, 255),
    14: (0, 120, 0),
    15: (0, 0, 128),
}

ball_colors_outer= {
    0: (255, 255, 255),

    1: (0, 255, 255),
    2: (255, 0, 0),
    3: (0, 0, 250),
    4: (200, 150, 200),
    5: (0, 150, 255),
    6: (0, 120, 0),
    7: (0, 0, 128),

    8: (0, 0, 0),

    9: (255, 255, 255),
    10: (255, 255, 255),
    11: (255, 255, 255),
    12: (255, 255, 255),
    13: (255, 255, 255),
    14: (255, 255, 255),
    15: (255, 255, 255),
}

# Pads an image so it will match the YOLO desired definition
# TODO: copied verbgatim from training_set_generator
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


# Parses the output of the neural network, as specified by output_data, and returns the results.
# Returns three arrays, with as many entries as there were balls detected. These contain
# ball IDs, confidence values for detections, and box boundaries for the balls.
def wrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]

    image_height, image_width, _ = input_image.shape

    # The neural net output predictions for 640x640 image, padded by black.
    # These are scaling factors to re-compute the boundaries back into the resolution
    # of the input image.
    x_factor = image_width / TARGET_WIDTH
    y_factor =  image_height / (TARGET_HEIGHT /2)

    # TODO(alexta): was writing this code in a semi-conscious state, for a class.
    # Don't remember details; these should be in Yolo docs.
    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    # Call OpenCV NMS on results.
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes


# Capture the first frame of the input video to get video dimensions and FPS.
cap = cv2.VideoCapture(args.input_video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
channels = 3

video_writer = None
if args.output_video_path is not None:
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(
        args.output_video_path,
        fourcc, fps,
        (frame_height, frame_width), True)

# Read the weights of the DNN.
# Note: we had to export the Yolo weights in the format readable by the OpenCV.
net = cv2.dnn.readNet(args.weights_path)
# TODO(alexta): doesn't work. Alex built OpenCV from source, and didn't enable the CUDA support.
# Don't do like Alex. Use standard OpenCV package.
if args.use_cuda:
    print("Attempty to use CUDA")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
else:
    print("Running on CPU")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

if args.show_ui:
    cv2.namedWindow('Detected balls', cv2.WINDOW_NORMAL)

while True:
    # Read a frame from the first video
    ret, frame = cap.read()
    if not ret:
        break

    # Squeeze video so it will have width as expected by Yolo, and pad it with black.
    resized_frame, ratio = PadToSize(frame, TARGET_WIDTH, TARGET_HEIGHT)
    # Prepare the frame for OpenCV format of CNN input. normalize to [0,1] and swap Red and Blue channels
    blob = cv2.dnn.blobFromImage(resized_frame, 1/255.0, (TARGET_WIDTH, TARGET_HEIGHT), swapRB=True)
    # Detect balls using CNN.
    net.setInput(blob)
    detections = net.forward()
    class_ids, confidences, boxes = wrap_detection(frame, detections[0])

    # Draw ball bounding boxes on the frame.
    for (classid, confidence, box) in zip(class_ids, confidences, boxes):
        inner_color = ball_colors_inner[classid]
        outer_color = ball_colors_outer[classid]
        box_outer = box + [-1, -1, 2, 2]
        cv2.rectangle(frame, box_outer, outer_color, 2)
        box_inner = box + [1, 1, -2, -2]
        cv2.rectangle(frame, box_inner, inner_color, 2)

    if video_writer is not None:
        # Output to the resulting video.
        video_writer.write(frame)

    if args.show_ui:
        cv2.imshow('Detected balls', frame)
    # delay = int (1000 / fps)
    # Check for key press to exit the loop
        key = cv2.waitKey(1) 
        if key == ord('q'):
            break
    
if video_writer is not None:
    video_writer.release()
