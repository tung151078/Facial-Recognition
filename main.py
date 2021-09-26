import streamlit as st
import argparse, cv2, os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from PIL import Image, ImageOps
from time import sleep
from tensorflow.keras.preprocessing import image
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--capture', type=bool, default=False, help='Save capture detection or not')
parser.add_argument('--name', type=str, default=None, help='Target name to save')
args = parser.parse_args()

MODEL = './yolo/yolov3-face.cfg'
WEIGHT = './yolo/yolov3-wider_16000.weights'
DATA_DIR= './model/data'

IMG_WIDTH, IMG_HEIGHT = 416, 416
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# Load YOLO
net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load Model
model =load_model('model/face_detect.h5')
labels = ['long','minh','tung']

def _main():
    wind_name = 'Face Detection with YOLOv3'
    cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()

        # Load image to net
        blob = cv2.dnn.blobFromImage(frame, 1/255, (224, 224), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        output_layers = net.getUnconnectedOutLayersNames()
        outs = net.forward(output_layers)

        # Remove the bounding boxes with low confidence
        boxes = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)

        # Run capture mode
        if args.capture == True:
            capture(frame, args.name, boxes)
        
        # Display prediction if there are people
        if len(boxes) != 0:
            visualize(frame, boxes, model, labels)
            

        # Display the resulting frame
        cv2.imshow(wind_name, frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            print('[i] ==> The program was stop!')
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    _main()

