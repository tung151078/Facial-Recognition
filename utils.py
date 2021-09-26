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

# Default colors
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)

def post_process(frame, outs, conf_threshold, nms_threshold):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        # Quét qua tất cả bounding boxes đầu ra, chỉ giữ lại box có confidence scores cao. 
        # Gán label của box  có điểm cao nhất
        confidences = []
        boxes = []
        final_boxes = []
        for out in outs:
            for detection in out:
                confidence = detection[-1]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)
                    
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant
        # overlapping boxes with lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                                nms_threshold)

        for i in indices:
            i = i[0]
            box = boxes[i]
            confidence = confidences[i]
            final_boxes.append((box, confidence))
        return final_boxes
    
def visualize(frame, boxes, model, label_dict):
    for box, conf in boxes: 
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        right = left + width
        bottom = top + height
        
        top_left = (left,top)
        bottom_right = (left + width, top + height)
       
        # Crop frame and run prediction 
        try: 
            crop_frame = frame[top:top+height, left:left+width]
            crop_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
            
            crop = img_to_array(crop_frame)
            crop = np.expand_dims(crop,axis=0)
            prediction = model.predict(crop)[0]
            name = label_dict[prediction.argmax()]
        
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,255), 2)
            text = f'{name} - {conf:.2f}'
            cv2.putText(frame, text, (left, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            
        except:
            name = 'Detecting...'
            print('Out of frame')                       

    # Draw bouding box and text
    text2 = f"Number of faces detected: {len(boxes)}"
    print(text2)
    cv2.putText(frame, text2, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
