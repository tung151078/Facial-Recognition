import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import streamlit as st



COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)

menu = ['Face Recognition using webcam']



#load detection model
Model = r'/home/long/Documents/Project/Facial_Recognition/yolo/yolov3-face.cfg'
WEIGHT = r'/home/long/Documents/Project/Facial_Recognition/yolo/yolov3-wider_16000.weights'

net = cv2.dnn.readNetFromDarknet(Model, WEIGHT)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#prediction model
MODEL_p = load_model('/home/long/Documents/Project/Facial_Recognition/model_1.h5')



data_dir = r'/home/long/Documents/Project/Facial_Recognition/data'
name_list = os.listdir(data_dir)
name_list.sort()
labels = {i:name_list[i] for i in range(len(name_list))}
# labels = {0: 'long', 1: 'minh', 2: 'tung'}

st.title('Face Recognition using webcam')
st.warning('Webcam show on local computer ONLY')
# show = st.checkbox('Show!')
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)
#webcam check
# if not cap.isOpened():
#     raise IOError('Cannot open webcam')

#capture frame by frame
while True:
    ret, frame = cap.read()
#detect face with yolo
    #get detection
    
    IMG_WIDTH, IMG_HEIGHT = 416, 416
    # Making blob object from original image
    blob = cv2.dnn.blobFromImage(frame, 1/255, (IMG_WIDTH, IMG_HEIGHT),
                                [0, 0, 0], 1, crop=False)

    # Set model input
    net.setInput(blob)

    # Define the layers that we want to get the outputs from
    output_layers = net.getUnconnectedOutLayersNames()

    # Run 'prediction'
    outs = net.forward(output_layers)

    #Choosing high confidence detection
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only
    # the ones with high confidence scores. Assign the box's class label as the
    # class with the highest score.
    confidences = []
    boxes = []

    # Each frame produces 3 outs corresponding to 3 output layers
    for out in outs:
    # One out has multiple predictions for multiple captured objects.
        for detection in out:
            confidence = detection[-1]
            # Extract position data of face area (only area with #high confidence)
            if confidence > 0.5:
                print(confidence)
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                
    # Find the top left point of the bounding box 
                topleft_x = int(center_x - width/2)
                topleft_y = int(center_y - height/2)
                confidences.append(float(confidence))
                boxes.append([topleft_x, topleft_y, width, height])

    # Perform non-maximum suppression to eliminate 
    # redundant overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    #Display bounding box and text
    result = frame.copy()
    final_boxes = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        final_boxes.append(box)
        # Extract position data
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        # Draw bouding box with the above measurements
        result = cv2.rectangle(frame, (left, top), (left+width, top+height), (0, 255, 0), 1)
        face_frame = result[top:top+height, left:left+width] 
        #make prediction with trained model 
        if np.sum(face_frame) != 0:
            face_frame = cv2.resize(face_frame, (224,224))
            face_frame = img_to_array(face_frame)
            face_frame = np.expand_dims(face_frame, axis = 0)
            prediction = MODEL_p.predict(face_frame)
            label = np.argmax(prediction, axis = 1)
            name = labels[int(label)]
            prop = round(max(prediction[0])*100)

            text = f'{name}, {prop}%'
        #display prediction & probability
        cv2.putText(result, text, (left, top-2), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_GREEN, thickness=2)

# Display text about number of detected faces on topleft corner
    text2 = f'Number of faces detected: {len(indices)}'
    coor2 = (20, 25)
    cv2.putText(result, text2, coor2, cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_GREEN, 2)

#showing final result
    cv2.imshow('Input', result)
    
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if cv2.waitKey(1) & 0xFF == ord('q'): #wait for 1ms
        break

cap.release()
cv2.destroyAllWindows()
    