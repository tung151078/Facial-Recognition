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

menu = ['Face Recognition using webcam', 'Face Recognition on picture']

choice = st.sidebar.selectbox('Choose source for prediction', menu)
    
if choice == 'Face Recognition using webcam':
    
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

    # def _main():
    # wind_name = 'Face Detection with YOLOv3'
    # cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)
    
    st.title('Face Recognition using webcam')
    st.warning('Webcam show on local computer ONLY')
    show = st.checkbox('Show!')
    FRAME_WINDOW = st.image([])
    
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
        # cv2.imshow(wind_name, frame)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        key = cv2.waitKey(1)
        
        
        if key == ord('q'):
            print('[i] ==> The program was stop!')
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

elif choice == 'Face Recognition on picture':
    file_uploaded = st.file_uploader("Choose the file", type = ['jpg', 'png', 'jpeg'])
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        figure = plt.figure()
        plt.axis('off')
        #-------------------------------------
        # result = predict_image(image)
        classifier =load_model('model/face_detect.h5')
        shape = ((224,224,3))
        model = tf.keras.Sequential(hub[hub.KerasLayer(classifier_model=classifier, input_shape=shape)])
        test_image = image.resize((224,224))
        test_image = preprocessing.image.img_to_array(test_image)
        test_image = test_image/255.0
        test_image = np.expand_dims(test_image, axis=0)
        class_names = ['long','tung','minh']
        predictions = model.predict(test_image)
        scores = tf.nn.softmax(predictions[0])
        scores = scores.numpy()
        image_class = class_names[np.argmax(scores)]
        result = 'The image uploaded is: {}'.format(image_class)
        #--------------------------------------
        st.write(result)
        st.pyplot(figure)
# Default colors
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)

def predict_image(image):
    classifier = tf.keras.models.load_mode('model/face_detect.h5')
    shape = ((224,224,3))
    model = tf.keras.Sequential(hub[hub.KerasLayer(classifier_model=classifier, input_shape=shape)])
    test_image = image.resize((224,224))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image/255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = ['long','tung','minh']
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    image_class = class_names[np.argmax(scores)]
    result = 'The image uploaded is: {}'.format(image_class)
    return result()

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

# if __name__ == '__main__':
#     _main()

