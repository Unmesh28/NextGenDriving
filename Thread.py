import bluetooth
import requests
from PyQt5.QtCore import QThread, pyqtSignal
from datetime import datetime
import os
import argparse
import cv2  
import numpy as np
import sys
import importlib.util
# from sympy import Point, Polygon
import time
from shapely.geometry import Polygon
# import matplotlib.pyplot as plt
import pygame

class WorkerThread (QThread):
    sig = pyqtSignal(int, int)

    # timeNow = pyqtSignal(str)
    # IsWiFi = pyqtSignal(bool)
    # IsBlue = pyqtSignal(bool)
    # # batVal = pyqtSignal(int)

    # def run(self):
    #     while (True):
    #         # val = psutil.sensors_battery().percent
    #         # self.batVal.emit(val)
    #         nowtime = datetime.now()
    #         current_time = nowtime.strftime("%H:%M")
    #         self.timeNow.emit(current_time)
    #         try:
    #             devices = bluetooth.discover_devices(lookup_names=True)
    #             number_of_devices = len(devices)
    #             if (number_of_devices == 0):
    #                 self.IsBlue.emit(False)
    #             else:
    #                 self.IsBlue.emit(True)

    #             url = "http://www.google.com"
    #             timeout = 5
    #             try:
    #                 request = requests.get(url, timeout=timeout)
    #                 self.IsWiFi.emit(True)
    #                 # print("Connected to the Internet")
    #             except (requests.ConnectionError, requests.Timeout) as exception:
    #                 self.IsWiFi.emit(False)
    #                 # print("No internet connection.")
    #         except:
    #             self.IsBlue.emit(False)


    MODEL_NAME = './Sample_TFLite_model/'
    GRAPH_NAME = 'detect.tflite'
    LABELMAP_NAME = 'labelmap.txt'
    VIDEO_NAME = './00380017.AVI'
    min_conf_threshold = float(0.5)
    use_TPU = False

    # Import TensorFlow libraries
    # If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
    # If using Coral Edge TPU, import the load_delegate library
    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
        if use_TPU:
            from tflite_runtime.interpreter import load_delegate
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        if use_TPU:
            from tensorflow.lite.python.interpreter import load_delegate

    # If using Edge TPU, assign filename for Edge TPU model
    if use_TPU:
        # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
        if (GRAPH_NAME == 'detect.tflite'):
            GRAPH_NAME = 'edgetpu.tflite'   

    # Get path to current working directory
    CWD_PATH = os.getcwd()

    # Path to video file
    VIDEO_PATH = os.path.join(CWD_PATH,VIDEO_NAME)
    # Output_video_path = os.path.join(CWD_PATH,VIDEO_NAME)

    # Path to .tflite file, which contains the model that is used for object detection
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Have to do a weird fix for label map if using the COCO "starter model" from
    # https://www.tensorflow.org/lite/models/object_detection/overview
    # First label is '???', which has to be removed.
    if labels[0] == '???':
        del(labels[0])

    # Load the Tensorflow Lite model.
    # If using Edge TPU, use special load_delegate argument
    if use_TPU:
        interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        print(PATH_TO_CKPT)
    else:
        interpreter = Interpreter(model_path=PATH_TO_CKPT)

    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Check output layer name to determine if this model was created with TF2 or TF1,
    # because outputs are ordered differently for TF2 and TF1 models
    outname = output_details[0]['name']

    if ('StatefulPartitionedCall' in outname): # This is a TF2 model
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
    else: # This is a TF1 model
        boxes_idx, classes_idx, scores_idx = 0, 1, 2

    # Open video file
    video = cv2.VideoCapture(VIDEO_PATH)
    imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    result = cv2.VideoWriter("Collision_warning_demo.avi", fourcc, 5, (1920, 1080))

    def run(self):
        print('Inside Run')
        while(self.video.isOpened()):
            print('Video Opened')
    
            # tic = time.time() 
            # Acquire frame and resize to expected shape [1xHxWx3]
            ret, frame = self.video.read()
            frame_num = self.video.get(cv2.CAP_PROP_POS_FRAMES)
            print('Frame No' + str(frame_num))
            print(self.video.get(cv2.CAP_PROP_POS_FRAMES))
            if not ret:
                print('Reached the end of the video!')
                break
            if int(frame_num)%5 == 1:
                print('Inside if')
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
                input_data = np.expand_dims(frame_resized, axis=0)

                # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
                if self.floating_model:
                    input_data = (np.float32(input_data) - self.input_mean) / self.input_std

                # Perform the actual detection by running the model with the image as input
                tic = time.time() 
                self.interpreter.set_tensor(self.input_details[0]['index'],input_data)
                self.interpreter.invoke()
                toc = time.time()
                print(toc-tic, 'seconds')

                # Retrieve detection results
                boxes = self.interpreter.get_tensor(self.output_details[self.boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
                classes = self.interpreter.get_tensor(self.output_details[self.classes_idx]['index'])[0] # Class index of detected objects
                scores = self.interpreter.get_tensor(self.output_details[self.scores_idx]['index'])[0] # Confidence of detected objects
                
                # Plot predicted trajectory (safe zone)
                # p1, p2, p3, p4 = map(Point, [(570, 1074), (856, 758), (1062, 756), (1372, 1078)])
                # poly1 = Polygon(p1, p2, p3, p4) - sympy
                # Singapore data safe zone contours
                # poly1 = Polygon([(570, 1074), (856, 758), (1062, 756), (1372, 1078)])  #shapely
                
                # Delhi night data safe zone contours
                poly1 = Polygon([(850, 1079), (1314, 734), (1456, 738), (1620, 1079)])  #shapely
                # contours = np.array([[570, 1074], [856, 758], [1062, 756], [1372, 1078]])
                print('Poly1 : ' + str(poly1))
                
                poly_critical = Polygon([(808, 1079), (1196, 792), (1424, 788), (1422, 1079)])        
                # contours = np.array([[850, 1079], [1314, 734], [1456, 738], [1620, 1079]])
                
                # cv2.fillPoly(frame, pts = [contours], color =(255,255,255))
                # cv2.imshow('Intermediate', cv2.pyrDown(frame))
                
                # Loop over all detections and draw detection box if confidence is above minimum threshold
                for i in range(len(scores)):
                    print('inside for')
                    if ((scores[i] > self.min_conf_threshold) and (scores[i] <= 1.0)):
                        print('inside if')

                        # Get bounding box coordinates and draw box
                        # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                        ymin = int(max(1,(boxes[i][0] * self.imH)))
                        xmin = int(max(1,(boxes[i][1] * self.imW)))
                        ymax = int(min(self.imH,(boxes[i][2] * self.imH)))
                        xmax = int(min(self.imW,(boxes[i][3] * self.imW)))
                        
                        # cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)
                        
                        
                        # p5, p6, p7, p8 = map(Point, [(xmin,ymin), (xmin, ymax), (xmax,ymax), (xmax,ymin)])
                        # poly2 = Polygon(p5, p6, p7, p8)
                        poly2 = Polygon([(xmin,ymin), (xmin, ymax), (xmax,ymax), (xmax,ymin)])
                        
                        # Find intersection(whether overlapping)
                        if poly1.intersects(poly2):
                            print('intersection1')
                            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0, 255, 255), 4)
                            print('Collision')
                            self.sig.emit(1, 28)
                            pygame.mixer.init()
                            pygame.mixer.music.load("beep-08b.wav")
                            pygame.mixer.music.play()
                            
                        if poly_critical.intersects(poly2):
                            print('Intersection2')
                            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0, 0, 255), 4)
                            print('Collision')
                            self.sig.emit(1, 28)
                            pygame.mixer.init()
                            pygame.mixer.music.load("beep-09.wav")
                            pygame.mixer.music.play()
        
                        # print(isIntersection)
                        
                        
                        # Draw label
                        object_name = self.labels[int(classes[i])] # Look up object name from "labels" array using class index
                        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2) # Draw label text
                        
                        # if poly1.intersection(poly2):
                        #     print('collision! Alert!')
                            
                # All the results have been drawn on the frame, so it's time to display it.
                #cv2.imshow('FCW + PCW + MBCW + ACW', cv2.pyrDown(frame))
                # plt.show()
                
                self.result.write(frame)
                
                # toc = time.time()
                # print(toc-tic, 'seconds')

                # Press 'q' to quit
                if cv2.waitKey(1) == ord('q'):
                    break

        # Clean up
        self.video.release()
        self.result.release()

        cv2.destroyAllWindows()





