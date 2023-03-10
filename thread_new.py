# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import pygame
from shapely.geometry import Polygon
import math
from PyQt5.QtCore import QThread, pyqtSignal, QObject, Qt
from PyQt5.QtWidgets import QApplication



# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        print('Inside Read')
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True


class NewWorkerThread (QObject):
    
    sig = pyqtSignal(int, int)
    #logging.debug('Test Logging')

    def __init__(self, parent=None):
        super().__init__(parent)
        # Set the thread's daemon attribute to True.
        #self.setDaemon(True)
        # Create the QThread object and set the Qt.WA_DeleteOnClose attribute
        self.thread = QThread(parent=self, objectName='myThread')
        self.parent().setAttribute(Qt.WA_DeleteOnClose)
        # Move the worker object to the new thread
        self.moveToThread(self.thread)
        # Connect the thread's started signal to a slot that will start the worker's task
        self.thread.started.connect(self.start_task)
        # Connect the thread's finished signal to a slot that will delete the thread
        self.thread.finished.connect(self.thread.deleteLater)
            
    def start(self):
        # Start the thread
        self.thread.start()


    
    def start_task(self):
        self.run()
        self.thread.quit()

    def stop(self):
        self.thread.stop()
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
    imW, imH = 1920, 1080 #int(resW), int(resH)
    stopped = False

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

   # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    
    # Calibration 13/01/2023
    Zone_10 = Polygon([(704, 1078), (844, 972), (972, 972), (1054, 1078)])  #shapely
    Zone_20 = Polygon([(704, 1078), (854, 914), (936, 914), (1054, 1078)])  #shapely
    Zone_30 = Polygon([(704, 1078), (864, 892), (918, 892), (1054, 1078)])  #shapely
    Zone_40 = Polygon([(704, 1078), (874, 874), (914, 874), (1054, 1078)])  #shapely
    Zone_50 = Polygon([(704, 1078), (882, 860), (908, 860), (1054, 1078)])  #shapely


    Default_zone1 = Zone_20 # Polygon([(640, 564), (559, 715), (937, 704), (708, 563)])# Zone_30
    Default_zone2 = Zone_40 # Polygon([(649, 533), (565, 715), (919, 707), (677, 533)])# Zone_60

    person_flag = False
    car_flag = False
    lane_flag = False
    speed_flag = False
    bike_flag = False

    def findImageNo(self, objectName):
        print('innside finnd image')
        print(objectName)
        print(type(objectName))
        if objectName == 'person' :
            return 5
        elif objectName == 'car' :
            return 4
        elif objectName == 'animal' :
            return 1
        else :
            return 4 


    def run(self):
        try : 
            #self.lock.acquire()
            print('Inside Run') 
            self.sig.emit(1, 37)
            # Initialize video stream
            videostream = VideoStream(resolution=(self.imW,self.imH),framerate=30).start()
            time.sleep(1)
            frame_num = 0
            
            while not self.stopped:
                    # Start timer (for calculating frame rate)
                t1 = cv2.getTickCount()

                # Grab frame from video stream
                frame1 = videostream.read()
                print(type(frame1))
                frame_num += 1
                f = open("/home/pi/NextGenDriving/GPS_speed.txt", "r")
                Speed = float(f.read())
                if Speed < 0.0:
                    continue    
                # frame_num = videostream.get(cv2.CAP_PROP_POS_FRAMES)
                # print(frame_num)

                # Acquire frame and resize to expected shape [1xHxWx3]
                frame = frame1.copy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
                input_data = np.expand_dims(frame_resized, axis=0)

                # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
                if self.floating_model:
                    input_data = (np.float32(input_data) - self.input_mean) / self.input_std

                # Perform the actual detection by running the model with the image as input
                self.interpreter.set_tensor(self.input_details[0]['index'],input_data)
                self.interpreter.invoke()

                # Retrieve detection results
                boxes = self.interpreter.get_tensor(self.output_details[self.boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
                classes = self.interpreter.get_tensor(self.output_details[self.classes_idx]['index'])[0] # Class index of detected objects
                scores = self.interpreter.get_tensor(self.output_details[self.scores_idx]['index'])[0] # Confidence of detected objects
                # print(math.ceil(int(Speed)*2/10)*10)
                
                Crit_zone1 = float(Speed)*1.5
                Crit_zone2 = float(Speed)*2
                # Crit_zone3 = math.ceil(int(Speed)*3/10)*10
                
                if Crit_zone1 < 10:
                    poly_critical = self.Zone_10
                elif Crit_zone1 > 10 and Crit_zone1 < 20:
                    poly_critical = self.Zone_20
                elif Crit_zone1 > 20 and Crit_zone1 < 30:
                    poly_critical = self.Zone_30
                elif Crit_zone1 > 30 and Crit_zone1 < 40:
                    poly_critical = self.Zone_40
                else:
                    poly_critical = self.Zone_50
                    
                if Crit_zone2 < 10:
                    poly1 = self.Zone_10
                elif Crit_zone2 > 10 and Crit_zone2 < 20:
                    poly1 = self.Zone_20
                elif Crit_zone2 > 20 and Crit_zone2 < 30:
                    poly1 = self.Zone_30
                elif Crit_zone2 > 30 and Crit_zone2 < 40:
                    poly1 = self.Zone_40
                else:
                    poly1 = self.Zone_50
                
                
                # poly1 = 'Zone_' + str(math.ceil(int(Speed)*2/10)*10)
                # exec("%s = %d" % ('Zone_' + str(math.ceil(int(Speed)*2/10)*10),2))  
                #print(poly1)
                # poly1 = Default_zone2 # Polygon([(850, 1079), (1314, 734), (1456, 738), (1620, 1079)])  #shapely
                # contours = np.array([[570, 1074], [856, 758], [1062, 756], [1372, 1078]])
                # poly_critical = 'Zone_' + str(math.ceil(int(Speed)*1.5/10)*10)  
                # poly_critical = Default_zone1 # Polygon([(808, 1079), (1196, 792), (1424, 788), (1422, 1079)])
                
                # Loop over all detections and draw detection box if confidence is above minimum threshold
                for i in range(len(scores)):
                    object_name = self.labels[int(classes[i])] # Look up object name from "labels" array using class index
                    if ((scores[i] > self.min_conf_threshold) and (scores[i] <= 1.0)):

                        # Get bounding box coordinates and draw box
                        # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                        ymin = int(max(1,(boxes[i][0] * self.imH)))
                        xmin = int(max(1,(boxes[i][1] * self.imW)))
                        ymax = int(min(self.imH,(boxes[i][2] * self.imH)))
                        xmax = int(min(self.imW,(boxes[i][3] * self.imW)))
                        
                        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                        
                        poly2 = Polygon([(xmin,ymin), (xmin, ymax), (xmax,ymax), (xmax,ymin)])
                            
                        # Find intersection(whether overlapping)
                        if poly1.intersects(poly2):
                            imageNo = self.findImageNo(object_name)
                            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0, 255, 255), 4)
                            self.sig.emit(imageNo, 60 + imageNo)
                            pygame.mixer.init()
                            pygame.mixer.music.load("beep-08b.wav")
                            pygame.mixer.music.play()
                        
                        else :
                            print('No intersection')
                            self.sig.emit(1, 1)
                            self.sig.emit(2, 2)
                            self.sig.emit(3, 3)
                            self.sig.emit(4, 4)
                            self.sig.emit(5, 5)
                                
                        if poly_critical.intersects(poly2):
                            imageNo = self.findImageNo(object_name)
                            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0, 0, 255), 4)
                            self.sig.emit(imageNo, 50 + imageNo)
                            pygame.mixer.init()
                            pygame.mixer.music.load("beep-09.wav")
                            pygame.mixer.music.play()

                        else :
                            print('No intersection')
                            self.sig.emit(1, 1)
                            self.sig.emit(2, 2)
                            self.sig.emit(3, 3)
                            self.sig.emit(4, 4)
                            self.sig.emit(5, 5)

                        # Draw label
                        
                        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                # Draw framerate in corner of frame
                cv2.putText(frame,'FPS: {0:.2f}'.format(self.frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
                frameS = cv2.resize(frame, (960, 540))  
                # All the results have been drawn on the frame, so it's time to display it.
                #cv2.imshow('FCW + PCW + MBCW + ACW', frameS)
                # cv2.imshow('FCW + PCW + MBCW + ACW', cv2.pyrDown(frame))
                # print(i)
                cv2.imwrite('/home/pi/tflite1/AI_Buddy_Field_Test/7_inch_DSI_LCD_C/28_Sep_2022/Test25/Frame'+str(frame_num).zfill(4)+'.jpg', frame)
                frame_num = frame_num+1
                # Calculate framerate
                t2 = cv2.getTickCount()
                time1 = (t2-t1)/self.freq
                self.frame_rate_calc= 1/time1

                QApplication.processEvents()

                # Press 'q' to quit
                if cv2.waitKey(1) == ord('q'):
                    break

                    
            # Clean up
            self.video.release()
            self.result.release()
            

            cv2.destroyAllWindows()
            #self.lock.release()
        except Exception as e:
            print('Exception Occured')
            print(e)